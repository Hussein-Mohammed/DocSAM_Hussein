"""
Simplified text region detection and visualization using DocSAM.

Detects and visualises text regions in document/scene images.
The model uses text-encoded class names as semantic queries, so different
text-related class names guide the model to find distinct region types
(e.g. body text vs. titles vs. captions).

Usage example:
    python -B -u detect_text.py \
        --eval-path ./data/demo_data/ \
        --restore-from ./pretrained_model/docsam_large_all_dataset.pth \
        --model-size large \
        --save-path ./outputs/text_regions/ \
        --max-num 100 \
        --short-range 704,896 \
        --patch-size 640,640 \
        --patch-num 1 \
        --keep-size False \
        --gpus 0
"""

import os
import sys
from pathlib import Path

# Windows runtime guard: normalize DLL search paths before importing torch.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1")

    env_prefix = Path(sys.prefix)
    preferred = [
        env_prefix / "Library" / "bin",
        env_prefix / "DLLs",
        env_prefix / "Lib" / "site-packages" / "torch" / "lib",
    ]

    path_items = []
    for item in os.environ.get("PATH", "").split(os.pathsep):
        low = item.lower()
        if "anaconda3" in low and "library\\bin" in low and not str(env_prefix).lower() in low:
            continue
        path_items.append(item)

    for dll_dir in reversed([str(d) for d in preferred if d.exists()]):
        path_items.insert(0, dll_dir)
        try:
            os.add_dll_directory(dll_dir)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    os.environ["PATH"] = os.pathsep.join(path_items)

import timeit
import random
import argparse
import math
import json
import copy
import pickle
import numpy as np

np.set_printoptions(linewidth=400, precision=4)

import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torchvision.ops import box_iou
from torch.nn.utils.rnn import pad_sequence
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

import cv2
from PIL import Image as PILImage
from typing import Dict, List, Tuple
import pycocotools.mask as mask_utils

from datasets.dataset import DocSAM_GT
from models.DocSAM import DocSAM

import torch.multiprocessing as mp

# ── Default constants ──────────────────────────────────────────────────────────

SHORT_RANGE = (704, 896)
PATCH_SIZE = (640, 640)
PATCH_NUM = 1
KEEP_SIZE = False
MAX_NUM = 10
BATCH_SIZE = 1
MODEL_SIZE = "base"
SAVE_PATH = "./outputs/text_regions/"
RESTORE_FROM = "./pretrained_model/docsam_large_all_dataset.pth"
GPU_IDS = "0"

# Text-focused class names.  The model encodes these via Sentence-BERT and uses
# the resulting embeddings as semantic queries, so each distinct name guides the
# model to look for a different *type* of text region.
#
# "_background_" must always be the last entry.
DEFAULT_TEXT_CLASSES = [
    "text",
    "title",
    "list",
    "caption",
    "_background_",
]


# ── Argument parsing ──────────────────────────────────────────────────────────

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_tuple(s):
    try:
        t = tuple(map(int, s.split(",")))
        if len(t) != 2:
            raise ValueError
        return t
    except ValueError:
        raise argparse.ArgumentTypeError("Input must be two integers separated by a comma (e.g. '704,896')")


def get_arguments():
    p = argparse.ArgumentParser(description="DocSAM – text region detection")
    p.add_argument("--model-size", type=str, default=MODEL_SIZE)
    p.add_argument("--eval-path", type=str, nargs="+", required=True, help="Directories with images to process")
    p.add_argument("--save-path", type=str, default=SAVE_PATH)
    p.add_argument("--short-range", type=parse_tuple, default=SHORT_RANGE)
    p.add_argument("--patch-size", type=parse_tuple, default=PATCH_SIZE)
    p.add_argument("--patch-num", type=int, default=PATCH_NUM)
    p.add_argument("--keep-size", type=str2bool, default=KEEP_SIZE)
    p.add_argument("--max-num", type=int, default=MAX_NUM)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--restore-from", type=str, default=RESTORE_FROM)
    p.add_argument("--gpus", type=str, default=GPU_IDS)
    p.add_argument(
        "--text-classes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Custom text class names for detection (without _background_). "
            "Default: text title list caption"
        ),
    )
    p.add_argument("--score-threshold", type=float, default=0.3,
                    help="Minimum confidence score for displaying detections (default: 0.3)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_path(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.collate_fn = getattr(dataset, "collate_fn", None)


# ── Visualisation helpers ─────────────────────────────────────────────────────


# ── Geometric classification (orientation + scale) ────────────────────────────

def classify_orientation(mask_np):
    """Classify a binary mask's orientation using its minimum-area rotated rect.

    Returns (orientation_label, angle_degrees):
        orientation_label: one of "horizontal", "vertical", "tilted", "flipped"
        angle_degrees: the raw angle from minAreaRect (for the label on the viz)

    cv2.minAreaRect returns an angle in [-90, 0).  We normalise so that:
        - 0° means the longer side is horizontal
        - 90° means the longer side is vertical

    "flipped" is detected when the rotated rect is nearly 180° from horizontal
    (i.e. text that is upside-down).  Since minAreaRect can't truly distinguish
    upside-down from right-side-up (it has no notion of reading direction), we
    approximate: if the angle is very close to ±180° we call it "flipped".  In
    practice this means almost-horizontal regions where the rect happened to
    snap to the 180° equivalent — genuinely upside-down text requires OCR to
    confirm, so treat "flipped" as a *hint*.
    """
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown", 0.0

    # Merge all contour points
    pts = np.concatenate(contours)
    if len(pts) < 5:
        return "unknown", 0.0

    rect = cv2.minAreaRect(pts)          # ((cx,cy), (w,h), angle)
    (w, h), angle = rect[1], rect[2]

    # Normalise: make angle represent deviation of the *longer* side from horizontal
    if w < h:
        angle = angle + 90.0             # rotate so longer side drives the angle

    # angle is now in roughly [0, 180)
    angle = angle % 180.0

    if angle > 170 or angle < 10:
        label = "horizontal"
    elif 80 < angle < 100:
        label = "vertical"
    else:
        label = "tilted"

    return label, round(angle, 1)


def classify_scale(mask_np, image_shape):
    """Classify a mask's scale relative to the image.

    Returns (scale_label, area_ratio):
        scale_label: "small", "medium", or "large"
        area_ratio:  mask_area / image_area
    """
    mask_area = float(mask_np.sum())
    image_area = float(image_shape[0] * image_shape[1])
    ratio = mask_area / max(image_area, 1.0)

    if ratio < 0.01:
        label = "small"
    elif ratio < 0.10:
        label = "medium"
    else:
        label = "large"

    return label, round(ratio, 5)


def classify_region(mask_np, image_shape):
    """Return a combined descriptor like 'small tilted' and raw measurements."""
    orient, angle = classify_orientation(mask_np)
    scale, ratio = classify_scale(mask_np, image_shape)
    descriptor = f"{scale} {orient}"
    return {
        "descriptor": descriptor,
        "orientation": orient,
        "angle": angle,
        "scale": scale,
        "area_ratio": ratio,
    }


def random_palette(n):
    """Return a flat list of 3*n random RGB values; index 0 is black (background)."""
    pal = [0, 0, 0]
    for _ in range(n - 1):
        pal.extend([random.randint(60, 255) for _ in range(3)])
    return pal


def id_map_to_color(id_map, palette):
    h, w = id_map.shape
    colour = np.zeros((h, w, 3), np.uint8)
    for i in range(id_map.max() + 1):
        colour[id_map == i] = palette[3 * i: 3 * i + 3]
    return colour


def bimask_to_id_mask(bimasks):
    out = np.zeros(bimasks.shape[1:], np.int32)
    for i, m in enumerate(bimasks):
        out[m] = i + 1
    return out


# ── Mask / bbox utilities ────────────────────────────────────────────────────

def mask_bbox(masks, img_shape=None, is_norm=False):
    Q, H, W = masks.shape
    x_proj = masks.any(dim=-2)
    y_proj = masks.any(dim=-1)
    x1 = (x_proj.cumsum(dim=-1) == 1).float().argmax(dim=-1)
    x2 = W - 1 - (x_proj.flip([-1]).cumsum(dim=-1) == 1).float().argmax(dim=-1)
    y1 = (y_proj.cumsum(dim=-1) == 1).float().argmax(dim=-1)
    y2 = H - 1 - (y_proj.flip([-1]).cumsum(dim=-1) == 1).float().argmax(dim=-1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).float()
    if is_norm:
        bboxes[..., [0, 2]] /= W
        bboxes[..., [1, 3]] /= H
    elif img_shape is not None:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / W * img_shape[-1]
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / H * img_shape[-2]
    return bboxes


def mask_iou(masks1, masks2, scale_factor=None):
    if scale_factor is not None:
        masks1 = F.interpolate(masks1[None].float(), scale_factor=scale_factor, mode="bilinear", align_corners=False)[0]
        masks2 = F.interpolate(masks2[None].float(), scale_factor=scale_factor, mode="bilinear", align_corners=False)[0]
    masks1 = masks1.float().flatten(1)
    masks2 = masks2.float().flatten(1)
    num = torch.matmul(masks1, masks2.T)
    den = masks1.sum(-1)[:, None] + masks2.sum(-1)[None, :] - num
    return num / den.clamp(min=1)


def non_max_suppression(masks, scores, threshold=0.5):
    bboxes = mask_bbox(masks)
    bbox_ious = box_iou(bboxes, bboxes).cpu().numpy()
    order = torch.argsort(scores, descending=True).cpu().numpy()
    keep = []
    while np.size(order) > 0:
        keep.append(order[0])
        ious = bbox_ious[order[0], order[1:]]
        flags = ious > 0.01
        if flags.sum() > 0:
            m_ious = mask_iou(masks[order[0]][None], masks[order[1:][flags]])
            ious[flags] = m_ious[0].cpu().numpy()
        order = order[1:][ious < threshold]
    return keep


def non_max_suppression_multiclass(masks, bboxes, labels, scores, threshold=0.5):
    all_keep = []
    for label in labels.unique():
        idx = [i.item() for i in (labels == label).nonzero(as_tuple=False)]
        keep = non_max_suppression(masks[idx], scores[idx], threshold)
        all_keep += [idx[k] for k in keep]
    return masks[all_keep], bboxes[all_keep], labels[all_keep], scores[all_keep]


# ── Post-processing ──────────────────────────────────────────────────────────

def post_process_instance_segmentation(
    pred_instance_masks, pred_instance_bboxes, pred_instance_labels,
    pred_semantic_masks, oriimg_size, target_size,
    threshold_prob=0.01, threshold_nms=0.25,
):
    scores, labels = pred_instance_labels.softmax(dim=-1)[:, :-1].max(dim=-1)
    idx = scores >= threshold_prob
    if idx.sum() == 0:
        idx[0] = True
    pred_instance_masks = pred_instance_masks[idx]
    pred_instance_bboxes = pred_instance_bboxes[idx]
    pred_instance_labels = pred_instance_labels[idx]

    idx = (pred_instance_masks.sigmoid() > 0.5).sum(dim=[1, 2]) > 0
    if idx.sum() == 0:
        idx[0] = True
    pred_instance_masks = pred_instance_masks[idx]
    pred_instance_bboxes = pred_instance_bboxes[idx]
    pred_instance_labels = pred_instance_labels[idx]

    logits = F.interpolate(pred_instance_masks[None], size=target_size, mode="bilinear", align_corners=False)[0].sigmoid()
    bin_masks = logits > 0.5
    mask_scores = torch.stack([(s * m).sum() / max(m.sum(), 1e-6) for s, m in zip(logits, bin_masks)])

    scores, labels = pred_instance_labels.softmax(dim=-1)[:, :-1].max(dim=-1)
    scores = scores * mask_scores
    labels = labels + 1  # 0 → background

    pred_instance_bboxes[:, [0, 2]] = pred_instance_bboxes[:, [0, 2]] / oriimg_size[1] * target_size[1]
    pred_instance_bboxes[:, [1, 3]] = pred_instance_bboxes[:, [1, 3]] / oriimg_size[0] * target_size[0]
    pred_instance_bboxes[:, 2] -= pred_instance_bboxes[:, 0]
    pred_instance_bboxes[:, 3] -= pred_instance_bboxes[:, 1]

    sem_logits = F.interpolate(pred_semantic_masks[None], size=target_size, mode="bilinear", align_corners=False)[0].sigmoid()
    sem_masks = sem_logits > 0.5

    bin_masks, pred_instance_bboxes, labels, scores = non_max_suppression_multiclass(
        bin_masks, pred_instance_bboxes, labels, scores, threshold=threshold_nms)

    return [bin_masks, pred_instance_bboxes, labels, scores, sem_masks]


def get_instance_segmentation_results(seg_results, image_bbox, class_names):
    instance_masks, instance_bboxes, instance_labels, instance_scores, semantic_masks = seg_results
    x1, y1, x2, y2 = image_bbox
    instance_masks = instance_masks[:, y1:y2, x1:x2]
    semantic_masks = semantic_masks[:, y1:y2, x1:x2]
    instance_bboxes[:, 0] -= x1
    instance_bboxes[:, 1] -= y1

    # Sort by area (largest first)
    areas = instance_masks.sum(dim=[1, 2])
    order = areas.argsort(descending=True)
    instance_masks = instance_masks[order]
    instance_bboxes = instance_bboxes[order]
    instance_scores = instance_scores[order]
    instance_labels = instance_labels[order]

    semantic_masks = semantic_masks[: len(class_names) - 1]

    return {
        "instance_maskes": instance_masks,
        "instance_bboxes": instance_bboxes,
        "instance_scores": instance_scores,
        "instance_labels": instance_labels,
        "semantic_maskes": semantic_masks,
    }


# ── Sliding window inference ─────────────────────────────────────────────────

def sliding_window_crop(image, mask, patch_size):
    C, H, W = image.size()
    pixel_values = [F.interpolate(image[None], size=patch_size, mode="area")[0]]
    pixel_mask = [F.interpolate(mask[None], size=patch_size, mode="nearest")[0]]
    patch_bboxes = [[0, 0, W, H]]
    image_bboxes = [[0, 0, patch_size[1], patch_size[0]]]

    stride_v, stride_h = patch_size[0] // 2, patch_size[1] // 2
    rows = int(math.ceil((H - patch_size[0]) / stride_v) + 1)
    cols = int(math.ceil((W - patch_size[1]) / stride_h) + 1)

    for r in range(rows):
        for c in range(cols):
            y1 = int(r * stride_v)
            y2 = min(y1 + patch_size[0], H)
            y1 = max(y2 - patch_size[0], 0)
            x1 = int(c * stride_h)
            x2 = min(x1 + patch_size[1], W)
            x1 = max(x2 - patch_size[1], 0)
            pixel_values.append(image[:, y1:y2, x1:x2])
            pixel_mask.append(mask[:, y1:y2, x1:x2])
            patch_bboxes.append([x1, y1, x2, y2])
            image_bboxes.append([0, 0, x2 - x1, y2 - y1])

    return (
        torch.stack(pixel_values),
        torch.stack(pixel_mask),
        torch.tensor(patch_bboxes).long(),
        torch.tensor(image_bboxes).long(),
    )


def predict_with_dynamic_batch_size(model, pixel_values, pixel_mask, image_bboxes, class_names, batch_size=64):
    patch_num = pixel_values.shape[0]
    batch_size = min(batch_size, patch_num)

    while batch_size >= 1:
        try:
            inst_masks, bbox_preds, cate_preds, sem_masks = [], [], [], []
            n_batches = math.ceil(patch_num / batch_size)
            for i in range(n_batches):
                print(f"  batch {i + 1}/{n_batches}  (bs={batch_size})")
                s, e = i * batch_size, min((i + 1) * batch_size, patch_num)
                preds = model.mask2former(
                    pixel_values=pixel_values[s:e],
                    pixel_mask=pixel_mask[s:e],
                    image_bboxes=image_bboxes[s:e],
                    class_names=class_names[s:e],
                )
                inst_masks += [x[0] for x in preds.transformer_decoder_instance_masks[-1].split(1)]
                bbox_preds += [x[0] for x in preds.transformer_decoder_bbox_predictions[-1].split(1)]
                cate_preds += [x[0] for x in preds.transformer_decoder_cate_predictions[-1].split(1)]
                sem_masks += [x[0] for x in preds.transformer_decoder_semantic_masks[-1].split(1)]

            return (
                pad_sequence(inst_masks, batch_first=True, padding_value=-1e10),
                pad_sequence(bbox_preds, batch_first=True, padding_value=0),
                pad_sequence(cate_preds, batch_first=True, padding_value=0),
                pad_sequence(sem_masks, batch_first=True, padding_value=-1e10),
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at bs={batch_size}, halving")
            torch.cuda.empty_cache()
            batch_size = math.ceil(batch_size / 2)


def predict_slide_window(model, batch, patch_size):
    all_pv, all_pm, all_pb, all_ib, all_cn = [], [], [], [], []
    img_patch_nums = []

    for image, mask, cn in zip(batch["pixel_values"], batch["pixel_mask"], batch["class_names"]):
        pv, pm, pb, ib = sliding_window_crop(image, mask, patch_size)
        all_pv.append(pv)
        all_pm.append(pm)
        all_pb.append(pb)
        all_ib.append(ib)
        all_cn += [cn] * pv.shape[0]
        img_patch_nums.append(pv.shape[0])

    all_pv = torch.cat(all_pv)
    all_pm = torch.cat(all_pm)
    all_ib = torch.cat(all_ib)

    all_inst, all_bbox, all_cate, all_sem = predict_with_dynamic_batch_size(
        model, all_pv, all_pm, all_ib, all_cn, batch_size=16)

    all_inst = all_inst.split(img_patch_nums)
    all_bbox = all_bbox.split(img_patch_nums)
    all_cate = all_cate.split(img_patch_nums)
    all_sem = all_sem.split(img_patch_nums)

    batch_results = []
    for n in range(len(batch["pixel_values"])):
        C, H, W = batch["pixel_values"][n].shape
        inst = all_inst[n]
        bbox = all_bbox[n]
        cate = all_cate[n]
        sem = all_sem[n]
        pb = all_pb[n]
        count_map = torch.ones(1, H, W).to(sem.device)

        for p in range(inst.shape[0]):
            tgt = (H, W) if p == 0 else patch_size
            seg = post_process_instance_segmentation(
                inst[p], bbox[p], cate[p], sem[p], oriimg_size=patch_size, target_size=tgt)
            if p == 0:
                seg[3] = seg[3] * 0.75
                seg[4] = seg[4].float()
                batch_results.append(seg)
            else:
                i_masks, i_bboxes, i_labels, i_scores, s_masks = seg
                bbs = mask_bbox(i_masks)
                boundary = (bbs[:, 0] < 4) | (bbs[:, 1] < 4) | (bbs[:, 2] > patch_size[1] - 4) | (bbs[:, 3] > patch_size[0] - 4)
                i_scores[boundary] *= 0.5

                x1, y1, x2, y2 = pb[p]
                i_masks = F.pad(i_masks, (x1, W - x2, y1, H - y2), "constant", 0)
                i_bboxes[:, 0] += x1
                i_bboxes[:, 1] += y1

                batch_results[n][0] = torch.cat((batch_results[n][0], i_masks))
                batch_results[n][1] = torch.cat((batch_results[n][1], i_bboxes))
                batch_results[n][2] = torch.cat((batch_results[n][2], i_labels))
                batch_results[n][3] = torch.cat((batch_results[n][3], i_scores))
                batch_results[n][4][:, y1:y2, x1:x2] += s_masks.float()
                count_map[:, y1:y2, x1:x2] += 1

                batch_results[n][0], batch_results[n][1], batch_results[n][2], batch_results[n][3] = \
                    non_max_suppression_multiclass(
                        batch_results[n][0], batch_results[n][1], batch_results[n][2], batch_results[n][3], threshold=0.5)

        batch_results[n][4] = (batch_results[n][4] / count_map) > 0.5
        batch_results[n] = get_instance_segmentation_results(
            batch_results[n], image_bbox=batch["image_bboxes"][n], class_names=batch["class_names"][n])

    return batch_results


# ── Model loading ─────────────────────────────────────────────────────────────

def load_weights(model, path):
    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except pickle.UnpicklingError:
        print("Retrying with weights_only=False for:", path)
        state = torch.load(path, weights_only=False, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    cur = model.state_dict()
    matched = {k: v for k, v in state.items() if k in cur and cur[k].size() == v.size()}
    unmatched = [k for k in cur if k not in matched]
    if unmatched:
        print(f"  {len(unmatched)} unmatched keys (expected for text-encoder / query projection)")
    model.load_state_dict(matched, strict=False)
    print("Pretrained model loaded:", path)
    return model


# ── Main inference loop ──────────────────────────────────────────────────────

def run_inference(args, model, dataloader, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)
    model.eval()
    device = next(model.parameters()).device

    score_thr = args.score_threshold

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            torch.cuda.empty_cache()

            batch["pixel_values"] = [v.cuda(device) for v in batch["pixel_values"]] if isinstance(
                batch["pixel_values"], list) else batch["pixel_values"].cuda(device)
            batch["pixel_mask"] = [v.cuda(device) for v in batch["pixel_mask"]] if isinstance(
                batch["pixel_mask"], list) else batch["pixel_mask"].cuda(device)

            batch_results = predict_slide_window(model, batch, patch_size=args.patch_size)

            for i, results in enumerate(batch_results):
                image_name = batch["image_names"][i]
                dataset_name = batch["dataset_names"][i]
                class_names = batch["class_names"][i]
                x1, y1, x2, y2 = batch["image_bboxes"][i]
                pixel_values = batch["pixel_values"][i][:, y1:y2, x1:x2]

                n_kept = (results['instance_scores'] >= score_thr).sum().item()
                print(f"  {image_name}: {n_kept} text regions (>={score_thr})")

                # ── Classify each region and save as JSONL ───────────────
                img_h, img_w = pixel_values.shape[1], pixel_values.shape[2]
                detections = []
                for j in range(results["instance_maskes"].size(0)):
                    label_idx = results["instance_labels"][j].item()
                    score = results["instance_scores"][j].item()
                    cat_name = class_names[label_idx - 1]
                    mask_np = results["instance_maskes"][j].cpu().numpy().astype(np.uint8)
                    geo = classify_region(mask_np, (img_h, img_w))
                    rle = mask_utils.encode(np.asfortranarray(mask_np))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox = results["instance_bboxes"][j].cpu().numpy().tolist()
                    detections.append({
                        "category": cat_name,
                        "descriptor": geo["descriptor"],
                        "orientation": geo["orientation"],
                        "angle": geo["angle"],
                        "scale": geo["scale"],
                        "area_ratio": geo["area_ratio"],
                        "bbox": bbox,
                        "segmentation": rle,
                        "score": round(score, 4),
                    })

                out_dir = os.path.join(args.save_path, dataset_name)
                jsonl_path = os.path.join(out_dir, image_name + "_text_detections.jsonl")
                make_path(jsonl_path)
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for det in detections:
                        if det["score"] >= score_thr:
                            f.write(json.dumps(det, ensure_ascii=False) + "\n")

                # ── Visualisations ────────────────────────────────────────
                image = pixel_values.permute(1, 2, 0).cpu().numpy().astype(np.uint8)[:, :, ::-1]
                image = np.ascontiguousarray(image)

                # Save original
                orig_path = os.path.join(out_dir, image_name + ".jpg")
                make_path(orig_path)
                PILImage.fromarray(image).save(orig_path)

                # -- Instance detection overlay --
                inst_palette = random_palette(5000)
                sem_palette = random_palette(len(class_names) + 2)

                vis = image.copy()
                kept = results["instance_scores"] >= score_thr
                if kept.sum() > 0:
                    inst_id = bimask_to_id_mask(results["instance_maskes"][kept].cpu().numpy())
                    colours = id_map_to_color(inst_id, inst_palette)
                    vis[inst_id > 0] = vis[inst_id > 0] // 2 + colours[inst_id > 0] // 2

                for j, det in enumerate(detections):
                    score = results["instance_scores"][j].item()
                    if score < score_thr:
                        continue
                    label_idx = results["instance_labels"][j].item()
                    bbox = results["instance_bboxes"][j].cpu().numpy().tolist()
                    bx1, by1 = int(bbox[0]), int(bbox[1])
                    bx2, by2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                    colour = sem_palette[label_idx * 3: label_idx * 3 + 3]
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), color=colour, thickness=2)
                    # Show: category | descriptor | score
                    label_text = f"{det['category']} | {det['descriptor']} | {score:.2f}"
                    cv2.putText(vis, label_text, (bx1, by2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)

                det_path = os.path.join(out_dir, image_name + "_text_instances.png")
                make_path(det_path)
                PILImage.fromarray(vis).save(det_path)

                # -- Semantic overlay --
                vis_sem = image.copy()
                sem_id = bimask_to_id_mask(results["semantic_maskes"].cpu().numpy())
                sem_colours = id_map_to_color(sem_id, sem_palette)
                vis_sem[sem_id > 0] = vis_sem[sem_id > 0] // 2 + sem_colours[sem_id > 0] // 2
                sem_path = os.path.join(out_dir, image_name + "_text_semantic.png")
                make_path(sem_path)
                PILImage.fromarray(vis_sem).save(sem_path)

                # -- Category overlay (each class gets its own colour) --
                vis_cat = image.copy()
                cat_map = torch.zeros(results["instance_maskes"].shape[-2:]).long()
                for j in range(results["instance_maskes"].size(0)):
                    if results["instance_scores"][j] >= score_thr:
                        cat_map[results["instance_maskes"][j] == 1] = results["instance_labels"][j]
                cat_colours = id_map_to_color(cat_map.numpy(), sem_palette)
                vis_cat[cat_map.numpy() > 0] = vis_cat[cat_map.numpy() > 0] // 2 + cat_colours[cat_map.numpy() > 0] // 2
                cat_path = os.path.join(out_dir, image_name + "_text_categories.png")
                make_path(cat_path)
                PILImage.fromarray(vis_cat).save(cat_path)

    model.train()
    torch.cuda.empty_cache()


# ── Custom dataset wrapper that overrides class names to text-only ───────────

class TextDetectionDataset(DocSAM_GT):
    """Thin wrapper that forces class_names to text-focused categories."""

    def __init__(self, *a, text_classes=None, **kw):
        super().__init__(*a, **kw)
        self._text_classes = text_classes or DEFAULT_TEXT_CLASSES

    def __getitem__(self, idx):
        samples = super().__getitem__(idx)
        for sample in samples:
            sample["class_names"] = list(self._text_classes)
            # Prefix the background class with the dataset name (expected by the model)
            if sample["class_names"][-1] == "_background_":
                sample["class_names"][-1] = sample["dataset_names"] + " _background_"
        return samples


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = timeit.default_timer()
    mp.set_start_method("spawn", force=True)

    args = get_arguments()

    if args.gpus != "None":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Resolve text classes
    if args.text_classes:
        text_classes = list(args.text_classes) + ["_background_"]
    else:
        text_classes = list(DEFAULT_TEXT_CLASSES)

    print(f"Text classes: {text_classes[:-1]}")

    model = DocSAM(model_size=args.model_size)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model parameters: {params_m:.1f}M")

    if os.path.isfile(args.restore_from):
        model = load_weights(model, args.restore_from)

    for data_path in args.eval_path:
        print(f"\nProcessing: {data_path}")
        ds = TextDetectionDataset(
            [data_path],
            short_range=args.short_range,
            patch_size=args.patch_size,
            patch_num=args.patch_num,
            keep_size=args.keep_size,
            stage="inference",
            text_classes=text_classes,
        )
        ds = CustomSubset(ds, range(min(args.max_num, len(ds))))
        loader = DataLoaderX(ds, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=ds.collate_fn)
        run_inference(args, model, loader, gpu_id=0)

    elapsed = timeit.default_timer() - start
    print(f"\nDone in {elapsed:.1f}s")
