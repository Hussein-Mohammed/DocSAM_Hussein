# CLAUDE.md — AI Assistant Guide for DocSAM

This file provides AI assistants (Claude, Copilot, etc.) with a comprehensive understanding of the DocSAM codebase to support development, debugging, and research tasks effectively.

---

## Project Overview

**DocSAM** (Document Segment Anything Model) is a CVPR 2025 research implementation of a unified document image segmentation model. It supports multiple segmentation tasks (layout, table, scene text, handwriting, ancient documents, etc.) across ~50 heterogeneous datasets via a single unified architecture.

**Paper:** *DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning* (CVPR 2025)

**Architecture summary:**
- Backbone: **Mask2Former** (Swin Transformer + pixel/transformer decoder)
- Text encoder: **Sentence-BERT** (`all-MiniLM-L6-v2`) embeds class names into semantic queries
- Training: heterogeneous mixed learning across diverse document datasets in COCO format

---

## Repository Structure

```
DocSAM_Hussein/
├── train.py                    # Main distributed training script
├── test.py                     # Testing, evaluation, and inference script (~1,817 lines)
├── models/
│   ├── DocSAM.py               # DocSAM model wrapper (187 lines)
│   └── mask2former/            # Full Mask2Former implementation
│       ├── __init__.py
│       ├── configuration_mask2former.py
│       ├── image_processing_mask2former.py
│       ├── modeling_mask2former.py         # Core model (~3,617 lines)
│       └── convert_mask2former_original_pytorch_checkpoint_to_pytorch.py
├── datasets/
│   └── dataset.py              # Custom COCO-format dataset loader (692 lines)
├── scripts/                    # Training/testing shell scripts (22 scripts + setup_assets.py)
├── figures/
│   └── DocSAM.png              # Architecture diagram
├── requirements.txt            # Python dependencies (Linux/Mac)
├── requirements.windows.txt    # Python dependencies (Windows)
├── README.md                   # Project documentation
└── MODELZOO.md                 # Pre-trained model listings and download links

# Created at runtime (excluded from git via .gitignore):
# data/                         # Datasets in COCO format
# pretrained_model/             # Backbone and DocSAM weights
# snapshots/                    # Training checkpoints
# outputs/                      # Inference results
# logs/                         # Training logs
# temp/                         # Temporary files
```

---

## Key Source Files

| File | Purpose |
|---|---|
| `models/DocSAM.py` | `DocSAM` class — wraps Mask2Former with Sentence-BERT queries, configures hyperparams |
| `models/mask2former/modeling_mask2former.py` | Full Mask2Former: pixel decoder, transformer decoder, bipartite matching, losses |
| `datasets/dataset.py` | `DocSAM_GT` — COCO-format dataset loader with patch-based sampling |
| `train.py` | Distributed training loop, LR scheduling, checkpointing, evaluation hooks |
| `test.py` | COCO metric evaluation (`evaluate_all_datasets()`), inference output saving |
| `scripts/setup_assets.py` | Downloads pretrained weights and demo dataset from HuggingFace/Google Drive |

---

## Development Environment

### Installation

```bash
conda create --name DocSAM python=3.11 -y
conda activate DocSAM
pip install -r requirements.txt          # Linux/Mac
# pip install -r requirements.windows.txt  # Windows
python scripts/setup_assets.py           # Downloads weights and demo data
```

**Python version:** 3.11+
**CUDA:** 11.8+ required for GPU training

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.5.1 | Core deep learning framework |
| `torchvision` | 0.20.1 | Image transforms and utilities |
| `transformers` | 4.49.0 | Mask2Former config and base classes |
| `sentence-transformers` | latest | Class name embedding via `all-MiniLM-L6-v2` |
| `opencv-python` | 4.10.0.84 | Image loading and processing |
| `pycocotools` | 2.0.8 | COCO format evaluation metrics |
| `einops` | 0.8.1 | Tensor rearrangement operations |
| `accelerate` | 1.1.1 | Distributed training utilities |
| `torch_dct` | 0.1.6 | DCT operations for frequency-domain processing |
| `prefetch_generator` | 1.0.3 | Async background data loading |

**Note:** `torch_xla` (TPU support) is only in `requirements.txt`, not `requirements.windows.txt`.

### Environment Variables

Set before running any script:

```bash
OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0,1        # adjust to available GPUs
# Windows only:
KMP_DUPLICATE_LIB_OK=TRUE
```

These are set automatically at the top of `train.py` and `test.py`.

---

## Running the Code

### Inference Demo

```bash
sh run_test_demo.sh
# Outputs saved to ./outputs/
```

### Training Demo

```bash
sh run_train_demo.sh
# Checkpoints saved to ./snapshots/
```

### Full Distributed Training (2 GPUs example)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --nnodes=1 --nproc_per_node=2 \
  train.py \
  --train-path ./data/dataset1 ./data/dataset2 \
  --eval-path ./data/eval_dataset \
  --model-size large \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --total-iter 4000 \
  --gpus 0,1
```

### Evaluation / Test

```bash
CUDA_VISIBLE_DEVICES=0,1 python test.py \
  --eval-path ./data/eval_dataset \
  --stage test \
  --model-size large \
  --batch-size 1 \
  --restore-from ./pretrained_model/docsam_large_all_dataset.pth
```

### Inference (no ground truth needed)

```bash
CUDA_VISIBLE_DEVICES=0,1 python test.py \
  --eval-path ./data/my_dataset \
  --stage inference \
  --model-size large \
  --save-path ./outputs/
```

---

## Important CLI Arguments

### `train.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--model-size` | `"base"` | `"base"` or `"large"` |
| `--train-path` | — | One or more paths to training datasets |
| `--eval-path` | — | One or more paths to evaluation datasets |
| `--short-range` | `704,896` | Short-side resize range during training |
| `--patch-size` | `640,640` | Patch size sampled per image |
| `--patch-num` | `1` | Patches sampled per image per iteration |
| `--keep-size` | `False` | If `True`, keeps original image size |
| `--max-num` | `10` | Max images for evaluation subset |
| `--batch-size` | `8` | Total batch size across all GPUs |
| `--learning-rate` | `1e-2` | Initial LR (use `2e-5` for fine-tuning) |
| `--weight-decay` | `5e-4` | L2 regularization (`1e-2` for fine-tuning) |
| `--lr-scheduler` | `"cosine"` | `"cosine"` or `"step"` |
| `--fine-tune` | `False` | Load `--restore-from` weights before training |
| `--restore-from` | `./snapshots/last_model.pth` | Checkpoint to restore from |
| `--snapshot-dir` | `./snapshots/` | Where to save checkpoints |
| `--total-iter` | `1000` | Total training iterations |
| `--gpus` | `"0"` | Comma-separated GPU IDs |

### `test.py` arguments (subset)

| Argument | Description |
|---|---|
| `--stage` | `"test"` (evaluate with ground truth) or `"inference"` (no GT) |
| `--restore-from` | Path to the DocSAM `.pth` checkpoint |
| `--save-path` | Output directory for inference results |

---

## Model Architecture Details

### `DocSAM` class (`models/DocSAM.py`)

- Wraps `Mask2FormerForUniversalSegmentation` from the local `models/mask2former/` module
- Loads backbone from `./pretrained_model/mask2former/facebook-mask2former-swin-{base,large}-coco-panoptic/`
- Loads sentence encoder from `./pretrained_model/sentence/all-MiniLM-L6-v2`
- Key configuration overrides:

```python
config.num_queries = 900
config.query_selection = True
config.min_num_queries = 100
config.max_num_queries = 900

config.class_weight = 1.0
config.sml1_weight  = 1.0
config.diou_weight  = 1.0
config.dice_weight  = 5.0    # dominant loss
config.focal_weight = 5.0    # dominant loss

config.encoder_layers = 6
config.decoder_layers = 4
config.num_levels = 4
config.hidden_dim = 256
config.feature_size = 256
config.mask_feature_size = 256
config.feature_strides = [4, 8, 16, 32]
```

### `forward()` batch dict keys

```python
batch = {
    "pixel_values":    Tensor,            # [B, 3, H, W]
    "pixel_mask":      Tensor,            # [B, H, W]
    "instance_masks":  List[Tensor],      # per-image instance masks
    "instance_bboxes": List[Tensor],      # per-image bboxes [N, 4]
    "instance_labels": List[Tensor],      # per-image class labels
    "semantic_masks":  List[Tensor],      # per-image semantic masks
    "class_names":     List[List[str]],   # class names per image
    "coco_datas":      Dict,              # raw COCO annotation dicts
    "image_bboxes":    Tensor,            # [B, 4] full image boxes
    "dataset_names":   List[str],         # source dataset per image
    "image_names":     List[str],         # image file names
}
```

---

## Dataset Format

All datasets must be in **COCO JSON format**.

### Expected directory layout per dataset

```
./data/my_dataset/
├── images/
│   ├── image1.jpg
│   └── ...
├── list.txt           # list of all image filenames
├── list_train.txt     # training split
├── list_val.txt       # validation split
└── instances_*.json   # COCO annotation file with segmentation masks
```

### COCO annotation structure

Standard COCO JSON with:
- `images`, `annotations`, `categories` keys
- `annotations` entries have `segmentation` (polygon or RLE), `bbox`, `category_id`, `area`

Refer to the demo dataset (download via `scripts/setup_assets.py`) for a concrete example.

---

## Training Conventions

### Distributed training

- Always launch with `torchrun` (not `python -m torch.distributed.launch`)
- Uses `nccl` backend on Linux, `gloo` on Windows
- `LOCAL_RANK` environment variable set automatically by `torchrun`
- Only rank 0 runs evaluation and saves checkpoints

### Checkpointing

- `best_model.pth` — saved when `mask_mAP` improves
- `last_model.pth` — saved every 200 iterations
- Checkpoint evaluation runs at iteration 0 and every 200 iterations

### Learning rate schedule

Two schedulers available (set via `--lr-scheduler`):

- **cosine** (default): cosine annealing with cycle length = `total_iter`
- **step**: decay by 0.1 at `[0, total_iter/2, 3*total_iter/4, total_iter]`

### Fine-tuning vs. training from scratch

- **From scratch:** `--fine-tune false` — ignores `--restore-from`, starts at iter 0
- **Fine-tuning:** `--fine-tune true` — loads matching weights from `--restore-from`

Typical fine-tuning hyperparameters: `--learning-rate 2e-5 --weight-decay 1e-2`

### Curriculum learning

Scripts in `scripts/` with `_curriculum` suffix implement progressive dataset incorporation for faster convergence. Use `run_train_large_all_dataset_curriculum.sh` as the reference for full training.

---

## Evaluation Metrics

The model is evaluated using COCO metrics computed via `pycocotools`:

- **bbox_mAP** — mean Average Precision on bounding boxes
- **mask_mAP** — mean Average Precision on segmentation masks (primary metric)
- **mask_mF1** — mean F1 score on masks
- **mIoU** — mean Intersection over Union

Best model selection uses `mask_mAP`.

### pycocotools modifications required

Standard `pycocotools` has two limitations that must be patched for this project:
1. It only shows global mAP, not per-category AP
2. It caps detections at 100 per image

See the **Pycocotools Modifications** section in `README.md` for the exact diff to apply to `coco.py` and `cocoeval.py`.

---

## Scripts Reference

Root-level scripts (quick start):

| Script | Purpose |
|---|---|
| `run_train_demo.sh` | Training demo (distributed, 2 GPUs) |
| `run_test_demo.sh` | Evaluation/inference demo |
| `run_inference_demo.sh` | Inference-only demo |
| `run_train_curriculum_large_all_dataset.sh` | Full curriculum training, large model |

`scripts/` directory (specialized runs):

| Script pattern | Purpose |
|---|---|
| `run_train_base_*.sh` | Base model training variants |
| `run_train_large_*.sh` | Large model training variants |
| `run_test_base.sh` / `run_test_large.sh` | Evaluation with base/large model |
| `run_test_large_<dataset>.sh` | Dataset-specific evaluation (doclaynet, m6doc, etc.) |
| `run_test_all_dataset_*.sh` | Parallel or sequential evaluation over all datasets |
| `nohup_run_*.sh` | Background execution wrappers |
| `setup_assets.py` | Asset download and setup |

---

## Pretrained Model Paths

All paths are relative to the project root. Models must be downloaded manually (see `README.md` and `MODELZOO.md`) or via `scripts/setup_assets.py`.

```
pretrained_model/
├── mask2former/
│   ├── facebook-mask2former-swin-base-coco-panoptic/   # base backbone
│   └── facebook-mask2former-swin-large-coco-panoptic/  # large backbone
├── sentence/
│   └── all-MiniLM-L6-v2/                              # sentence encoder
├── docsam_base_all_dataset.pth                         # base DocSAM weights
├── docsam_large_all_dataset.pth                        # large DocSAM weights
└── docsam_large_<dataset>.pth                          # dataset-specific weights
```

---

## Code Conventions

### Python style
- No external formatter config found; follow PEP 8
- Docstrings use Google style (Args/Returns sections)
- Type hints are used sparingly; don't add them to existing code unless changing it
- `sys.dont_write_bytecode = True` is set in both `train.py` and `test.py`
- `warnings.filterwarnings("ignore")` suppresses noisy warnings at import time

### Model modifications
- Override `Mask2FormerConfig` fields directly in `DocSAM.__init__()` before loading weights
- Use `ignore_mismatched_sizes=True` when calling `.from_pretrained()` to allow config changes
- Partial weight loading uses `load_state_dict(strict=False)` with explicit key matching

### Adding a new dataset
1. Convert annotations to COCO JSON format
2. Create the directory structure shown in **Dataset Format** above
3. Add the dataset path to `--train-path` or `--eval-path`
4. Ensure `class_names` in the COCO JSON `categories` list matches your intended query names

### Adding a new training script
- Copy the closest existing script from `scripts/`
- Adjust `CUDA_VISIBLE_DEVICES`, `--model-size`, `--train-path`, `--eval-path`, `--total-iter`
- Use `torchrun` with `--nproc_per_node` matching the number of GPUs

---

## Git Workflow

- Main branch: `master`
- Development branches follow the pattern `claude/<session-id>`
- Directories `data/`, `logs/`, `outputs/`, `pretrained_model/`, `snapshots/`, `temp/` are gitignored — never commit these
- Do not commit large binary files (model weights, images)

---

## Common Pitfalls

1. **Missing pretrained weights** — `DocSAM.__init__()` will crash if backbone paths don't exist. Run `scripts/setup_assets.py` first.
2. **Batch size with multi-GPU** — `--batch-size` is the *total* batch size; it is divided by the number of GPUs inside `train.py` when creating the DataLoader.
3. **`LOCAL_RANK` not set** — training crashes if not launched via `torchrun`. Do not run `train.py` with plain `python` for multi-GPU.
4. **pycocotools 100-detection limit** — per-image detection cap causes truncated metrics on dense layouts (e.g., character-level annotation). Apply the patch in `README.md`.
5. **Windows OpenMP conflict** — `KMP_DUPLICATE_LIB_OK=TRUE` must be set on Windows; it is set automatically if `os.name == 'nt'`.
6. **`torch_xla` on Windows** — not supported; use `requirements.windows.txt` which omits it.
