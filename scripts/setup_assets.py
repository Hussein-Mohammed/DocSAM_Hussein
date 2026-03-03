#!/usr/bin/env python3
"""
One-shot asset setup for DocSAM.

What it does:
1) Creates required folders under ./pretrained_model and ./data.
2) Downloads Hugging Face dependencies:
   - facebook/mask2former-swin-base-coco-panoptic -> ./pretrained_model/mask2former/facebook-mask2former-swin-base-coco-panoptic
   - facebook/mask2former-swin-large-coco-panoptic -> ./pretrained_model/mask2former/facebook-mask2former-swin-large-coco-panoptic
   - sentence-transformers/all-MiniLM-L6-v2       -> ./pretrained_model/sentence/all-MiniLM-L6-v2
3) Downloads DocSAM checkpoints (Google Drive) into ./pretrained_model.
4) Downloads demo dataset (Google Drive) into ./data/demo_data (auto-extracts if archive).

Usage:
    python scripts/setup_assets.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Iterable

import requests
from huggingface_hub import snapshot_download

HF_REPOS = {
    "facebook/mask2former-swin-base-coco-panoptic": "pretrained_model/mask2former/facebook-mask2former-swin-base-coco-panoptic",
    "facebook/mask2former-swin-large-coco-panoptic": "pretrained_model/mask2former/facebook-mask2former-swin-large-coco-panoptic",
    "sentence-transformers/all-MiniLM-L6-v2": "pretrained_model/sentence/all-MiniLM-L6-v2",
}

# From MODELZOO.md and README.md
DOCSAM_GDRIVE_FILES = {
    "docsam_base_all_dataset.pth": "1M7Zc63eBGTKjynkmQ-2sI3m2WBvzqMO4",
    "docsam_large_all_dataset.pth": "1YvdDMtnDpfZTyxF59z3gbK2FKrfHaxJn",
    "docsam_large_all_dataset_keepsize.pth": "15X9lLhPmQdDXnVQiuzB8G6kU13X2bBtP",
    "docsam_large_m6doc.pth": "1RhBTSIY8aox9WJyVTBDrfeldA7GEH-hu",
    "docsam_large_doclaynet.pth": "1pOmu6FkZK9j6f1KrGL33W9hGRmCxkYz4",
    "docsam_large_scut_cab.pth": "1pR9Z9UPASLgGEdgq4SiT3CqNdtI_ZhEa",
    "docsam_large_ctw1500.pth": "1kO6Pyk4h36fnVRzVQJ5cScfFkFMchqOd",
    "docsam_large_totaltext.pth": "1aiN_C9eHC0fRX27CPRu8Jfaz5DQiz83D",
}

DEMO_DATASET_FILE_ID = "1gvfco5zyRDASGO2BOYCjZuRxbN7MHhsT"


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _extract_confirm_token(html: str) -> str | None:
    patterns = [
        r'confirm=([0-9A-Za-z_\-]+)',
        r'name="confirm"\s+value="([0-9A-Za-z_\-]+)"',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return None


def download_google_drive_file(file_id: str, dst_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a Google Drive file by ID with large-file confirm token handling."""
    if dst_path.exists() and dst_path.stat().st_size > 0:
        print(f"[skip] exists: {dst_path}")
        return

    url = "https://drive.google.com/uc?export=download"
    with requests.Session() as session:
        resp = session.get(url, params={"id": file_id}, stream=True, timeout=120)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        is_html = "text/html" in content_type or "text/plain" in content_type
        if is_html:
            token = _extract_confirm_token(resp.text)
            if token:
                resp = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=120)
                resp.raise_for_status()

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(dst_path)
        print(f"[ok] downloaded: {dst_path}")


def maybe_extract_archive(archive_path: Path, target_dir: Path) -> bool:
    if not archive_path.exists():
        return False

    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        return True

    if suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        try:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(target_dir)
            return True
        except tarfile.ReadError:
            return False

    return False


def download_hf_assets(root: Path) -> None:
    for repo_id, rel_dir in HF_REPOS.items():
        local_dir = root / rel_dir
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"[skip] HF repo already present: {local_dir}")
            continue

        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] snapshot_download({repo_id}) -> {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
        print(f"[ok] HF repo ready: {local_dir}")


def download_docsam_checkpoints(root: Path) -> None:
    dst_dir = root / "pretrained_model"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in DOCSAM_GDRIVE_FILES.items():
        dst_path = dst_dir / filename
        try:
            download_google_drive_file(file_id=file_id, dst_path=dst_path)
        except Exception as exc:
            print(f"[warn] failed checkpoint {filename}: {exc}")


def download_demo_dataset(root: Path) -> None:
    data_root = root / "data"
    demo_dir = data_root / "demo_data"
    tmp_dir = data_root / "_downloads"
    ensure_dirs([data_root, demo_dir, tmp_dir])

    archive_path = tmp_dir / "demo_data_download"
    try:
        download_google_drive_file(file_id=DEMO_DATASET_FILE_ID, dst_path=archive_path)
    except Exception as exc:
        print(f"[warn] failed demo dataset download: {exc}")
        return

    extracted = maybe_extract_archive(archive_path, tmp_dir)
    if extracted:
        print(f"[ok] extracted demo dataset archive: {archive_path}")
        candidates = [p for p in tmp_dir.iterdir() if p.is_dir() and p.name != "demo_data"]
        if len(candidates) == 1:
            src = candidates[0]
            # Merge/copy into demo_dir
            for item in src.iterdir():
                dst = demo_dir / item.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
        else:
            # If structure is unknown, keep extracted files in tmp and just inform user.
            print("[warn] extracted structure was not unique; please inspect ./data/_downloads manually.")
    else:
        # Could already be a folder-less format; keep file and inform user.
        print(f"[warn] download is not a recognized archive: {archive_path}. Please inspect manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DocSAM assets into standard project folders.")
    parser.add_argument("--root", type=str, default=".", help="Project root directory (default: current directory).")
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face model downloads.")
    parser.add_argument("--skip-docsam", action="store_true", help="Skip DocSAM checkpoints download.")
    parser.add_argument("--skip-demo-data", action="store_true", help="Skip demo dataset download.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ensure_dirs(
        [
            root / "pretrained_model",
            root / "pretrained_model" / "mask2former",
            root / "pretrained_model" / "sentence",
            root / "data",
        ]
    )

    print(f"[info] using root: {root}")

    if not args.skip_hf:
        download_hf_assets(root)
    if not args.skip_docsam:
        download_docsam_checkpoints(root)
    if not args.skip_demo_data:
        download_demo_dataset(root)

    print("[done] setup_assets completed.")


if __name__ == "__main__":
    main()
