"""Prepare and validate dataset files for the face recognition experiments.

This repository intentionally does not include dataset files. The script can
create the Olivetti `.npy` files from scikit-learn and verify whether FERET and
Yale files have been placed in the expected locations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FERET_DIR = DATA_DIR / "数据库-feret_k175_s7_w80_h80" / "feret_k175_s7_w80_h80"
YALE_IMAGES = DATA_DIR / "yale_images.npy"
YALE_LABELS = DATA_DIR / "yale_labels.npy"
OLIVETTI_IMAGES = DATA_DIR / "olivetti_images.npy"
OLIVETTI_LABELS = DATA_DIR / "olivetti_labels.npy"


def prepare_olivetti() -> None:
    """Download Olivetti via scikit-learn and save arrays as uint8 images."""
    import cv2
    from sklearn.datasets import fetch_olivetti_faces

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_olivetti_faces(data_home=str(DATA_DIR), shuffle=False)
    images = (dataset.images * 255).astype(np.uint8)
    images = np.array([cv2.resize(img, (64, 64)) for img in images], dtype=np.uint8)
    labels = dataset.target.astype(np.int64)
    np.save(OLIVETTI_IMAGES, images)
    np.save(OLIVETTI_LABELS, labels)
    print(f"[OK] Saved {OLIVETTI_IMAGES.relative_to(ROOT)}: {images.shape}")
    print(f"[OK] Saved {OLIVETTI_LABELS.relative_to(ROOT)}: {labels.shape}")


def check_file(path: Path, label: str) -> bool:
    if path.exists():
        print(f"[OK] {label}: {path.relative_to(ROOT)}")
        return True
    print(f"[MISS] {label}: {path.relative_to(ROOT)}")
    return False


def check_datasets() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Dataset checklist")
    print("=================")
    if FERET_DIR.exists():
        count = len(list(FERET_DIR.glob("*.bmp")))
        print(f"[OK] FERET BMP directory: {FERET_DIR.relative_to(ROOT)} ({count} files)")
    else:
        print(f"[MISS] FERET BMP directory: {FERET_DIR.relative_to(ROOT)}")
    check_file(YALE_IMAGES, "Yale images")
    check_file(YALE_LABELS, "Yale labels")
    check_file(OLIVETTI_IMAGES, "Olivetti images")
    check_file(OLIVETTI_LABELS, "Olivetti labels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--olivetti", action="store_true", help="create Olivetti npy files")
    parser.add_argument("--check", action="store_true", help="check expected dataset files")
    args = parser.parse_args()

    if args.olivetti:
        prepare_olivetti()
    if args.check or not args.olivetti:
        check_datasets()


if __name__ == "__main__":
    main()
