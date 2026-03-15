from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces


@dataclass
class DatasetBundle:
    images: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    description: str


def _read_grayscale_image(image_path: Path, image_size: tuple[int, int] | None) -> np.ndarray:
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        if image_size is not None:
            grayscale = grayscale.resize(image_size, Image.Resampling.BILINEAR)
        return np.asarray(grayscale, dtype=np.float64)


def load_orl_faces_dataset(
    dataset_root: str | Path,
    image_size: tuple[int, int] | None = None,
) -> DatasetBundle:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset path does not exist: {root}")

    subject_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not subject_dirs:
        raise ValueError("dataset path must contain one subdirectory per subject")

    images: list[np.ndarray] = []
    labels: list[int] = []
    class_names: list[str] = []
    extensions = {".pgm", ".png", ".jpg", ".jpeg", ".bmp"}

    for label, subject_dir in enumerate(subject_dirs):
        image_paths = sorted(
            path for path in subject_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions
        )
        if not image_paths:
            continue

        class_names.append(subject_dir.name)
        for image_path in image_paths:
            images.append(_read_grayscale_image(image_path, image_size=image_size))
            labels.append(label)

    if not images:
        raise ValueError("no supported image files were found in the dataset path")

    return DatasetBundle(
        images=np.stack(images, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=class_names,
        description="Local ORL-style face dataset",
    )


def load_olivetti_faces_dataset(shuffle: bool = False, random_state: int = 0) -> DatasetBundle:
    dataset = fetch_olivetti_faces(shuffle=shuffle, random_state=random_state)
    images = dataset.images.astype(np.float64) * 255.0
    labels = dataset.target.astype(np.int64)
    class_names = [f"subject_{index:02d}" for index in np.unique(labels)]
    return DatasetBundle(
        images=images,
        labels=labels,
        class_names=class_names,
        description="scikit-learn Olivetti faces dataset",
    )