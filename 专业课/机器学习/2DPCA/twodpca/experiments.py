from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .algorithms import PCARecognizer, TwoDPCARecognizer


@dataclass
class ExperimentResult:
    frame: pd.DataFrame
    summary: pd.DataFrame


def _split_train_test_by_class(
    labels: np.ndarray,
    train_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size <= train_per_class:
            raise ValueError(
                f"class {class_id} has {class_indices.size} samples, which is not enough for train_per_class={train_per_class}"
            )
        shuffled = rng.permutation(class_indices)
        train_indices.extend(shuffled[:train_per_class].tolist())
        test_indices.extend(shuffled[train_per_class:].tolist())

    return np.asarray(train_indices, dtype=np.int64), np.asarray(test_indices, dtype=np.int64)


def run_train_size_experiment(
    images: np.ndarray,
    labels: np.ndarray,
    train_sizes: list[int],
    repeats: int,
    pca_components: int,
    twodpca_components: int,
    random_state: int = 0,
) -> ExperimentResult:
    rng = np.random.default_rng(random_state)
    records: list[dict[str, float | int | str]] = []

    for repeat in range(repeats):
        for train_per_class in train_sizes:
            train_index, test_index = _split_train_test_by_class(labels, train_per_class, rng)
            train_images = images[train_index]
            train_labels = labels[train_index]
            test_images = images[test_index]
            test_labels = labels[test_index]

            models = {
                "PCA": PCARecognizer(n_components=pca_components),
                "2DPCA": TwoDPCARecognizer(n_components=twodpca_components),
            }

            for method_name, model in models.items():
                model.fit(train_images, train_labels)
                accuracy = model.score(test_images, test_labels)
                records.append(
                    {
                        "repeat": repeat,
                        "train_per_class": train_per_class,
                        "method": method_name,
                        "accuracy": accuracy,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    summary = frame.groupby(["train_per_class", "method"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
    )
    return ExperimentResult(frame=frame, summary=summary)


def run_component_sweep(
    images: np.ndarray,
    labels: np.ndarray,
    train_per_class: int,
    component_values: list[int],
    repeats: int,
    random_state: int = 0,
) -> ExperimentResult:
    rng = np.random.default_rng(random_state)
    records: list[dict[str, float | int | str]] = []

    for repeat in range(repeats):
        train_index, test_index = _split_train_test_by_class(labels, train_per_class, rng)
        train_images = images[train_index]
        train_labels = labels[train_index]
        test_images = images[test_index]
        test_labels = labels[test_index]

        for component_count in component_values:
            model = TwoDPCARecognizer(n_components=component_count)
            model.fit(train_images, train_labels)
            accuracy = model.score(test_images, test_labels)
            records.append(
                {
                    "repeat": repeat,
                    "components": component_count,
                    "method": "2DPCA",
                    "accuracy": accuracy,
                }
            )

    frame = pd.DataFrame.from_records(records)
    summary = frame.groupby(["components", "method"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
    )
    return ExperimentResult(frame=frame, summary=summary)