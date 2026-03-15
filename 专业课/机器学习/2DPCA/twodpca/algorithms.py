from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_image_tensor(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("images must have shape (n_samples, height, width)")
    return array


def _resolve_component_count(
    eigenvalues: np.ndarray,
    n_components: int | None,
    energy_ratio: float | None,
    max_components: int,
) -> int:
    if n_components is not None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        return min(int(n_components), max_components)

    if energy_ratio is None:
        return max_components

    if not 0.0 < energy_ratio <= 1.0:
        raise ValueError("energy_ratio must be in (0, 1]")

    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 1

    cumulative = np.cumsum(eigenvalues) / total
    return int(np.searchsorted(cumulative, energy_ratio) + 1)


@dataclass
class PCARecognizer:
    n_components: int | None = None
    energy_ratio: float | None = None

    def fit(self, images: np.ndarray, labels: np.ndarray) -> "PCARecognizer":
        train_images = _validate_image_tensor(images)
        if train_images.shape[0] < 2:
            raise ValueError("at least two training images are required")
        self.labels_ = np.asarray(labels)
        if train_images.shape[0] != self.labels_.shape[0]:
            raise ValueError("images and labels must have the same number of samples")

        flattened = train_images.reshape(train_images.shape[0], -1)
        self.mean_vector_ = flattened.mean(axis=0)
        centered = flattened - self.mean_vector_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

        denom = max(centered.shape[0] - 1, 1)
        eigenvalues = (singular_values**2) / denom
        max_components = vt.shape[0]
        component_count = _resolve_component_count(
            eigenvalues=eigenvalues,
            n_components=self.n_components,
            energy_ratio=self.energy_ratio,
            max_components=max_components,
        )

        self.components_ = vt[:component_count].T
        self.eigenvalues_ = eigenvalues[:component_count]
        self.explained_variance_ratio_ = self.eigenvalues_ / np.sum(eigenvalues)
        self.train_features_ = centered @ self.components_
        self.image_shape_ = train_images.shape[1:]
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        array = _validate_image_tensor(images)
        flattened = array.reshape(array.shape[0], -1)
        centered = flattened - self.mean_vector_
        return centered @ self.components_

    def predict(self, images: np.ndarray) -> np.ndarray:
        features = self.transform(images)
        predictions = []
        for feature in features:
            distances = np.linalg.norm(self.train_features_ - feature, axis=1)
            predictions.append(self.labels_[np.argmin(distances)])
        return np.asarray(predictions)

    def score(self, images: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(images)
        labels_array = np.asarray(labels)
        return float(np.mean(predictions == labels_array))

    def reconstruct(self, images: np.ndarray, n_components: int | None = None) -> np.ndarray:
        array = _validate_image_tensor(images)
        flattened = array.reshape(array.shape[0], -1)
        centered = flattened - self.mean_vector_

        if n_components is None:
            component_count = self.components_.shape[1]
        else:
            component_count = min(max(int(n_components), 1), self.components_.shape[1])

        basis = self.components_[:, :component_count]
        projected = centered @ basis
        reconstructed = projected @ basis.T + self.mean_vector_
        return reconstructed.reshape(array.shape)


@dataclass
class TwoDPCARecognizer:
    n_components: int | None = None
    energy_ratio: float | None = None
    distance_metric: str = "column_l2_sum"

    def fit(self, images: np.ndarray, labels: np.ndarray) -> "TwoDPCARecognizer":
        train_images = _validate_image_tensor(images)
        if train_images.shape[0] < 2:
            raise ValueError("at least two training images are required")
        self.labels_ = np.asarray(labels)
        if train_images.shape[0] != self.labels_.shape[0]:
            raise ValueError("images and labels must have the same number of samples")

        self.mean_image_ = train_images.mean(axis=0)
        centered = train_images - self.mean_image_
        scatter = np.mean(np.transpose(centered, (0, 2, 1)) @ centered, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        component_count = _resolve_component_count(
            eigenvalues=eigenvalues,
            n_components=self.n_components,
            energy_ratio=self.energy_ratio,
            max_components=eigenvectors.shape[1],
        )

        self.projection_axes_ = eigenvectors[:, :component_count]
        self.eigenvalues_ = eigenvalues[:component_count]
        self.explained_variance_ratio_ = self.eigenvalues_ / np.sum(eigenvalues)
        self.train_features_ = centered @ self.projection_axes_
        self.image_shape_ = train_images.shape[1:]
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        array = _validate_image_tensor(images)
        centered = array - self.mean_image_
        return centered @ self.projection_axes_

    def _distances(self, features: np.ndarray) -> np.ndarray:
        if self.distance_metric == "frobenius":
            return np.linalg.norm(self.train_features_ - features, axis=(1, 2))
        if self.distance_metric == "column_l2_sum":
            return np.linalg.norm(self.train_features_ - features, axis=1).sum(axis=1)
        raise ValueError(f"unsupported distance metric: {self.distance_metric}")

    def predict(self, images: np.ndarray) -> np.ndarray:
        features = self.transform(images)
        predictions = []
        for feature in features:
            distances = self._distances(feature)
            predictions.append(self.labels_[np.argmin(distances)])
        return np.asarray(predictions)

    def score(self, images: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(images)
        labels_array = np.asarray(labels)
        return float(np.mean(predictions == labels_array))

    def reconstruct(self, images: np.ndarray, n_components: int | None = None) -> np.ndarray:
        array = _validate_image_tensor(images)
        centered = array - self.mean_image_

        if n_components is None:
            component_count = self.projection_axes_.shape[1]
        else:
            component_count = min(max(int(n_components), 1), self.projection_axes_.shape[1])

        basis = self.projection_axes_[:, :component_count]
        projected = centered @ basis
        reconstructed = projected @ basis.T + self.mean_image_
        return reconstructed