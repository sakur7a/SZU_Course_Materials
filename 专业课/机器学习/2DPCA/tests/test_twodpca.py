from __future__ import annotations

import unittest

import numpy as np

from twodpca.algorithms import PCARecognizer, TwoDPCARecognizer


def _build_toy_faces() -> tuple[np.ndarray, np.ndarray]:
    class_left = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    class_right = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float64,
    )
    images = np.stack(
        [
            class_left,
            class_left + 0.05,
            class_left + 0.10,
            class_right,
            class_right + 0.05,
            class_right + 0.10,
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return images, labels


class TwoDPCATestCase(unittest.TestCase):
    def test_projection_shape_and_orthogonality(self) -> None:
        images, labels = _build_toy_faces()
        model = TwoDPCARecognizer(n_components=2)
        model.fit(images, labels)

        transformed = model.transform(images[:1])
        self.assertEqual(transformed.shape, (1, 4, 2))
        gram = model.projection_axes_.T @ model.projection_axes_
        self.assertTrue(np.allclose(gram, np.eye(2), atol=1e-6))

    def test_two_dpca_recognition(self) -> None:
        images, labels = _build_toy_faces()
        train_images = images[[0, 1, 3, 4]]
        train_labels = labels[[0, 1, 3, 4]]
        test_images = images[[2, 5]]
        test_labels = labels[[2, 5]]

        model = TwoDPCARecognizer(n_components=1)
        model.fit(train_images, train_labels)
        accuracy = model.score(test_images, test_labels)
        self.assertGreaterEqual(accuracy, 1.0)

    def test_pca_recognition(self) -> None:
        images, labels = _build_toy_faces()
        train_images = images[[0, 1, 3, 4]]
        train_labels = labels[[0, 1, 3, 4]]
        test_images = images[[2, 5]]
        test_labels = labels[[2, 5]]

        model = PCARecognizer(n_components=1)
        model.fit(train_images, train_labels)
        accuracy = model.score(test_images, test_labels)
        self.assertGreaterEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()