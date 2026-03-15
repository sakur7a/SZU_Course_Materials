from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twodpca.algorithms import PCARecognizer, TwoDPCARecognizer
from twodpca.datasets import load_olivetti_faces_dataset, load_orl_faces_dataset


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def _clip_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 255.0)


def _save_panel(
    originals: np.ndarray,
    reconstructions: dict[str, dict[int, np.ndarray]],
    k_values: list[int],
    output_path: Path,
) -> None:
    methods = list(reconstructions.keys())
    n_rows = len(originals)
    n_cols = 1 + len(methods) * len(k_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n_rows):
        axes[row, 0].imshow(_clip_to_uint8(originals[row]), cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title("Original")
        axes[row, 0].axis("off")

        col = 1
        for method in methods:
            for k in k_values:
                axes[row, col].imshow(
                    _clip_to_uint8(reconstructions[method][k][row]),
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                axes[row, col].set_title(f"{method} k={k}")
                axes[row, col].axis("off")
                col += 1

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_progress_gif(
    original: np.ndarray,
    recon_dict: dict[int, np.ndarray],
    k_values: list[int],
    output_path: Path,
    title_prefix: str,
) -> None:
    frames: list[Image.Image] = []
    for k in k_values:
        recon = _clip_to_uint8(recon_dict[k])
        canvas = Image.new("L", (original.shape[1] * 2, original.shape[0]), color=0)
        left = Image.fromarray(_clip_to_uint8(original).astype(np.uint8))
        right = Image.fromarray(recon.astype(np.uint8))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (original.shape[1], 0))

        canvas = canvas.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 20), fill=(0, 0, 0))
        draw.text((6, 4), f"{title_prefix}: original | reconstruction (k={k})", fill=(255, 255, 255))
        frames.append(canvas)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reconstruction demo with increasing k.")
    parser.add_argument("--dataset", choices=["olivetti", "orl"], default="orl")
    parser.add_argument("--dataset-path", type=str, default="data/att_faces_raw")
    parser.add_argument("--output-dir", type=str, default="results_orl")
    parser.add_argument("--sample-indices", type=str, default="0,55,110")
    parser.add_argument("--k-values", type=str, default="1,2,4,6,8,12,16")
    parser.add_argument("--max-pca-components", type=int, default=120)
    parser.add_argument("--max-twodpca-components", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "olivetti":
        dataset = load_olivetti_faces_dataset(shuffle=False, random_state=args.random_state)
    else:
        dataset = load_orl_faces_dataset(args.dataset_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = _parse_int_list(args.sample_indices)
    k_values = sorted(_parse_int_list(args.k_values))
    max_k = max(k_values)

    samples = dataset.images[sample_indices]

    pca = PCARecognizer(n_components=max(max_k, args.max_pca_components))
    pca.fit(dataset.images, dataset.labels)

    twodpca = TwoDPCARecognizer(n_components=max(max_k, args.max_twodpca_components))
    twodpca.fit(dataset.images, dataset.labels)

    pca_recons: dict[int, np.ndarray] = {}
    twodpca_recons: dict[int, np.ndarray] = {}
    for k in k_values:
        pca_recons[k] = pca.reconstruct(samples, n_components=k)
        twodpca_recons[k] = twodpca.reconstruct(samples, n_components=k)

    panel_path = output_dir / "reconstruction_progression.png"
    _save_panel(
        originals=samples,
        reconstructions={"PCA": pca_recons, "2DPCA": twodpca_recons},
        k_values=k_values,
        output_path=panel_path,
    )

    pca_gif_path = output_dir / "reconstruction_pca.gif"
    twodpca_gif_path = output_dir / "reconstruction_2dpca.gif"
    first_original = samples[0]
    _save_progress_gif(
        original=first_original,
        recon_dict={k: pca_recons[k][0] for k in k_values},
        k_values=k_values,
        output_path=pca_gif_path,
        title_prefix="PCA",
    )
    _save_progress_gif(
        original=first_original,
        recon_dict={k: twodpca_recons[k][0] for k in k_values},
        k_values=k_values,
        output_path=twodpca_gif_path,
        title_prefix="2DPCA",
    )

    print("Saved reconstruction demo to:", panel_path.resolve())
    print("Saved PCA gif to:", pca_gif_path.resolve())
    print("Saved 2DPCA gif to:", twodpca_gif_path.resolve())


if __name__ == "__main__":
    main()