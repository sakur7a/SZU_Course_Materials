from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twodpca.datasets import load_olivetti_faces_dataset, load_orl_faces_dataset
from twodpca.experiments import run_component_sweep, run_train_size_experiment


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _plot_train_size_summary(summary, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for method in summary["method"].unique():
        subset = summary[summary["method"] == method]
        plt.errorbar(
            subset["train_per_class"],
            subset["mean_accuracy"],
            yerr=subset["std_accuracy"],
            marker="o",
            capsize=4,
            label=method,
        )
    plt.xticks(sorted(summary["train_per_class"].unique()))
    plt.xlabel("Training images per class")
    plt.ylabel("Recognition accuracy")
    plt.title("2DPCA vs PCA")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_component_summary(summary, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["components"],
        summary["mean_accuracy"],
        yerr=summary["std_accuracy"],
        marker="o",
        capsize=4,
    )
    plt.xticks(sorted(summary["components"].unique()))
    plt.xlabel("Projection axes")
    plt.ylabel("Recognition accuracy")
    plt.title("2DPCA component sweep")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the main 2DPCA experiments.")
    parser.add_argument("--dataset", choices=["olivetti", "orl"], default="olivetti")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--train-sizes", type=str, default="1,2,3,4,5")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--twodpca-components", type=int, default=8)
    parser.add_argument("--sweep-train-per-class", type=int, default=3)
    parser.add_argument("--sweep-components", type=str, default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "olivetti":
        dataset = load_olivetti_faces_dataset(shuffle=False, random_state=args.random_state)
    else:
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required when --dataset=orl")
        dataset = load_orl_faces_dataset(args.dataset_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sizes = _parse_int_list(args.train_sizes)
    sweep_components = _parse_int_list(args.sweep_components)

    train_size_result = run_train_size_experiment(
        images=dataset.images,
        labels=dataset.labels,
        train_sizes=train_sizes,
        repeats=args.repeats,
        pca_components=args.pca_components,
        twodpca_components=args.twodpca_components,
        random_state=args.random_state,
    )
    train_size_result.frame.to_csv(output_dir / "train_size_runs.csv", index=False)
    train_size_result.summary.to_csv(output_dir / "train_size_summary.csv", index=False)
    _plot_train_size_summary(train_size_result.summary, output_dir / "train_size_summary.png")

    component_result = run_component_sweep(
        images=dataset.images,
        labels=dataset.labels,
        train_per_class=args.sweep_train_per_class,
        component_values=sweep_components,
        repeats=args.repeats,
        random_state=args.random_state,
    )
    component_result.frame.to_csv(output_dir / "component_runs.csv", index=False)
    component_result.summary.to_csv(output_dir / "component_summary.csv", index=False)
    _plot_component_summary(component_result.summary, output_dir / "component_summary.png")

    summary = {
        "dataset": dataset.description,
        "num_images": int(dataset.images.shape[0]),
        "image_shape": [int(dataset.images.shape[1]), int(dataset.images.shape[2])],
        "num_classes": int(len(dataset.class_names)),
        "train_sizes": train_sizes,
        "repeats": args.repeats,
        "pca_components": args.pca_components,
        "twodpca_components": args.twodpca_components,
        "sweep_train_per_class": args.sweep_train_per_class,
        "sweep_components": sweep_components,
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("Saved results to:", output_dir.resolve())


if __name__ == "__main__":
    main()