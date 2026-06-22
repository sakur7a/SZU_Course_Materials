"""
Unified results I/O for traditional face recognition experiments.
Handles saving/loading experiment results in CSV and JSON formats.
"""

import csv
import json
import os
import time as _time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Track which CSV files have been cleared in the current overwrite session.
# Key: (experiment, dataset) — reset by clear_overwrite_session().
_overwrite_cleared: set[tuple[str, str]] = set()

# Fixed field schema for all result rows
FIELDNAMES = [
    "dataset",
    "experiment",
    "feature",
    "feature_params",
    "preprocess",
    "classifier",
    "classifier_params",
    "augmentation",
    "fold",
    "train_acc",
    "test_acc",
    "rank5_acc",
    "macro_f1",
    "precision",
    "recall",
    "fps",
    "train_time",
    "feature_dim",
]


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_results(experiment: str, dataset: str):
    """Delete CSV file for a given experiment+dataset before re-running."""
    fname = RESULTS_DIR / f"{experiment}_{dataset}.csv"
    if fname.exists():
        fname.unlink()


def _flatten_row(row: dict) -> dict:
    """Ensure all complex fields are serialized to strings for CSV."""
    flat = {}
    for k in FIELDNAMES:
        val = row.get(k, "")
        if isinstance(val, (dict, list)):
            flat[k] = json.dumps(val, ensure_ascii=False)
        elif val is None:
            flat[k] = ""
        else:
            flat[k] = val
    return flat


def clear_overwrite_session():
    """Reset the overwrite tracking state. Call at the start of a new experiment run."""
    _overwrite_cleared.clear()


def save_result(row: dict, experiment: str, dataset: str, mode: str = "append"):
    """Save a single result row. mode='append' or 'overwrite'.

    'overwrite' deletes the existing CSV once per (experiment, dataset) per session,
    so multiple calls in the same run accumulate rows without duplicate headers.
    Call clear_overwrite_session() at the start of each experiment run.
    """
    ensure_results_dir()
    fname = RESULTS_DIR / f"{experiment}_{dataset}.csv"
    row = _flatten_row(row)
    row["experiment"] = experiment
    row["dataset"] = dataset
    key = (experiment, dataset)
    if mode == "overwrite" and key not in _overwrite_cleared:
        if fname.exists():
            fname.unlink()
        _overwrite_cleared.add(key)
    write_header = not fname.exists()
    with open(fname, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def save_results(rows: list[dict], experiment: str, dataset: str, mode: str = "append"):
    """Save multiple result rows."""
    for row in rows:
        save_result(row, experiment, dataset, mode=mode)


def load_results(experiment: str, dataset: str) -> list[dict]:
    """Load results from a CSV file."""
    fname = RESULTS_DIR / f"{experiment}_{dataset}.csv"
    if not fname.exists():
        return []
    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_all_results(experiment: str) -> list[dict]:
    """Load results for an experiment across all datasets."""
    rows = []
    for ds in ["FERET", "Yale", "Olivetti"]:
        rows.extend(load_results(experiment, ds))
    return rows


def save_summary(summary: dict, experiment: str, dataset: str):
    """Save a JSON summary (e.g., best configs, aggregated stats)."""
    ensure_results_dir()
    fname = RESULTS_DIR / f"summary_{experiment}_{dataset}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


def load_summary(experiment: str, dataset: str) -> dict:
    """Load a JSON summary."""
    fname = RESULTS_DIR / f"summary_{experiment}_{dataset}.json"
    if not fname.exists():
        return {}
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)


def save_final_report(report: dict):
    """Save the final cross-experiment report."""
    ensure_results_dir()
    fname = RESULTS_DIR / "final_report.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


def list_experiments() -> list[str]:
    """List all experiments that have saved results."""
    ensure_results_dir()
    experiments = set()
    for f in RESULTS_DIR.glob("*.csv"):
        parts = f.stem.split("_", 1)
        if len(parts) >= 2:
            experiments.add(parts[0])
    return sorted(experiments)
