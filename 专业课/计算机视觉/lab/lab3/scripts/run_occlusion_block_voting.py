"""
Occlusion-Aware Block Voting Recognition
=========================================
Compares Global LBP vs Block Voting vs Weighted Block Voting
under clean / 10% / 20% / 30% occlusion.

Usage:
    uv run python scripts/run_occlusion_block_voting.py
    uv run python scripts/run_occlusion_block_voting.py --dataset FERET
    uv run python scripts/run_occlusion_block_voting.py --quick
"""

import argparse
import os
import sys
import time
import warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from face_recognition_system import LBPExtractor, load_feret_dataset
from scripts.traditional_configs import DATASET_CONFIGS, RANDOM_SEED
from scripts.results_io import save_result, save_summary, clear_results

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
FERET_DIR = os.path.join(DATA_DIR, "数据库-feret_k175_s7_w80_h80", "feret_k175_s7_w80_h80")
IMG_SIZE = (80, 80)


def load_all_datasets(selected=None):
    datasets = {}
    ds_list = selected or ["FERET", "Yale", "Olivetti"]

    if "FERET" in ds_list:
        print("  Loading FERET...")
        imgs, labels = load_feret_dataset(FERET_DIR)
        if imgs is not None:
            datasets["FERET"] = (imgs, labels)
            print(f"    {len(imgs)} images, {len(np.unique(labels))} people")

    if "Yale" in ds_list:
        print("  Loading Yale...")
        yale_imgs = np.load(os.path.join(DATA_DIR, "yale_images.npy"))
        yale_lbls = np.load(os.path.join(DATA_DIR, "yale_labels.npy"))
        if yale_imgs.ndim == 2:
            yale_imgs = yale_imgs.reshape(-1, *IMG_SIZE)
        datasets["Yale"] = (yale_imgs.astype(np.uint8), yale_lbls)
        print(f"    {len(yale_imgs)} images, {len(np.unique(yale_lbls))} people")

    if "Olivetti" in ds_list:
        print("  Loading Olivetti...")
        oli_imgs = np.load(os.path.join(DATA_DIR, "olivetti_images.npy"))
        oli_lbls = np.load(os.path.join(DATA_DIR, "olivetti_labels.npy"))
        if oli_imgs.ndim == 2:
            oli_imgs = oli_imgs.reshape(-1, *IMG_SIZE)
        if oli_imgs.shape[1] != IMG_SIZE[0] or oli_imgs.shape[2] != IMG_SIZE[1]:
            resized = np.stack([cv2.resize(oli_imgs[i], IMG_SIZE) for i in range(len(oli_imgs))])
            oli_imgs = resized
        datasets["Olivetti"] = (oli_imgs.astype(np.uint8), oli_lbls)
        print(f"    {len(oli_imgs)} images, {len(np.unique(oli_lbls))} people")

    return datasets


def apply_random_occlusion(images, ratio, seed=42):
    rng = np.random.RandomState(seed)
    h, w = IMG_SIZE
    occ_h, occ_w = int(h * ratio), int(w * ratio)
    occluded = []
    for img in images:
        img_copy = img.copy()
        y = rng.randint(0, h - occ_h + 1)
        x = rng.randint(0, w - occ_w + 1)
        img_copy[y:y+occ_h, x:x+occ_w] = 0
        occluded.append(img_copy)
    return np.array(occluded)


def extract_block_features(images, grid=4):
    """Batch-extract LBP features per block. Returns (n_imgs, grid*grid, feat_dim)."""
    h, w = IMG_SIZE
    bh, bw = h // grid, w // grid
    n_imgs = len(images)
    n_blocks = grid * grid

    # Crop all blocks into flat array
    all_block_imgs = np.empty((n_imgs * n_blocks, bh, bw), dtype=images.dtype)
    idx = 0
    for img in images:
        for bi in range(grid):
            for bj in range(grid):
                all_block_imgs[idx] = img[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw]
                idx += 1

    # Single LBP call for all blocks
    ext = LBPExtractor(grid_x=2, grid_y=2)
    all_feats = ext.fit_transform(all_block_imgs)
    return all_feats.reshape(n_imgs, n_blocks, -1)


def extract_global_features(images):
    ext = LBPExtractor(grid_x=3, grid_y=3)
    return ext.fit_transform(images)


def global_classify(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance")
    clf.fit(X_tr_s, y_tr)
    return clf.predict(X_te_s)


def block_vote_classify(blocks_tr, y_tr, blocks_te, method="majority"):
    """Block-level voting. method: 'majority' or 'weighted'."""
    n_train, n_blocks, feat_dim = blocks_tr.shape
    n_test = blocks_te.shape[0]

    scaler = StandardScaler()
    scaler.fit(blocks_tr.reshape(-1, feat_dim))

    y_pred = []
    for i in range(n_test):
        votes = {}
        for b in range(n_blocks):
            test_feat = scaler.transform(blocks_te[i, b:b+1, :])
            train_feats = scaler.transform(blocks_tr[:, b, :])
            dists = pairwise_distances(test_feat, train_feats, metric="cosine")[0]
            nn_idx = np.argmin(dists)
            nn_label = y_tr[nn_idx]

            if method == "majority":
                votes[nn_label] = votes.get(nn_label, 0) + 1
            elif method == "weighted":
                weight = 1.0 / (dists[nn_idx] + 1e-6)
                votes[nn_label] = votes.get(nn_label, 0) + weight

        y_pred.append(max(votes, key=votes.get) if votes else y_tr[0])
    return np.array(y_pred)


def block_vote_reject_classify(blocks_tr, y_tr, blocks_te, reject_ratio=0.2):
    """Block voting with rejection of top-k most distant blocks."""
    n_train, n_blocks, feat_dim = blocks_tr.shape
    n_test = blocks_te.shape[0]
    n_keep = max(1, int(n_blocks * (1 - reject_ratio)))

    scaler = StandardScaler()
    scaler.fit(blocks_tr.reshape(-1, feat_dim))

    y_pred = []
    for i in range(n_test):
        block_dists = []
        block_labels = []
        for b in range(n_blocks):
            test_feat = scaler.transform(blocks_te[i, b:b+1, :])
            train_feats = scaler.transform(blocks_tr[:, b, :])
            dists = pairwise_distances(test_feat, train_feats, metric="cosine")[0]
            nn_idx = np.argmin(dists)
            block_labels.append(y_tr[nn_idx])
            block_dists.append(dists[nn_idx])

        keep_idx = np.argsort(block_dists)[:n_keep]
        votes = {}
        for j in keep_idx:
            votes[block_labels[j]] = votes.get(block_labels[j], 0) + 1
        y_pred.append(max(votes, key=votes.get) if votes else y_tr[0])
    return np.array(y_pred)


def run_cv(images, labels, n_folds, method, grid=4, occ_ratio=0.0, seed=RANDOM_SEED):
    unique, counts = np.unique(labels, return_counts=True)
    actual_folds = min(n_folds, min(counts))
    if actual_folds < 2:
        actual_folds = 2

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for tr_idx, te_idx in cv.split(images, labels):
        X_tr, y_tr = images[tr_idx], labels[tr_idx]
        X_te, y_te = images[te_idx], labels[te_idx]

        if occ_ratio > 0:
            X_te = apply_random_occlusion(X_te, occ_ratio, seed=seed)

        if method == "global":
            feat_tr = extract_global_features(X_tr)
            feat_te = extract_global_features(X_te)
            y_pred = global_classify(feat_tr, y_tr, feat_te)
        else:
            blocks_tr = extract_block_features(X_tr, grid=grid)
            blocks_te = extract_block_features(X_te, grid=grid)
            if method == "block_vote":
                y_pred = block_vote_classify(blocks_tr, y_tr, blocks_te, "majority")
            elif method == "weighted_vote":
                y_pred = block_vote_classify(blocks_tr, y_tr, blocks_te, "weighted")
            elif method == "reject_vote":
                y_pred = block_vote_reject_classify(blocks_tr, y_tr, blocks_te, 0.2)

        fold_accs.append(accuracy_score(y_te, y_pred))
    return fold_accs


METHODS = {
    "global_lbp": {"method": "global", "label": "Global LBP"},
    "block_vote_lbp": {"method": "block_vote", "label": "Block Vote (LBP)"},
    "weighted_vote_lbp": {"method": "weighted_vote", "label": "Weighted Block Vote (LBP)"},
    "reject_vote_lbp": {"method": "reject_vote", "label": "Reject-TopK Vote (LBP)"},
}

OCCLUSION_LEVELS = [0.0, 0.1, 0.2, 0.3]


def run_experiment(datasets, quick=False):
    print("\n" + "=" * 60)
    print("  Occlusion-Aware Block Voting Recognition")
    print("=" * 60)

    occ_levels = [0.0, 0.1] if quick else OCCLUSION_LEVELS
    grid = 3 if quick else 4

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n{'='*60}")
        print(f"  {ds_name} ({len(images)} images, {n_folds}-fold CV, grid={grid}x{grid})")
        print(f"{'='*60}")

        clear_results("occlusion_block_voting", ds_name)
        all_results = []

        for method_key, method_cfg in METHODS.items():
            method_name = method_cfg["method"]
            method_label = method_cfg["label"]

            for occ in occ_levels:
                occ_label = f"occ_{int(occ*100)}" if occ > 0 else "clean"

                t0 = time.time()
                fold_accs = run_cv(images, labels, n_folds, method_name, grid=grid, occ_ratio=occ)
                elapsed = time.time() - t0

                mean_acc = np.mean(fold_accs)
                std_acc = np.std(fold_accs, ddof=1) if len(fold_accs) > 1 else 0
                t_val = stats.t.ppf(0.975, df=len(fold_accs)-1) if len(fold_accs) > 1 else 1.96
                ci95 = t_val * std_acc / np.sqrt(len(fold_accs)) if len(fold_accs) > 1 else 0

                print(f"  {method_label:30s} {occ_label:8s} acc={mean_acc*100:.2f}% +/- {ci95*100:.2f}%  ({elapsed:.1f}s)")

                row = {
                    "method": method_key,
                    "method_label": method_label,
                    "occlusion": occ,
                    "occlusion_label": occ_label,
                    "test_acc": mean_acc,
                    "ci95": ci95,
                    "fold_accs": [float(a) for a in fold_accs],
                    "elapsed": elapsed,
                }
                all_results.append(row)

                for fold_i, acc in enumerate(fold_accs):
                    save_result({
                        "feature": method_key,
                        "feature_params": json.dumps({"grid": grid, "occ": occ}),
                        "preprocess": "none",
                        "classifier": "knn_k3_cosine",
                        "classifier_params": json.dumps({"type": "knn", "n_neighbors": 3, "metric": "cosine"}),
                        "augmentation": occ_label,
                        "fold": fold_i,
                        "test_acc": acc,
                        "train_acc": 0,
                        "macro_f1": 0,
                        "fps": 0,
                        "train_time": 0,
                        "feature_dim": 0,
                    }, "occlusion_block_voting", ds_name)

        # Compute delta
        for method_key in METHODS:
            method_rows = [r for r in all_results if r["method"] == method_key]
            clean_acc = next((r["test_acc"] for r in method_rows if r["occlusion"] == 0), 0)
            for r in method_rows:
                r["delta"] = r["test_acc"] - clean_acc

        save_summary({
            "results": all_results,
            "grid": grid,
            "occlusion_levels": [float(o) for o in occ_levels],
            "methods": list(METHODS.keys()),
        }, "occlusion_block_voting", ds_name)

        print(f"\n  Summary for {ds_name}:")
        print(f"  {'Method':<30s} ", end="")
        for occ in occ_levels:
            print(f"{'Occ'+str(int(occ*100))+'%':>8s}", end="")
        print(f" {'AvgDrop':>8s}")
        print(f"  {'-'*60}")
        for method_key, method_cfg in METHODS.items():
            method_rows = [r for r in all_results if r["method"] == method_key]
            clean = next((r["test_acc"] for r in method_rows if r["occlusion"] == 0), 0)
            vals = []
            for occ in occ_levels:
                r = next((r for r in method_rows if r["occlusion"] == occ), None)
                vals.append(r["test_acc"] if r else 0)
            drops = [clean - v for v in vals[1:] if v > 0]
            avg_drop = np.mean(drops) if drops else 0
            print(f"  {method_cfg['label']:<30s}", end="")
            for v in vals:
                print(f" {v*100:>7.1f}%", end="")
            print(f" {-avg_drop*100:>+7.1f}%")


import json


def main():
    parser = argparse.ArgumentParser(description="Occlusion Block Voting")
    parser.add_argument("--dataset", "-d", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Loading datasets")
    print("=" * 60)
    datasets = load_all_datasets(args.dataset)
    if not datasets:
        print("ERROR: No datasets found!")
        sys.exit(1)

    run_experiment(datasets, quick=args.quick)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
