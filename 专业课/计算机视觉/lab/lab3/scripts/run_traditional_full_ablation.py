"""
Traditional Face Recognition Full Ablation Suite
=================================================
Main entry point for all 8 traditional method experiments.

Usage:
    uv run python scripts/run_traditional_full_ablation.py --experiment feature
    uv run python scripts/run_traditional_full_ablation.py --experiment preprocess
    uv run python scripts/run_traditional_full_ablation.py --experiment classifier
    uv run python scripts/run_traditional_full_ablation.py --experiment params
    uv run python scripts/run_traditional_full_ablation.py --experiment augment
    uv run python scripts/run_traditional_full_ablation.py --experiment robustness
    uv run python scripts/run_traditional_full_ablation.py --experiment fusion
    uv run python scripts/run_traditional_full_ablation.py --experiment stats
    uv run python scripts/run_traditional_full_ablation.py --experiment all
    uv run python scripts/run_traditional_full_ablation.py --experiment feature --dataset FERET --quick
"""

import argparse
import os
import sys
import time
import warnings
from itertools import product

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition_system import (
    EigenfacesExtractor,
    LBPExtractor,
    GaborExtractor,
    HOGExtractor,
    CombinedExtractor,
    augment_dataset,
    add_gaussian_noise,
    add_salt_pepper_noise,
    simulate_occlusion,
    simulate_illumination_change,
    load_feret_dataset,
    preprocess_histogram_eq,
    preprocess_clahe,
    preprocess_gaussian_filter,
    preprocess_median_filter,
    preprocess_bilateral_filter,
    preprocess_nlm_denoising,
    preprocess_gradient_magnitude,
    preprocess_illuminant_normalized,
)

PREPROCESS_FUNCS = {
    "none": None,
    "hist_eq": preprocess_histogram_eq,
    "clahe": preprocess_clahe,
    "gaussian": preprocess_gaussian_filter,
    "median": preprocess_median_filter,
    "bilateral": preprocess_bilateral_filter,
    "nlm": preprocess_nlm_denoising,
    "gradient": preprocess_gradient_magnitude,
    "illuminant_norm": preprocess_illuminant_normalized,
}
from scripts.traditional_configs import (
    DATASET_CONFIGS,
    FEATURE_CONFIGS,
    FEATURE_CONFIGS_QUICK,
    PREPROCESS_CONFIGS,
    PREPROCESS_CONFIGS_QUICK,
    PREPROCESS_FEATURE_GRID,
    CLASSIFIER_CONFIGS,
    CLASSIFIER_CONFIGS_QUICK,
    PARAM_CONFIGS,
    AUGMENT_CONFIGS,
    AUGMENT_CONFIGS_QUICK,
    ROBUSTNESS_CONFIGS,
    ROBUSTNESS_CONFIGS_QUICK,
    ROBUSTNESS_METHODS,
    FUSION_CONFIGS,
    FUSION_ENSEMBLE_CONFIGS,
    DEFAULT_CLASSIFIER,
    RANDOM_SEED,
)
from scripts.results_io import save_result, save_results, save_summary, load_results, clear_results, clear_overwrite_session

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
FERET_DIR = os.path.join(DATA_DIR, "数据库-feret_k175_s7_w80_h80", "feret_k175_s7_w80_h80")
IMG_SIZE = (80, 80)


# ─── Dataset loading ─────────────────────────────────────────────────────────

def load_all_datasets(datasets=None):
    """Load specified datasets (or all). Returns dict of name -> (images, labels)."""
    all_ds = {}
    names = datasets or ["FERET", "Yale", "Olivetti"]

    if "FERET" in names and os.path.exists(FERET_DIR):
        print("  Loading FERET...")
        imgs, lbls = load_feret_dataset(FERET_DIR, img_size=IMG_SIZE)
        all_ds["FERET"] = (imgs, lbls)
        print(f"    {len(imgs)} images, {len(np.unique(lbls))} people, {imgs[0].shape}")

    if "Yale" in names:
        yp = os.path.join(DATA_DIR, "yale_images.npy")
        lp = os.path.join(DATA_DIR, "yale_labels.npy")
        if os.path.exists(yp) and os.path.exists(lp):
            print("  Loading Yale...")
            imgs, lbls = np.load(yp), np.load(lp)
            if imgs.shape[1:] != IMG_SIZE:
                imgs = np.array([cv2.resize(im, IMG_SIZE) for im in imgs])
            all_ds["Yale"] = (imgs, lbls)
            print(f"    {len(imgs)} images, {len(np.unique(lbls))} people, {imgs[0].shape}")

    if "Olivetti" in names:
        op = os.path.join(DATA_DIR, "olivetti_images.npy")
        lp = os.path.join(DATA_DIR, "olivetti_labels.npy")
        if os.path.exists(op) and os.path.exists(lp):
            print("  Loading Olivetti...")
            imgs, lbls = np.load(op), np.load(lp)
            if imgs.shape[1:] != IMG_SIZE:
                imgs = np.array([cv2.resize(im, IMG_SIZE) for im in imgs])
            all_ds["Olivetti"] = (imgs, lbls)
            print(f"    {len(imgs)} images, {len(np.unique(lbls))} people, {imgs[0].shape}")

    return all_ds


# ─── Factories ───────────────────────────────────────────────────────────────

def create_extractor(name, params=None):
    """Create a feature extractor from config name and params dict."""
    p = params or {}
    if name == "eigenfaces":
        return EigenfacesExtractor(n_components=p.get("n_components", 50))
    elif name == "lbp":
        return LBPExtractor(grid_x=p.get("grid_x", 3), grid_y=p.get("grid_y", 3))
    elif name == "gabor":
        return GaborExtractor(
            frequencies=p.get("frequencies", (0.1, 0.2, 0.3)),
            orientations=p.get("orientations", (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)),
            grid_x=p.get("grid_x", 2),
            grid_y=p.get("grid_y", 2),
        )
    elif name == "hog":
        cs = p.get("cell_size", (8, 8))
        bs = p.get("block_size", (cs[0] * 2, cs[1] * 2))
        return HOGExtractor(
            win_size=p.get("win_size", IMG_SIZE),
            cell_size=cs,
            block_size=bs,
            block_stride=(cs[0], cs[1]),
            nbins=p.get("nbins", 9),
        )
    elif name == "combined":
        return CombinedExtractor(
            eigen_components=p.get("eigen_components", 50),
            lbp_components=p.get("lbp_components", 100),
            lbp_grid=p.get("lbp_grid", 3),
        )
    else:
        raise ValueError(f"Unknown extractor: {name}")


def create_classifier(cfg):
    """Create a classifier from a config dict."""
    t = cfg["type"]
    if t == "svm_linear":
        return SVC(kernel="linear", C=cfg.get("C", 1), class_weight="balanced",
                   probability=True, random_state=RANDOM_SEED)
    elif t == "svm_rbf":
        return SVC(kernel="rbf", C=cfg.get("C", 1), gamma=cfg.get("gamma", "scale"),
                   class_weight="balanced", probability=True, random_state=RANDOM_SEED)
    elif t == "knn":
        return KNeighborsClassifier(
            n_neighbors=cfg.get("n_neighbors", 3), weights="distance",
            metric=cfg.get("metric", "cosine"),
        )
    elif t == "logistic":
        return LogisticRegression(C=cfg.get("C", 1), max_iter=2000, solver="saga", random_state=RANDOM_SEED)
    elif t == "rf":
        return RandomForestClassifier(n_estimators=cfg.get("n_estimators", 100), random_state=RANDOM_SEED, n_jobs=-1)
    elif t == "gbdt":
        return GradientBoostingClassifier(n_estimators=cfg.get("n_estimators", 100), random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown classifier type: {t}")


def apply_degradation(images, deg_cfg, seed=RANDOM_SEED):
    """Apply a degradation to test images."""
    t = deg_cfg["type"]
    if t == "gaussian_noise":
        return add_gaussian_noise(images, sigma=deg_cfg.get("sigma", 25), seed=seed)
    elif t == "salt_pepper":
        return add_salt_pepper_noise(images, amount=deg_cfg.get("amount", 0.05), seed=seed)
    elif t == "blur":
        k = deg_cfg.get("kernel_size", 5)
        return np.array([cv2.GaussianBlur(im, (k, k), 0) for im in images])
    elif t == "occlusion":
        return simulate_occlusion(images, occlusion_ratio=deg_cfg.get("ratio", 0.2), seed=seed)
    elif t == "gamma":
        return simulate_illumination_change(images, gamma=deg_cfg.get("gamma", 0.5))
    elif t == "low_res":
        scale = deg_cfg.get("scale", 0.5)
        degraded = []
        for im in images:
            h, w = im.shape[:2]
            small = cv2.resize(im, (int(w * scale), int(h * scale)))
            restored = cv2.resize(small, (w, h))
            degraded.append(restored)
        return np.array(degraded)
    elif t == "rotation":
        angle = deg_cfg.get("angle", 5)
        from PIL import Image as PILImage
        rotated = []
        for im in images:
            pil = PILImage.fromarray(im)
            rot = pil.rotate(angle, fillcolor=0)
            rotated.append(np.array(rot))
        return np.array(rotated)
    else:
        raise ValueError(f"Unknown degradation: {t}")


# ─── Core evaluation ─────────────────────────────────────────────────────────

def evaluate_fold(X_tr_img, y_tr, X_te_img, y_te, extractor_name, extractor_params,
                  classifier_cfg, augment_cfg=None, preprocess_cfg=None):
    """Train on one fold, return metrics dict."""
    # Apply preprocessing
    if preprocess_cfg and preprocess_cfg != "none":
        pp_func = PREPROCESS_FUNCS.get(preprocess_cfg)
        if pp_func:
            X_tr_img = pp_func(X_tr_img)
            X_te_img = pp_func(X_te_img)

    # Augment training data
    if augment_cfg:
        angles = augment_cfg.get("angles")
        flip = augment_cfg.get("flip", False)
        if angles or flip:
            X_tr_img, y_tr = augment_dataset(
                X_tr_img, y_tr,
                flips=flip,
                rotations=angles is not None,
                angles=angles or (-10, -5, 5, 10),
            )

    # Extract features
    ext = create_extractor(extractor_name, extractor_params)
    t0 = time.perf_counter()
    X_tr_feat = ext.fit_transform(X_tr_img)
    X_te_feat = ext.transform(X_te_img)
    feat_time = time.perf_counter() - t0

    # Scale
    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    # Train classifier
    clf = create_classifier(classifier_cfg)
    t0 = time.perf_counter()
    clf.fit(X_tr_feat, y_tr)
    train_time = time.perf_counter() - t0

    # Predict + FPS
    t0 = time.perf_counter()
    y_pred = clf.predict(X_te_feat)
    pred_time = time.perf_counter() - t0
    fps = len(X_te_feat) / pred_time if pred_time > 0 else 0

    n_classes = len(np.unique(y_tr))
    train_acc = clf.score(X_tr_feat, y_tr)
    test_acc = accuracy_score(y_te, y_pred)

    rank5_acc = None
    if hasattr(clf, "predict_proba") and n_classes >= 5:
        proba = clf.predict_proba(X_te_feat)
        rank5_acc = top_k_accuracy_score(y_te, proba, k=5, labels=clf.classes_)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "rank5_acc": rank5_acc,
        "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "fps": fps,
        "train_time": train_time,
        "feat_time": feat_time,
        "feature_dim": X_tr_feat.shape[1],
    }


def cross_validate(images, labels, n_folds, extractor_name, extractor_params,
                   classifier_cfg, augment_cfg=None, seed=RANDOM_SEED, desc="",
                   preprocess_cfg=None):
    """Stratified k-fold CV with per-fold augmentation."""
    unique, counts = np.unique(labels, return_counts=True)
    actual_folds = min(n_folds, min(counts))
    if actual_folds < 2:
        actual_folds = 2
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    fold_results = []
    for i, (tr_idx, te_idx) in enumerate(cv.split(images, labels)):
        if desc:
            pct = (i + 1) / actual_folds
            bar = "#" * int(40 * pct) + "-" * (40 - int(40 * pct))
            print(f"\r  {desc} [{bar}] {i+1}/{actual_folds} ({pct:.0%})", end="", flush=True)
        metrics = evaluate_fold(
            images[tr_idx], labels[tr_idx],
            images[te_idx], labels[te_idx],
            extractor_name, extractor_params,
            classifier_cfg, augment_cfg, preprocess_cfg,
        )
        fold_results.append(metrics)
    if desc:
        print()
    return fold_results


def summarize_folds(fold_results):
    """Aggregate fold results: mean, std, ci95."""
    n = len(fold_results)
    t_val = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96

    def _agg(key):
        vals = [r[key] for r in fold_results if r[key] is not None]
        if not vals:
            return None, None, None
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = float(t_val * s / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return m, s, ci

    result = {}
    for key in ["train_acc", "test_acc", "rank5_acc", "macro_f1", "precision", "recall", "fps", "train_time", "feat_time"]:
        m, s, ci = _agg(key)
        result[f"{key}_mean"] = m
        result[f"{key}_std"] = s
        result[f"{key}_ci95"] = ci
    result["gap"] = (result.get("train_acc_mean") or 0) - (result.get("test_acc_mean") or 0)
    result["fold_test_accs"] = [r["test_acc"] for r in fold_results]
    result["feature_dim"] = fold_results[0].get("feature_dim")
    return result


# ─── Experiment 1: Feature Comparison ────────────────────────────────────────

def run_exp_feature(datasets, quick=False):
    print_header("Exp 1: Feature Extraction Comparison")
    configs = FEATURE_CONFIGS_QUICK if quick else FEATURE_CONFIGS
    clf_cfg = DEFAULT_CLASSIFIER

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ({len(images)} images, {n_folds}-fold CV) ---")
        results = []
        for cfg_name, cfg in configs.items():
            ext_type = cfg["type"]
            ext_params = {k: v for k, v in cfg.items() if k != "type"}
            desc = f"{ds_name}|{cfg_name}"
            folds = cross_validate(images, labels, n_folds, ext_type, ext_params, clf_cfg, desc=desc)
            summary = summarize_folds(folds)
            row = {
                "feature": cfg_name,
                "feature_params": ext_params,
                "preprocess": "none",
                "classifier": "svm_rbf",
                "classifier_params": clf_cfg,
                "augmentation": "none",
                "test_acc": summary["test_acc_mean"],
                "rank5_acc": summary.get("rank5_acc_mean"),
                "macro_f1": summary["macro_f1_mean"],
                "fps": summary["fps_mean"],
                "train_time": summary["train_time_mean"],
                "feature_dim": summary["feature_dim"],
                "ci95": summary["test_acc_ci95"],
                "fold_test_accs": summary["fold_test_accs"],
            }
            results.append(row)
            print(f"  {cfg_name:25s} acc={summary['test_acc_mean']*100:.2f}% +/- {summary['test_acc_ci95']*100:.2f}  dim={summary['feature_dim']}")

        # Save per-fold results
        for row in results:
            for fold_i, acc in enumerate(row.get("fold_test_accs", [])):
                save_result({
                    "feature": row["feature"],
                    "feature_params": row["feature_params"],
                    "preprocess": "none",
                    "classifier": "svm_rbf",
                    "classifier_params": clf_cfg,
                    "augmentation": "none",
                    "fold": fold_i,
                    "train_acc": None,
                    "test_acc": acc,
                    "rank5_acc": None,
                    "macro_f1": None,
                    "precision": None,
                    "recall": None,
                    "fps": row["fps"],
                    "train_time": row["train_time"],
                    "feature_dim": row["feature_dim"],
                }, "feature", ds_name)

        # Sort by accuracy
        results.sort(key=lambda r: r["test_acc"] or 0, reverse=True)
        save_summary({"results": results, "best": results[0]["feature"] if results else None}, "feature", ds_name)
        print(f"\n  Best on {ds_name}: {results[0]['feature']} ({results[0]['test_acc']*100:.2f}%)")


# ─── Experiment 2: Preprocessing Ablation ────────────────────────────────────

def run_exp_preprocess(datasets, quick=False):
    print_header("Exp 2: Preprocessing Ablation")
    preprocess_list = PREPROCESS_CONFIGS_QUICK if quick else PREPROCESS_CONFIGS
    clf_cfg = DEFAULT_CLASSIFIER

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ---")
        results = []

        for feat_name in PREPROCESS_FEATURE_GRID:
            for pp in preprocess_list:
                # Apply preprocessing to all images (array-based functions)
                pp_func = PREPROCESS_FUNCS.get(pp)
                pp_images = pp_func(images) if pp_func else images

                ext_params = {}
                if feat_name == "eigenfaces":
                    ext_params = {"n_components": 50}
                elif feat_name == "lbp":
                    ext_params = {"grid_x": 3, "grid_y": 3}
                elif feat_name == "combined":
                    ext_params = {"eigen_components": 50, "lbp_components": 100}

                desc = f"{ds_name}|{feat_name}|{pp}"
                folds = cross_validate(pp_images, labels, n_folds, feat_name, ext_params, clf_cfg, desc=desc)
                summary = summarize_folds(folds)

                row = {
                    "feature": feat_name,
                    "preprocess": pp,
                    "test_acc": summary["test_acc_mean"],
                    "ci95": summary["test_acc_ci95"],
                    "macro_f1": summary["macro_f1_mean"],
                    "fold_test_accs": summary["fold_test_accs"],
                }
                results.append(row)
                print(f"  {feat_name:12s} + {pp:18s} acc={summary['test_acc_mean']*100:.2f}%")

                for fold_i, acc in enumerate(summary["fold_test_accs"]):
                    save_result({
                        "feature": feat_name,
                        "feature_params": ext_params,
                        "preprocess": pp,
                        "classifier": "svm_rbf",
                        "classifier_params": clf_cfg,
                        "augmentation": "none",
                        "fold": fold_i,
                        "test_acc": acc,
                        "train_acc": summary["train_acc_mean"],
                        "rank5_acc": summary.get("rank5_acc_mean"),
                        "macro_f1": summary["macro_f1_mean"],
                        "precision": summary["precision_mean"],
                        "recall": summary["recall_mean"],
                        "fps": summary["fps_mean"],
                        "train_time": summary["train_time_mean"],
                        "feature_dim": summary["feature_dim"],
                    }, "preprocess", ds_name)

        save_summary({"results": results}, "preprocess", ds_name)


# ─── Experiment 3: Classifier Ablation ───────────────────────────────────────

def run_exp_classifier(datasets, quick=False):
    print_header("Exp 3: Classifier & Hyperparameter Ablation")
    clf_configs = CLASSIFIER_CONFIGS_QUICK if quick else CLASSIFIER_CONFIGS

    # Use best features per dataset (default: eigenfaces_50 + lbp_3x3 + combined)
    test_features = [
        ("eigenfaces", {"n_components": 50}),
        ("lbp", {"grid_x": 3, "grid_y": 3}),
        ("combined", {"eigen_components": 50, "lbp_components": 100}),
    ]

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ---")
        results = []

        for feat_name, feat_params in test_features:
            for clf_name, clf_cfg in clf_configs.items():
                desc = f"{ds_name}|{feat_name}|{clf_name}"
                folds = cross_validate(images, labels, n_folds, feat_name, feat_params, clf_cfg, desc=desc)
                summary = summarize_folds(folds)

                row = {
                    "feature": feat_name,
                    "feature_params": feat_params,
                    "classifier": clf_name,
                    "classifier_params": clf_cfg,
                    "test_acc": summary["test_acc_mean"],
                    "ci95": summary["test_acc_ci95"],
                    "macro_f1": summary["macro_f1_mean"],
                    "fps": summary["fps_mean"],
                    "gap": summary["gap"],
                    "fold_test_accs": summary["fold_test_accs"],
                }
                results.append(row)
                print(f"  {feat_name:12s} + {clf_name:25s} acc={summary['test_acc_mean']*100:.2f}% gap={summary['gap']*100:.2f}%")

                for fold_i, acc in enumerate(summary["fold_test_accs"]):
                    save_result({
                        "feature": feat_name,
                        "feature_params": feat_params,
                        "preprocess": "none",
                        "classifier": clf_name,
                        "classifier_params": clf_cfg,
                        "augmentation": "none",
                        "fold": fold_i,
                        "test_acc": acc,
                        "train_acc": summary["train_acc_mean"],
                        "rank5_acc": summary.get("rank5_acc_mean"),
                        "macro_f1": summary["macro_f1_mean"],
                        "precision": summary["precision_mean"],
                        "recall": summary["recall_mean"],
                        "fps": summary["fps_mean"],
                        "train_time": summary["train_time_mean"],
                        "feature_dim": summary["feature_dim"],
                    }, "classifier", ds_name)

        save_summary({"results": results}, "classifier", ds_name)


# ─── Experiment 4: Feature Parameter Ablation ────────────────────────────────

def run_exp_params(datasets, quick=False):
    print_header("Exp 4: Feature Parameter Ablation")
    clf_cfg = DEFAULT_CLASSIFIER

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ---")
        results = []

        # Eigenfaces: n_components sweep
        for nc in ([50, 100] if quick else PARAM_CONFIGS["eigenfaces"]["n_components"]):
            for whiten in ([True] if quick else PARAM_CONFIGS["eigenfaces"]["whiten"]):
                ext_params = {"n_components": nc, "whiten": whiten}
                desc = f"{ds_name}|eigenfaces_nc{nc}_w{whiten}"
                folds = cross_validate(images, labels, n_folds, "eigenfaces", ext_params, clf_cfg, desc=desc)
                summary = summarize_folds(folds)
                row = {"feature": "eigenfaces", "params": f"nc={nc},whiten={whiten}",
                       "test_acc": summary["test_acc_mean"], "ci95": summary["test_acc_ci95"],
                       "feature_dim": summary["feature_dim"], "fps": summary["fps_mean"],
                       "fold_test_accs": summary["fold_test_accs"]}
                results.append(row)
                print(f"  eigenfaces nc={nc} whiten={whiten} acc={summary['test_acc_mean']*100:.2f}% dim={summary['feature_dim']}")

                for fold_i, acc in enumerate(summary["fold_test_accs"]):
                    save_result({
                        "feature": "eigenfaces", "feature_params": ext_params,
                        "preprocess": "none", "classifier": "svm_rbf", "classifier_params": clf_cfg,
                        "augmentation": "none", "fold": fold_i, "test_acc": acc,
                        "train_acc": summary["train_acc_mean"], "rank5_acc": summary.get("rank5_acc_mean"),
                        "macro_f1": summary["macro_f1_mean"], "precision": summary["precision_mean"],
                        "recall": summary["recall_mean"], "fps": summary["fps_mean"],
                        "train_time": summary["train_time_mean"], "feature_dim": summary["feature_dim"],
                    }, "params", ds_name)

        # LBP: grid sweep
        for grid in ([(3, 3)] if quick else PARAM_CONFIGS["lbp"]["grid"]):
            ext_params = {"grid_x": grid[0], "grid_y": grid[1]}
            desc = f"{ds_name}|lbp_{grid[0]}x{grid[1]}"
            folds = cross_validate(images, labels, n_folds, "lbp", ext_params, clf_cfg, desc=desc)
            summary = summarize_folds(folds)
            row = {"feature": "lbp", "params": f"grid={grid}",
                   "test_acc": summary["test_acc_mean"], "ci95": summary["test_acc_ci95"],
                   "feature_dim": summary["feature_dim"], "fps": summary["fps_mean"],
                   "fold_test_accs": summary["fold_test_accs"]}
            results.append(row)
            print(f"  lbp grid={grid} acc={summary['test_acc_mean']*100:.2f}% dim={summary['feature_dim']}")

            for fold_i, acc in enumerate(summary["fold_test_accs"]):
                save_result({
                    "feature": "lbp", "feature_params": ext_params,
                    "preprocess": "none", "classifier": "svm_rbf", "classifier_params": clf_cfg,
                    "augmentation": "none", "fold": fold_i, "test_acc": acc,
                    "train_acc": summary["train_acc_mean"], "rank5_acc": summary.get("rank5_acc_mean"),
                    "macro_f1": summary["macro_f1_mean"], "precision": summary["precision_mean"],
                    "recall": summary["recall_mean"], "fps": summary["fps_mean"],
                    "train_time": summary["train_time_mean"], "feature_dim": summary["feature_dim"],
                }, "params", ds_name)

        # HOG: cell_size sweep
        for cs in ([(8, 8)] if quick else PARAM_CONFIGS["hog"]["cell_size"]):
            ext_params = {"cell_size": cs, "block_size": (cs[0] * 2, cs[1] * 2), "nbins": 9}
            desc = f"{ds_name}|hog_cell{cs[0]}"
            folds = cross_validate(images, labels, n_folds, "hog", ext_params, clf_cfg, desc=desc)
            summary = summarize_folds(folds)
            row = {"feature": "hog", "params": f"cell={cs}",
                   "test_acc": summary["test_acc_mean"], "ci95": summary["test_acc_ci95"],
                   "feature_dim": summary["feature_dim"], "fps": summary["fps_mean"],
                   "fold_test_accs": summary["fold_test_accs"]}
            results.append(row)
            print(f"  hog cell={cs} acc={summary['test_acc_mean']*100:.2f}% dim={summary['feature_dim']}")

            for fold_i, acc in enumerate(summary["fold_test_accs"]):
                save_result({
                    "feature": "hog", "feature_params": ext_params,
                    "preprocess": "none", "classifier": "svm_rbf", "classifier_params": clf_cfg,
                    "augmentation": "none", "fold": fold_i, "test_acc": acc,
                    "train_acc": summary["train_acc_mean"], "rank5_acc": summary.get("rank5_acc_mean"),
                    "macro_f1": summary["macro_f1_mean"], "precision": summary["precision_mean"],
                    "recall": summary["recall_mean"], "fps": summary["fps_mean"],
                    "train_time": summary["train_time_mean"], "feature_dim": summary["feature_dim"],
                }, "params", ds_name)

        # Combined: eigen_components x lbp_components sweep
        for ec in ([50] if quick else PARAM_CONFIGS["combined"]["eigen_components"]):
            for lc in ([100] if quick else PARAM_CONFIGS["combined"]["lbp_components"]):
                ext_params = {"eigen_components": ec, "lbp_components": lc}
                desc = f"{ds_name}|combined_{ec}+{lc}"
                folds = cross_validate(images, labels, n_folds, "combined", ext_params, clf_cfg, desc=desc)
                summary = summarize_folds(folds)
                row = {"feature": "combined", "params": f"eigen={ec},lbp={lc}",
                       "test_acc": summary["test_acc_mean"], "ci95": summary["test_acc_ci95"],
                       "feature_dim": summary["feature_dim"], "fps": summary["fps_mean"],
                       "fold_test_accs": summary["fold_test_accs"]}
                results.append(row)
                print(f"  combined eigen={ec} lbp={lc} acc={summary['test_acc_mean']*100:.2f}% dim={summary['feature_dim']}")

                for fold_i, acc in enumerate(summary["fold_test_accs"]):
                    save_result({
                        "feature": "combined", "feature_params": ext_params,
                        "preprocess": "none", "classifier": "svm_rbf", "classifier_params": clf_cfg,
                        "augmentation": "none", "fold": fold_i, "test_acc": acc,
                        "train_acc": summary["train_acc_mean"], "rank5_acc": summary.get("rank5_acc_mean"),
                        "macro_f1": summary["macro_f1_mean"], "precision": summary["precision_mean"],
                        "recall": summary["recall_mean"], "fps": summary["fps_mean"],
                        "train_time": summary["train_time_mean"], "feature_dim": summary["feature_dim"],
                    }, "params", ds_name)

        save_summary({"results": results}, "params", ds_name)


# ─── Experiment 5: Augmentation Ablation ─────────────────────────────────────

def run_exp_augment(datasets, quick=False):
    print_header("Exp 5: Data Augmentation Ablation")
    aug_configs = AUGMENT_CONFIGS_QUICK if quick else AUGMENT_CONFIGS
    clf_cfg = DEFAULT_CLASSIFIER

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ---")
        results = []

        # Use best feature for each dataset (default: combined)
        feat_name = "combined"
        feat_params = {"eigen_components": 50, "lbp_components": 100}

        for aug_name, aug_cfg in aug_configs.items():
            desc = f"{ds_name}|aug_{aug_name}"
            folds = cross_validate(images, labels, n_folds, feat_name, feat_params, clf_cfg,
                                   augment_cfg=aug_cfg if aug_name != "none" else None, desc=desc)
            summary = summarize_folds(folds)

            row = {
                "augmentation": aug_name,
                "test_acc": summary["test_acc_mean"],
                "ci95": summary["test_acc_ci95"],
                "macro_f1": summary["macro_f1_mean"],
                "train_time": summary["train_time_mean"],
                "fold_test_accs": summary["fold_test_accs"],
            }
            results.append(row)
            print(f"  {aug_name:20s} acc={summary['test_acc_mean']*100:.2f}%")

            for fold_i, acc in enumerate(summary["fold_test_accs"]):
                save_result({
                    "feature": feat_name, "feature_params": feat_params,
                    "preprocess": "none", "classifier": "svm_rbf", "classifier_params": clf_cfg,
                    "augmentation": aug_name, "fold": fold_i, "test_acc": acc,
                    "train_acc": summary["train_acc_mean"], "rank5_acc": summary.get("rank5_acc_mean"),
                    "macro_f1": summary["macro_f1_mean"], "precision": summary["precision_mean"],
                    "recall": summary["recall_mean"], "fps": summary["fps_mean"],
                    "train_time": summary["train_time_mean"], "feature_dim": summary["feature_dim"],
                }, "augment", ds_name)

        save_summary({"results": results}, "augment", ds_name)


# ─── Experiment 6: Robustness ────────────────────────────────────────────────

def run_exp_robustness(datasets, quick=False):
    print_header("Exp 6: Robustness / Degradation Ablation")
    deg_configs = ROBUSTNESS_CONFIGS_QUICK if quick else ROBUSTNESS_CONFIGS
    clf_cfg = DEFAULT_CLASSIFIER

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ---")
        results = []

        for method_name in ROBUSTNESS_METHODS:
            cfg = FEATURE_CONFIGS.get(method_name, {})
            ext_type = cfg.get("type", "eigenfaces")
            ext_params = {k: v for k, v in cfg.items() if k != "type"}

            # First: clean accuracy
            desc = f"{ds_name}|{method_name}|clean"
            folds = cross_validate(images, labels, n_folds, ext_type, ext_params, clf_cfg, desc=desc)
            clean_summary = summarize_folds(folds)
            clean_acc = clean_summary["test_acc_mean"]
            print(f"  {method_name:25s} clean acc={clean_acc*100:.2f}%")

            # Then: each degradation
            for deg_name, deg_cfg in deg_configs.items():
                # For robustness, we train on clean data and test on degraded
                # We need a custom evaluation that degrades only test fold
                desc2 = f"{ds_name}|{method_name}|{deg_name}"
                unique, counts = np.unique(labels, return_counts=True)
                actual_folds = min(n_folds, min(counts))
                if actual_folds < 2:
                    actual_folds = 2
                cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_SEED)
                deg_fold_results = []

                for i, (tr_idx, te_idx) in enumerate(cv.split(images, labels)):
                    pct = (i + 1) / actual_folds
                    bar = "#" * int(40 * pct) + "-" * (40 - int(40 * pct))
                    print(f"\r  {desc2} [{bar}] {i+1}/{actual_folds} ({pct:.0%})", end="", flush=True)

                    X_tr, y_tr = images[tr_idx], labels[tr_idx]
                    X_te, y_te = images[te_idx], labels[te_idx]

                    # Degrade test images only
                    X_te_deg = apply_degradation(X_te, deg_cfg)

                    ext = create_extractor(ext_type, ext_params)
                    X_tr_feat = ext.fit_transform(X_tr)
                    X_te_feat = ext.transform(X_te_deg)

                    scaler = StandardScaler()
                    X_tr_feat = scaler.fit_transform(X_tr_feat)
                    X_te_feat = scaler.transform(X_te_feat)

                    clf = create_classifier(clf_cfg)
                    clf.fit(X_tr_feat, y_tr)
                    y_pred = clf.predict(X_te_feat)

                    deg_fold_results.append({
                        "train_acc": None,
                        "test_acc": accuracy_score(y_te, y_pred),
                        "rank5_acc": None,
                        "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
                        "precision": None,
                        "recall": None,
                        "fps": None,
                        "train_time": None,
                        "feat_time": None,
                        "feature_dim": None,
                    })
                print()

                deg_summary = summarize_folds(deg_fold_results) if len(deg_fold_results) > 1 else {
                    "test_acc_mean": np.mean([r["test_acc"] for r in deg_fold_results]),
                    "test_acc_ci95": 0,
                    "macro_f1_mean": np.mean([r["macro_f1"] for r in deg_fold_results]),
                }

                deg_acc = deg_summary["test_acc_mean"]
                delta = deg_acc - clean_acc

                row = {
                    "method": method_name,
                    "degradation": deg_name,
                    "clean_acc": clean_acc,
                    "degraded_acc": deg_acc,
                    "delta": delta,
                    "macro_f1": deg_summary.get("macro_f1_mean"),
                }
                results.append(row)
                print(f"    {deg_name:25s} deg_acc={deg_acc*100:.2f}% delta={delta*100:+.2f}%")

                for fold_i, fr in enumerate(deg_fold_results):
                    save_result({
                        "feature": method_name, "feature_params": ext_params,
                        "preprocess": "none", "classifier": "svm_rbf", "classifier_params": clf_cfg,
                        "augmentation": deg_name,  # misuse this field for degradation type
                        "fold": fold_i, "test_acc": fr["test_acc"],
                        "train_acc": None, "rank5_acc": None,
                        "macro_f1": fr["macro_f1"], "precision": None, "recall": None,
                        "fps": None, "train_time": None, "feature_dim": None,
                    }, "robustness", ds_name)

        save_summary({"results": results}, "robustness", ds_name)


# ─── Experiment 7: Fusion & Ensemble ─────────────────────────────────────────

def run_exp_fusion(datasets, quick=False):
    print_header("Exp 7: Feature Fusion & Classifier Ensemble")

    for ds_name, (images, labels) in datasets.items():
        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ---")
        results = []

        fusion_cfgs = {
            k: v for k, v in FUSION_CONFIGS.items()
            if not quick or k in ["eigenfaces_only", "lbp_only", "eigen_lbp"]
        }

        for fus_name, fus_cfg in fusion_cfgs.items():
            strategy = fus_cfg["strategy"]
            feat_names = fus_cfg["features"]

            if strategy == "single":
                cfg = FEATURE_CONFIGS.get(feat_names[0], {})
                ext_type = cfg.get("type", "eigenfaces")
                ext_params = {k: v for k, v in cfg.items() if k != "type"}
                desc = f"{ds_name}|{fus_name}"
                folds = cross_validate(images, labels, n_folds, ext_type, ext_params, DEFAULT_CLASSIFIER, desc=desc)
                summary = summarize_folds(folds)

            elif strategy == "early":
                # Early fusion: concatenate features
                desc = f"{ds_name}|{fus_name}"
                unique, counts = np.unique(labels, return_counts=True)
                actual_folds = min(n_folds, min(counts))
                if actual_folds < 2:
                    actual_folds = 2
                cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_SEED)
                fold_results = []

                for i, (tr_idx, te_idx) in enumerate(cv.split(images, labels)):
                    pct = (i + 1) / actual_folds
                    bar = "#" * int(40 * pct) + "-" * (40 - int(40 * pct))
                    print(f"\r  {desc} [{bar}] {i+1}/{actual_folds} ({pct:.0%})", end="", flush=True)

                    X_tr, y_tr = images[tr_idx], labels[tr_idx]
                    X_te, y_te = images[te_idx], labels[te_idx]

                    # Extract and concatenate features
                    tr_feats, te_feats = [], []
                    for fn in feat_names:
                        cfg = FEATURE_CONFIGS.get(fn, {})
                        ext_type = cfg.get("type", "eigenfaces")
                        ext_params = {k: v for k, v in cfg.items() if k != "type"}
                        ext = create_extractor(ext_type, ext_params)
                        tr_feats.append(ext.fit_transform(X_tr))
                        te_feats.append(ext.transform(X_te))

                    X_tr_feat = np.hstack(tr_feats)
                    X_te_feat = np.hstack(te_feats)

                    scaler = StandardScaler()
                    X_tr_feat = scaler.fit_transform(X_tr_feat)
                    X_te_feat = scaler.transform(X_te_feat)

                    clf = create_classifier(DEFAULT_CLASSIFIER)
                    clf.fit(X_tr_feat, y_tr)
                    y_pred = clf.predict(X_te_feat)
                    n_classes = len(np.unique(y_tr))

                    rank5_acc = None
                    if hasattr(clf, "predict_proba") and n_classes >= 5:
                        proba = clf.predict_proba(X_te_feat)
                        rank5_acc = top_k_accuracy_score(y_te, proba, k=5, labels=clf.classes_)

                    fold_results.append({
                        "train_acc": clf.score(X_tr_feat, y_tr),
                        "test_acc": accuracy_score(y_te, y_pred),
                        "rank5_acc": rank5_acc,
                        "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
                        "precision": precision_score(y_te, y_pred, average="macro", zero_division=0),
                        "recall": recall_score(y_te, y_pred, average="macro", zero_division=0),
                        "fps": 0,
                        "train_time": 0,
                        "feat_time": 0,
                        "feature_dim": X_tr_feat.shape[1],
                    })
                print()
                summary = summarize_folds(fold_results)

            row = {
                "fusion": fus_name,
                "strategy": strategy,
                "features": feat_names,
                "test_acc": summary["test_acc_mean"],
                "ci95": summary["test_acc_ci95"],
                "macro_f1": summary["macro_f1_mean"],
                "feature_dim": summary.get("feature_dim"),
                "fold_test_accs": summary.get("fold_test_accs", []),
            }
            results.append(row)
            print(f"  {fus_name:25s} ({strategy:6s}) acc={summary['test_acc_mean']*100:.2f}%")

            for fold_i, acc in enumerate(summary.get("fold_test_accs", [])):
                save_result({
                    "feature": fus_name, "feature_params": {"strategy": strategy, "features": feat_names},
                    "preprocess": "none", "classifier": "svm_rbf", "classifier_params": DEFAULT_CLASSIFIER,
                    "augmentation": "none", "fold": fold_i, "test_acc": acc,
                    "train_acc": summary["train_acc_mean"], "rank5_acc": summary.get("rank5_acc_mean"),
                    "macro_f1": summary["macro_f1_mean"], "precision": summary["precision_mean"],
                    "recall": summary["recall_mean"], "fps": summary["fps_mean"],
                    "train_time": summary["train_time_mean"], "feature_dim": summary.get("feature_dim"),
                }, "fusion", ds_name)

        save_summary({"results": results}, "fusion", ds_name)


# ─── Experiment 8: Statistical Significance & Final Selection ────────────────

def run_exp_stats(datasets, quick=False):
    print_header("Exp 8: Statistical Significance & Final Model Selection")

    for ds_name, (images, labels) in datasets.items():
        print(f"\n--- {ds_name} ---")

        # Collect fold-level results for all major configurations
        candidate_configs = [
            ("eigenfaces_50", "eigenfaces", {"n_components": 50}),
            ("lbp_3x3", "lbp", {"grid_x": 3, "grid_y": 3}),
            ("hog_default", "hog", {}),
            ("combined_50_100", "combined", {"eigen_components": 50, "lbp_components": 100}),
        ]
        if not quick:
            candidate_configs.extend([
                ("eigenfaces_100", "eigenfaces", {"n_components": 100}),
                ("lbp_4x4", "lbp", {"grid_x": 4, "grid_y": 4}),
                ("gabor_basic", "gabor", {}),
            ])

        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        all_fold_accs = {}

        for cfg_name, ext_type, ext_params in candidate_configs:
            desc = f"{ds_name}|stats|{cfg_name}"
            folds = cross_validate(images, labels, n_folds, ext_type, ext_params, DEFAULT_CLASSIFIER, desc=desc)
            summary = summarize_folds(folds)
            all_fold_accs[cfg_name] = summary["fold_test_accs"]
            print(f"  {cfg_name:25s} {summary['test_acc_mean']*100:.2f}% +/- {summary['test_acc_ci95']*100:.2f}")

        # Pairwise t-tests
        print(f"\n  Pairwise t-tests (alpha=0.05):")
        cfg_names = list(all_fold_accs.keys())
        sig_results = []
        for i in range(len(cfg_names)):
            for j in range(i + 1, len(cfg_names)):
                a = np.array(all_fold_accs[cfg_names[i]])
                b = np.array(all_fold_accs[cfg_names[j]])
                if len(a) == len(b) and len(a) > 1:
                    t_stat, p_val = stats.ttest_rel(a, b)
                    sig = "*" if p_val < 0.05 else ""
                    diff = np.mean(a) - np.mean(b)
                    sig_results.append({
                        "method_a": cfg_names[i], "method_b": cfg_names[j],
                        "mean_diff": float(diff), "t_stat": float(t_stat), "p_value": float(p_val),
                        "significant": p_val < 0.05,
                    })
                    print(f"    {cfg_names[i]:20s} vs {cfg_names[j]:20s} diff={diff*100:+.2f}% p={p_val:.4f} {sig}")

        # Top-5 ranking
        ranked = sorted(
            [(name, np.mean(accs)) for name, accs in all_fold_accs.items()],
            key=lambda x: x[1], reverse=True,
        )
        print(f"\n  Top-5 on {ds_name}:")
        for rank, (name, acc) in enumerate(ranked[:5], 1):
            print(f"    #{rank}: {name} ({acc*100:.2f}%)")

        save_summary({
            "ranking": [{"rank": i + 1, "method": n, "accuracy": float(a)} for i, (n, a) in enumerate(ranked[:5])],
            "pairwise_tests": sig_results,
            "fold_accuracies": {k: [float(v) for v in vals] for k, vals in all_fold_accs.items()},
        }, "stats", ds_name)


# ─── Experiment: Final Candidate Verification ────────────────────────────────

FINAL_CANDIDATES = {
    "FERET": [
        {"name": "combined_rbf_c1", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": None},
        {"name": "combined_rbf_c10", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 10, "gamma": "scale"}, "augment": None},
        {"name": "combined_linear_c1", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_linear", "C": 1}, "augment": None},
        {"name": "combined_linear_c10", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_linear", "C": 10}, "augment": None},
        {"name": "combined_rbf_c10_aug", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 10, "gamma": "scale"}, "augment": {"angles": (-10, -5, 5, 10), "flip": False}},
        {"name": "combined_rbf_c1_flip_aug", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": {"angles": (-10, -5, 5, 10), "flip": True}},
        {"name": "combined_logistic_c0.1", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "logistic", "C": 0.1}, "augment": None},
        {"name": "combined_logistic_c1", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "logistic", "C": 1}, "augment": None},
        {"name": "combined_logistic_c10", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "logistic", "C": 10}, "augment": None},
        {"name": "combined_logistic_c1_flip_aug", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "logistic", "C": 1}, "augment": {"angles": (-10, -5, 5, 10), "flip": True}},
        {"name": "combined_hist_eq_rbf_flip_aug", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": {"angles": (-10, -5, 5, 10), "flip": True},
         "preprocess": "hist_eq"},
        {"name": "eigenfaces100_linear", "feature": "eigenfaces", "feature_params": {"n_components": 100},
         "classifier": {"type": "svm_linear", "C": 1}, "augment": None},
        {"name": "lbp3x3_linear", "feature": "lbp", "feature_params": {"grid_x": 3, "grid_y": 3},
         "classifier": {"type": "svm_linear", "C": 1}, "augment": None},
    ],
    "Yale": [
        {"name": "hog_rbf_c1", "feature": "hog", "feature_params": {},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": None},
        {"name": "hog_linear_c1", "feature": "hog", "feature_params": {},
         "classifier": {"type": "svm_linear", "C": 1}, "augment": None},
        {"name": "hog_logistic", "feature": "hog", "feature_params": {},
         "classifier": {"type": "logistic", "C": 1}, "augment": None},
        {"name": "eigenfaces150_linear", "feature": "eigenfaces", "feature_params": {"n_components": 150},
         "classifier": {"type": "svm_linear", "C": 1}, "augment": None},
        {"name": "combined_rbf_c1", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": None},
    ],
    "Olivetti": [
        {"name": "hog_rbf_c1", "feature": "hog", "feature_params": {},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": None},
        {"name": "hog_linear_c1", "feature": "hog", "feature_params": {},
         "classifier": {"type": "svm_linear", "C": 1}, "augment": None},
        {"name": "hog_rbf_aug", "feature": "hog", "feature_params": {},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": {"angles": (-10, -5, 5, 10), "flip": False}},
        {"name": "hog_rbf_flip_aug", "feature": "hog", "feature_params": {},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": {"angles": (-10, -5, 5, 10), "flip": True}},
        {"name": "eigenfaces100_rbf", "feature": "eigenfaces", "feature_params": {"n_components": 100},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": None},
        {"name": "combined_rbf_c1", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": None},
        {"name": "combined_rbf_flip_aug", "feature": "combined", "feature_params": {"eigen_components": 50, "lbp_components": 100},
         "classifier": {"type": "svm_rbf", "C": 1, "gamma": "scale"}, "augment": {"angles": (-10, -5, 5, 10), "flip": True}},
    ],
}


def run_exp_final_candidates(datasets, quick=False):
    print_header("Exp Final: Final Candidate Configuration Verification")

    for ds_name, (images, labels) in datasets.items():
        candidates = FINAL_CANDIDATES.get(ds_name, [])
        if not candidates:
            print(f"\n--- {ds_name}: no candidates defined, skipping ---")
            continue

        n_folds = DATASET_CONFIGS[ds_name]["cv_folds"]
        print(f"\n--- {ds_name} ({len(images)} images, {n_folds}-fold CV, {len(candidates)} candidates) ---")

        # Clear old results
        clear_results("final_candidates", ds_name)

        results = []
        for cand in candidates:
            desc = f"{ds_name}|{cand['name']}"
            folds = cross_validate(
                images, labels, n_folds,
                cand["feature"], cand["feature_params"],
                cand["classifier"],
                augment_cfg=cand.get("augment"),
                preprocess_cfg=cand.get("preprocess"),
                desc=desc,
            )
            summary = summarize_folds(folds)

            row = {
                "name": cand["name"],
                "feature": cand["feature"],
                "feature_params": cand["feature_params"],
                "classifier": cand["classifier"],
                "augmentation": cand.get("augment", "none"),
                "test_acc": summary["test_acc_mean"],
                "ci95": summary["test_acc_ci95"],
                "macro_f1": summary["macro_f1_mean"],
                "fps": summary["fps_mean"],
                "train_time": summary["train_time_mean"],
                "feature_dim": summary["feature_dim"],
                "fold_test_accs": summary["fold_test_accs"],
            }
            results.append(row)
            print(f"  {cand['name']:30s} acc={summary['test_acc_mean']*100:.2f}% +/- {summary['test_acc_ci95']*100:.2f}")

            # Save per-fold results
            for fold_i, acc in enumerate(summary["fold_test_accs"]):
                save_result({
                    "feature": cand["feature"], "feature_params": cand["feature_params"],
                    "preprocess": "none", "classifier": cand["classifier"]["type"],
                    "classifier_params": cand["classifier"],
                    "augmentation": str(cand.get("augment", "none")),
                    "fold": fold_i, "test_acc": acc,
                    "train_acc": summary["train_acc_mean"],
                    "rank5_acc": summary.get("rank5_acc_mean"),
                    "macro_f1": summary["macro_f1_mean"],
                    "precision": summary["precision_mean"],
                    "recall": summary["recall_mean"],
                    "fps": summary["fps_mean"],
                    "train_time": summary["train_time_mean"],
                    "feature_dim": summary["feature_dim"],
                }, "final_candidates", ds_name)

        # Sort by accuracy
        results.sort(key=lambda r: r["test_acc"] or 0, reverse=True)

        # Pairwise t-tests between top candidates
        sig_results = []
        if len(results) > 1:
            print(f"\n  Pairwise t-tests (alpha=0.05):")
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    a = np.array(results[i]["fold_test_accs"])
                    b = np.array(results[j]["fold_test_accs"])
                    if len(a) == len(b) and len(a) > 1:
                        t_stat, p_val = stats.ttest_rel(a, b)
                        sig = "*" if p_val < 0.05 else ""
                        diff = results[i]["test_acc"] - results[j]["test_acc"]
                        sig_results.append({
                            "method_a": results[i]["name"], "method_b": results[j]["name"],
                            "mean_diff": float(diff), "t_stat": float(t_stat), "p_value": float(p_val),
                            "significant": p_val < 0.05,
                        })
                        print(f"    {results[i]['name']:25s} vs {results[j]['name']:25s} diff={diff*100:+.2f}% p={p_val:.4f} {sig}")

        print(f"\n  Final ranking on {ds_name}:")
        for rank, r in enumerate(results, 1):
            print(f"    #{rank}: {r['name']} ({r['test_acc']*100:.2f}%)")

        save_summary({
            "ranking": [{"rank": i + 1, "name": r["name"], "accuracy": r["test_acc"],
                         "ci95": r["ci95"], "macro_f1": r["macro_f1"]}
                        for i, r in enumerate(results)],
            "pairwise_tests": sig_results,
            "fold_accuracies": {r["name"]: [float(v) for v in r["fold_test_accs"]] for r in results},
        }, "final_candidates", ds_name)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


EXPERIMENT_MAP = {
    "feature": run_exp_feature,
    "preprocess": run_exp_preprocess,
    "classifier": run_exp_classifier,
    "params": run_exp_params,
    "augment": run_exp_augment,
    "robustness": run_exp_robustness,
    "fusion": run_exp_fusion,
    "stats": run_exp_stats,
    "final_candidates": run_exp_final_candidates,
}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Traditional Face Recognition Full Ablation Suite")
    parser.add_argument("--experiment", "-e", required=True,
                        choices=list(EXPERIMENT_MAP.keys()) + ["all"],
                        help="Which experiment to run")
    parser.add_argument("--dataset", "-d", nargs="*", default=None,
                        help="Datasets to use (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduced configs for smoke testing")
    parser.add_argument("--folds", type=int, default=None,
                        help="Override CV fold count for all datasets (e.g., --folds 5)")
    args = parser.parse_args()

    if args.folds:
        for ds in DATASET_CONFIGS:
            DATASET_CONFIGS[ds]["cv_folds"] = args.folds

    print_header("Loading datasets")
    datasets = load_all_datasets(args.dataset)
    if not datasets:
        print("ERROR: No datasets found!")
        sys.exit(1)

    # Determine which experiments to run
    if args.experiment == "all":
        exp_names = list(EXPERIMENT_MAP.keys())
    else:
        exp_names = [args.experiment]

    # Clear old CSV results for experiments about to run (avoids duplicate rows)
    ds_list = args.dataset or ["FERET", "Yale", "Olivetti"]
    for exp_name in exp_names:
        for ds in ds_list:
            clear_results(exp_name, ds)

    clear_overwrite_session()

    for exp_name in exp_names:
        EXPERIMENT_MAP[exp_name](datasets, quick=args.quick)

    print("\n" + "=" * 72)
    print("  Done! Results saved to results/ directory.")
    print("=" * 72)


if __name__ == "__main__":
    main()
