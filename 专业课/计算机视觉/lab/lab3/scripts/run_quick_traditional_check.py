"""
Quick Traditional Check (Smoke Test)
=====================================
Runs a minimal version of each experiment to verify code correctness.
Use this before running the full ablation suite.

Usage:
    uv run python scripts/run_quick_traditional_check.py
"""

import os
import sys
import time
import warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sklearn.model_selection import StratifiedKFold

from face_recognition_system import (
    EigenfacesExtractor,
    LBPExtractor,
    GaborExtractor,
    HOGExtractor,
    CombinedExtractor,
    augment_dataset,
    add_gaussian_noise,
    simulate_occlusion,
    simulate_illumination_change,
    load_feret_dataset,
    preprocess_histogram_eq,
    preprocess_clahe,
    preprocess_gaussian_filter,
    preprocess_median_filter,
)

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


def load_smallest_dataset():
    """Load the smallest available dataset for quick testing."""
    # Try Olivetti first (smallest: 400 images)
    op = os.path.join(DATA_DIR, "olivetti_images.npy")
    lp = os.path.join(DATA_DIR, "olivetti_labels.npy")
    if os.path.exists(op) and os.path.exists(lp):
        imgs, lbls = np.load(op), np.load(lp)
        if imgs.shape[1:] != (80, 80):
            import cv2
            imgs = np.array([cv2.resize(im, (80, 80)) for im in imgs])
        return "Olivetti", imgs, lbls

    # Fallback to FERET
    feret_dir = os.path.join(DATA_DIR, "数据库-feret_k175_s7_w80_h80", "feret_k175_s7_w80_h80")
    if os.path.exists(feret_dir):
        imgs, lbls = load_feret_dataset(feret_dir)
        # Use only first 20 people for speed
        mask = lbls <= 20
        return "FERET-subset", imgs[mask], lbls[mask]

    raise FileNotFoundError("No dataset found in data/")


def check(name, fn):
    """Run a check function and report pass/fail."""
    try:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  PASS  {name} ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("  Quick Traditional Check (Smoke Test)")
    print("=" * 60)

    # Load dataset
    print("\n[1] Loading dataset...")
    ds_name, images, labels = load_smallest_dataset()
    n_classes = len(np.unique(labels))
    print(f"    {ds_name}: {len(images)} images, {n_classes} classes, shape={images[0].shape}")

    # Use 2-fold CV for speed
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    train_idx, test_idx = next(cv.split(images, labels))
    X_tr, y_tr = images[train_idx], labels[train_idx]
    X_te, y_te = images[test_idx], labels[test_idx]
    print(f"    Train: {len(X_tr)}, Test: {len(X_te)}")

    passed = 0
    total = 0

    # Check feature extractors
    print("\n[2] Feature Extractors")
    extractors = {
        "EigenfacesExtractor(50)": lambda: EigenfacesExtractor(n_components=50),
        "EigenfacesExtractor(100)": lambda: EigenfacesExtractor(n_components=100),
        "LBPExtractor(3x3)": lambda: LBPExtractor(grid_x=3, grid_y=3),
        "LBPExtractor(4x4)": lambda: LBPExtractor(grid_x=4, grid_y=4),
        "GaborExtractor()": lambda: GaborExtractor(),
        "HOGExtractor()": lambda: HOGExtractor(),
        "CombinedExtractor(50+100)": lambda: CombinedExtractor(eigen_components=50, lbp_components=100),
    }
    for name, make_ext in extractors.items():
        def test_ext():
            ext = make_ext()
            feat_tr = ext.fit_transform(X_tr)
            feat_te = ext.transform(X_te)
            assert feat_tr.shape[0] == len(X_tr), f"Train shape mismatch: {feat_tr.shape}"
            assert feat_te.shape[0] == len(X_te), f"Test shape mismatch: {feat_te.shape}"
            assert feat_tr.shape[1] == feat_te.shape[1], f"Feature dim mismatch"
            assert not np.any(np.isnan(feat_tr)), "NaN in train features"
            assert not np.any(np.isnan(feat_te)), "NaN in test features"
            print(f"      dim={feat_tr.shape[1]}")
        total += 1
        if check(name, test_ext):
            passed += 1

    # Check preprocessing
    print("\n[3] Preprocessing Methods")
    pp_funcs = {
        "none": None,
        "hist_eq": preprocess_histogram_eq,
        "clahe": preprocess_clahe,
        "gaussian": preprocess_gaussian_filter,
        "median": preprocess_median_filter,
    }
    for pp_name, pp_func in pp_funcs.items():
        def test_pp(_fn=pp_func, _name=pp_name):
            batch = images[:5]
            result = _fn(batch) if _fn else batch
            assert result.shape == batch.shape, f"Shape mismatch after {_name}: {result.shape} vs {batch.shape}"
            assert not np.any(np.isnan(result)), f"NaN after {_name}"
        total += 1
        if check(f"preprocess({pp_name})", test_pp):
            passed += 1

    # Check augmentation
    print("\n[4] Data Augmentation")
    def test_augment():
        aug_imgs, aug_labels = augment_dataset(X_tr[:10], y_tr[:10], flips=True, angles=(-5, 5))
        assert len(aug_imgs) > len(X_tr[:10]), "Augmentation didn't increase dataset size"
        assert len(aug_imgs) == len(aug_labels), "Image/label count mismatch after augmentation"
        assert not np.any(np.isnan(aug_imgs)), "NaN after augmentation"
    total += 1
    if check("augment_dataset(flips+rot5)", test_augment):
        passed += 1

    def test_augment_no_flip():
        aug_imgs, aug_labels = augment_dataset(X_tr[:10], y_tr[:10], flips=False, angles=(-10, -5, 5, 10))
        assert len(aug_imgs) == len(X_tr[:10]) * 5, f"Expected 5x, got {len(aug_imgs)}"
    total += 1
    if check("augment_dataset(rot5+10, no flip)", test_augment_no_flip):
        passed += 1

    # Check degradation functions
    print("\n[5] Degradation Functions")
    def test_gaussian():
        degraded = add_gaussian_noise(images[:5], sigma=25)
        assert degraded.shape == images[:5].shape
        assert not np.array_equal(degraded, images[:5]), "Noise had no effect"
    total += 1
    if check("add_gaussian_noise", test_gaussian):
        passed += 1

    def test_occlusion():
        degraded = simulate_occlusion(images[:5], occlusion_ratio=0.2)
        assert degraded.shape == images[:5].shape
    total += 1
    if check("simulate_occlusion", test_occlusion):
        passed += 1

    def test_gamma():
        degraded = simulate_illumination_change(images[:5], gamma=0.5)
        assert degraded.shape == images[:5].shape
    total += 1
    if check("simulate_illumination_change", test_gamma):
        passed += 1

    # Check end-to-end pipeline (train + predict)
    print("\n[6] End-to-End Pipeline")
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    def test_e2e():
        ext = EigenfacesExtractor(n_components=50)
        feat_tr = ext.fit_transform(X_tr)
        feat_te = ext.transform(X_te)
        scaler = StandardScaler()
        feat_tr = scaler.fit_transform(feat_tr)
        feat_te = scaler.transform(feat_te)
        clf = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", probability=True)
        clf.fit(feat_tr, y_tr)
        y_pred = clf.predict(feat_te)
        acc = accuracy_score(y_te, y_pred)
        assert 0 < acc < 1, f"Unexpected accuracy: {acc}"
        proba = clf.predict_proba(feat_te)
        assert proba.shape == (len(X_te), n_classes)
        print(f"      accuracy={acc:.3f}")
    total += 1
    if check("Eigenfaces+SVM end-to-end", test_e2e):
        passed += 1

    def test_e2e_combined():
        ext = CombinedExtractor(eigen_components=50, lbp_components=100)
        feat_tr = ext.fit_transform(X_tr)
        feat_te = ext.transform(X_te)
        scaler = StandardScaler()
        feat_tr = scaler.fit_transform(feat_tr)
        feat_te = scaler.transform(feat_te)
        clf = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")
        clf.fit(feat_tr, y_tr)
        y_pred = clf.predict(feat_te)
        acc = accuracy_score(y_te, y_pred)
        print(f"      accuracy={acc:.3f}, dim={feat_tr.shape[1]}")
    total += 1
    if check("Combined+SVM end-to-end", test_e2e_combined):
        passed += 1

    # Check results_io
    print("\n[7] Results I/O")
    from scripts.results_io import save_result, load_results, RESULTS_DIR

    def test_results_io():
        save_result({
            "feature": "test", "feature_params": {}, "preprocess": "none",
            "classifier": "svm_rbf", "classifier_params": {}, "augmentation": "none",
            "fold": 0, "test_acc": 0.95, "train_acc": 0.98, "rank5_acc": None,
            "macro_f1": 0.94, "precision": 0.93, "recall": 0.95, "fps": 100,
            "train_time": 0.1, "feature_dim": 50,
        }, "smoke_test", "Olivetti")
        rows = load_results("smoke_test", "Olivetti")
        assert len(rows) >= 1, "No rows loaded"
        assert float(rows[-1]["test_acc"]) == 0.95, f"Acc mismatch: {rows[-1]['test_acc']}"
        # Clean up
        test_file = RESULTS_DIR / "smoke_test_Olivetti.csv"
        if test_file.exists():
            test_file.unlink()
    total += 1
    if check("results_io save/load", test_results_io):
        passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  All checks passed! Ready to run full experiments.")
    else:
        print("  Some checks failed. Fix issues before running full experiments.")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
