"""
Academic Face Recognition Experiments
======================================
Comprehensive evaluation with multiple protocols, feature comparisons,
classifier sweeps, augmentation ablation, and statistical rigor.

Usage:
    uv run python run_academic_experiments.py
"""

import os
import sys
import time
import warnings
from itertools import product

# Add project root to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _PROJECT_ROOT)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition_system import (
    augment_dataset,
    load_feret_dataset,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
FERET_DIR = os.path.join(_DATA_DIR, "数据库-feret_k175_s7_w80_h80", "feret_k175_s7_w80_h80")
YALE_IMAGES_PATH = os.path.join(_DATA_DIR, "yale_images.npy")
YALE_LABELS_PATH = os.path.join(_DATA_DIR, "yale_labels.npy")
OLIVETTI_IMAGES_PATH = os.path.join(_DATA_DIR, "olivetti_images.npy")
OLIVETTI_LABELS_PATH = os.path.join(_DATA_DIR, "olivetti_labels.npy")

IMG_SIZE = (80, 80)

# ── Progress helpers ───────────────────────────────────────────────────────

def progress_bar(current, total, width=40, prefix=""):
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r  {prefix}[{bar}] {current}/{total} ({pct:.0%})", end="", flush=True)
    if current == total:
        print()


def print_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_md_table(headers, rows, col_formats=None):
    """Print a clean markdown table."""
    n_cols = len(headers)
    if col_formats is None:
        col_formats = [None] * n_cols

    def fmt(val, fmt_spec, col_idx):
        if fmt_spec is not None:
            return fmt_spec.format(val)
        return str(val)

    str_rows = []
    for row in rows:
        str_rows.append([fmt(v, col_formats[i], i) for i, v in enumerate(row)])
    str_header = [str(h) for h in headers]

    widths = []
    for i in range(n_cols):
        w = len(str_header[i])
        for row in str_rows:
            w = max(w, len(row[i]))
        widths.append(w + 2)

    hdr_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(str_header)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(n_cols)) + " |"
    print(hdr_line)
    print(sep_line)
    for row in str_rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(n_cols)) + " |")


# ── Feature extractors (inline implementations) ───────────────────────────

class EigenfacesExtractor:
    """PCA-based eigenface features."""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None
        self.mean_face = None

    def fit(self, X):
        flat = X.reshape(X.shape[0], -1).astype(np.float64)
        self.mean_face = flat.mean(axis=0)
        flat = flat - self.mean_face
        n_comp = min(self.n_components, X.shape[0] - 1, flat.shape[1])
        self.pca = PCA(n_components=n_comp, whiten=True)
        self.pca.fit(flat)
        return self

    def transform(self, X):
        flat = X.reshape(X.shape[0], -1).astype(np.float64)
        flat = flat - self.mean_face
        return self.pca.transform(flat)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LBPExtractor:
    """LBP features with spatial grid histograms."""

    def __init__(self, grid_x=3, grid_y=3):
        self.grid_x = grid_x
        self.grid_y = grid_y

    def _compute_lbp(self, img):
        h, w = img.shape
        center = img[1 : h - 1, 1 : w - 1].astype(np.int32)
        code = np.zeros_like(center, dtype=np.uint8)
        for k, (dy, dx) in enumerate(
            [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        ):
            neighbor = img[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx].astype(np.int32)
            code |= ((neighbor >= center).astype(np.uint8) << k)
        return code

    def _histogram(self, img):
        lbp = self._compute_lbp(img)
        h, w = lbp.shape
        ch, cw = h // self.grid_y, w // self.grid_x
        hists = []
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                cell = lbp[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw]
                hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256), density=True)
                hists.append(hist)
        return np.concatenate(hists)

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([self._histogram(img) for img in X])

    def fit_transform(self, X):
        return self.transform(X)


class GaborExtractor:
    """Gabor filter bank features with spatial statistics."""

    def __init__(self, frequencies=(0.1, 0.2, 0.3), orientations=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
                 grid_x=2, grid_y=2):
        self.frequencies = frequencies
        self.orientations = orientations
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.kernels = []

    def _create_kernels(self):
        self.kernels = []
        for freq in self.frequencies:
            for theta in self.orientations:
                k = cv2.getGaborKernel((21, 21), 4.0, theta, 1.0 / freq, 0.5, 0, ktype=cv2.CV_32F)
                self.kernels.append(k)

    def _extract(self, img):
        features = []
        h, w = img.shape
        ch, cw = h // self.grid_y, w // self.grid_x
        for kernel in self.kernels:
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            for i in range(self.grid_y):
                for j in range(self.grid_x):
                    cell = filtered[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw]
                    features.append(cell.mean())
                    features.append(cell.std())
        return np.array(features)

    def fit(self, X):
        self._create_kernels()
        return self

    def transform(self, X):
        if not self.kernels:
            self._create_kernels()
        return np.array([self._extract(img) for img in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CombinedExtractor:
    """Eigenfaces + LBP (with PCA on LBP)."""

    def __init__(self, eigen_components=50, lbp_components=100, lbp_grid=3):
        self.eigen = EigenfacesExtractor(n_components=eigen_components)
        self.lbp = LBPExtractor(grid_x=lbp_grid, grid_y=lbp_grid)
        self.lbp_scaler = StandardScaler()
        self.lbp_pca = PCA(n_components=lbp_components)

    def fit(self, X):
        self.eigen.fit(X)
        lbp_raw = self.lbp.transform(X)
        lbp_scaled = self.lbp_scaler.fit_transform(lbp_raw)
        n_comp = min(self.lbp_pca.n_components, lbp_scaled.shape[0] - 1, lbp_scaled.shape[1])
        self.lbp_pca = PCA(n_components=n_comp)
        self.lbp_pca.fit(lbp_scaled)
        return self

    def transform(self, X):
        eigen_f = self.eigen.transform(X)
        lbp_raw = self.lbp.transform(X)
        lbp_scaled = self.lbp_scaler.transform(lbp_raw)
        lbp_f = self.lbp_pca.transform(lbp_scaled)
        return np.hstack([eigen_f, lbp_f])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class HOGExtractor:
    """HOG (Histogram of Oriented Gradients) features using cv2.HOGDescriptor."""

    def __init__(self, win_size=(80, 80), block_size=(16, 16), block_stride=(8, 8),
                 cell_size=(8, 8), nbins=9):
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        self.hog = None

    def _create_hog(self):
        self.hog = cv2.HOGDescriptor(
            self.win_size, self.block_size, self.block_stride,
            self.cell_size, self.nbins,
        )

    def fit(self, X):
        self._create_hog()
        return self

    def transform(self, X):
        if self.hog is None:
            self._create_hog()
        features = []
        for img in X:
            feat = self.hog.compute(img)
            features.append(feat.flatten())
        return np.array(features)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ── Extractor factory ─────────────────────────────────────────────────────

def create_extractor(name, **kwargs):
    if name == "eigenfaces":
        return EigenfacesExtractor(n_components=kwargs.get("n_components", 50))
    elif name == "lbp":
        return LBPExtractor(grid_x=kwargs.get("grid_x", 3), grid_y=kwargs.get("grid_y", 3))
    elif name == "gabor":
        return GaborExtractor()
    elif name == "combined":
        return CombinedExtractor(
            eigen_components=kwargs.get("eigen_components", 50),
            lbp_components=kwargs.get("lbp_components", 100),
            lbp_grid=kwargs.get("lbp_grid", 3),
        )
    elif name == "hog":
        return HOGExtractor()
    else:
        raise ValueError(f"Unknown extractor: {name}")


# ── Classifier factory ────────────────────────────────────────────────────

def create_classifier(name, **kwargs):
    if name == "svm_linear":
        return SVC(kernel="linear", C=kwargs.get("C", 1.0), class_weight="balanced", probability=True)
    elif name == "svm_rbf":
        return SVC(kernel="rbf", C=kwargs.get("C", 1.0), gamma=kwargs.get("gamma", "scale"),
                   class_weight="balanced", probability=True)
    elif name == "knn":
        return KNeighborsClassifier(
            n_neighbors=kwargs.get("k", 3), weights="distance",
            metric=kwargs.get("metric", "cosine"),
        )
    elif name == "rf":
        return RandomForestClassifier(n_estimators=kwargs.get("n_estimators", 100), random_state=42, n_jobs=-1)
    elif name == "logistic":
        return LogisticRegression(
            penalty=kwargs.get("penalty", "l2"), C=kwargs.get("C", 1.0),
            max_iter=2000, solver="saga", random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ── Augmentation configurations ───────────────────────────────────────────

AUGMENT_CONFIGS = {
    "none": {"flips": False, "rotations": False},
    "rot5": {"flips": False, "rotations": True, "angles": (-5, 5)},
    "rot5+10": {"flips": False, "rotations": True, "angles": (-10, -5, 5, 10)},
    "rot5+10+15": {"flips": False, "rotations": True, "angles": (-15, -10, -5, 5, 10, 15)},
    "flip+rot5+10": {"flips": True, "rotations": True, "angles": (-10, -5, 5, 10)},
}


# ── Core evaluation function ──────────────────────────────────────────────

def evaluate_fold(X_train_img, y_train, X_test_img, y_test, extractor_name, extractor_params,
                  classifier_name, classifier_params, augment_cfg_name="none"):
    """
    Train on one fold and return dict of metrics.
    Augmentation is applied ONLY to the training images within this fold.
    """
    # Augment training data
    cfg = AUGMENT_CONFIGS[augment_cfg_name]
    if cfg.get("rotations") or cfg.get("flips"):
        X_train_img, y_train = augment_dataset(
            X_train_img, y_train,
            flips=cfg.get("flips", False),
            rotations=cfg.get("rotations", True),
            angles=cfg.get("angles", (-10, -5, 5, 10)),
        )

    # Extract features
    extractor = create_extractor(extractor_name, **extractor_params)
    X_train_feat = extractor.fit_transform(X_train_img)
    X_test_feat = extractor.transform(X_test_img)

    # Scale
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)

    # Train classifier
    clf = create_classifier(classifier_name, **classifier_params)
    clf.fit(X_train_feat, y_train)

    # Predictions + FPS measurement
    import time as _time
    t0 = _time.perf_counter()
    y_pred = clf.predict(X_test_feat)
    t1 = _time.perf_counter()
    n_test = len(X_test_feat)
    fps = n_test / (t1 - t0) if (t1 - t0) > 0 else 0
    n_classes = len(np.unique(y_train))

    # Basic metrics
    train_acc = clf.score(X_train_feat, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    # Top-k accuracy (requires predict_proba)
    rank5_acc = None
    rank10_acc = None
    if hasattr(clf, 'predict_proba') and n_classes >= 5:
        proba = clf.predict_proba(X_test_feat)
        # top_k_accuracy_score needs labels parameter to match proba columns
        rank5_acc = top_k_accuracy_score(y_test, proba, k=5, labels=clf.classes_)
        if n_classes >= 10:
            rank10_acc = top_k_accuracy_score(y_test, proba, k=10, labels=clf.classes_)

    # Classification metrics (macro and weighted)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'rank5_acc': rank5_acc,
        'rank10_acc': rank10_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'fps': fps,
    }


def cross_validate(images, labels, n_folds, extractor_name, extractor_params,
                   classifier_name, classifier_params, augment_cfg_name="none",
                   seed=42, desc=""):
    """
    Stratified k-fold cross-validation with per-fold augmentation.
    Returns list of (train_acc, test_acc) per fold.
    """
    n_classes = len(np.unique(labels))
    actual_folds = min(n_folds, min(np.unique(labels, return_counts=True)[1]))
    if actual_folds < 2:
        actual_folds = 2
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(images, labels)):
        if desc:
            progress_bar(fold_idx + 1, actual_folds, prefix=f"{desc} fold ")
        X_tr, y_tr = images[train_idx], labels[train_idx]
        X_te, y_te = images[test_idx], labels[test_idx]
        metrics = evaluate_fold(
            X_tr, y_tr, X_te, y_te,
            extractor_name, extractor_params,
            classifier_name, classifier_params,
            augment_cfg_name,
        )
        fold_results.append(metrics)
    return fold_results


def summarize_folds(fold_results):
    """Return dict with mean, std, ci95 for all metrics across folds."""
    n = len(fold_results)
    t_val = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96

    def _agg(key):
        vals = [r[key] for r in fold_results if r[key] is not None]
        if not vals:
            return None
        m = np.mean(vals)
        s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        ci = t_val * s / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "ci95": ci}

    result = {}
    for key in ['train_acc', 'test_acc', 'rank5_acc', 'rank10_acc',
                 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall', 'fps']:
        agg = _agg(key)
        if agg:
            result[f"{key}_mean"] = agg["mean"]
            result[f"{key}_std"] = agg["std"]
            result[f"{key}_ci95"] = agg["ci95"]
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_std"] = None
            result[f"{key}_ci95"] = None

    result["gap"] = result.get("train_acc_mean", 0) - result.get("test_acc_mean", 0)
    result["fold_test_accs"] = [r["test_acc"] for r in fold_results]
    return result


def fmt_pct(mean, std=None, ci=None):
    """Format as percentage string."""
    if mean is None:
        return "N/A"
    if ci is not None:
        return f"{mean * 100:.2f}% +/- {ci * 100:.2f}"
    if std is not None:
        return f"{mean * 100:.2f}% +/- {std * 100:.2f}"
    return f"{mean * 100:.2f}%"


def fmt_metric(summary, key):
    """Format a metric from summary dict (key_mean, key_ci95)."""
    m = summary.get(f"{key}_mean")
    c = summary.get(f"{key}_ci95")
    if m is None:
        return "N/A"
    return fmt_pct(m, ci=c)


# ── Datasets ───────────────────────────────────────────────────────────────

def load_all_datasets():
    """Load all available datasets. Returns dict of name -> (images, labels)."""
    datasets = {}

    # FERET
    if os.path.exists(FERET_DIR):
        print("  Loading FERET...")
        imgs, lbls = load_feret_dataset(FERET_DIR, img_size=IMG_SIZE)
        datasets["FERET"] = (imgs, lbls)
        n_people = len(np.unique(lbls))
        print(f"    {len(imgs)} images, {n_people} people, {imgs[0].shape}")

    # Yale
    if os.path.exists(YALE_IMAGES_PATH) and os.path.exists(YALE_LABELS_PATH):
        print("  Loading Yale...")
        imgs = np.load(YALE_IMAGES_PATH)
        lbls = np.load(YALE_LABELS_PATH)
        # Resize to 80x80 if needed
        if imgs.shape[1:] != IMG_SIZE:
            resized = []
            for img in imgs:
                resized.append(cv2.resize(img, IMG_SIZE))
            imgs = np.array(resized)
        datasets["Yale"] = (imgs, lbls)
        n_people = len(np.unique(lbls))
        print(f"    {len(imgs)} images, {n_people} people, {imgs[0].shape}")

    # Olivetti
    if os.path.exists(OLIVETTI_IMAGES_PATH) and os.path.exists(OLIVETTI_LABELS_PATH):
        print("  Loading Olivetti...")
        imgs = np.load(OLIVETTI_IMAGES_PATH)
        lbls = np.load(OLIVETTI_LABELS_PATH)
        if imgs.shape[1:] != IMG_SIZE:
            resized = []
            for img in imgs:
                resized.append(cv2.resize(img, IMG_SIZE))
            imgs = np.array(resized)
        datasets["Olivetti"] = (imgs, lbls)
        n_people = len(np.unique(lbls))
        print(f"    {len(imgs)} images, {n_people} people, {imgs[0].shape}")

    return datasets


# ── Experiment 1: Feature Comparison ──────────────────────────────────────

def run_feature_comparison(datasets):
    """
    Systematic comparison of feature extractors across datasets and CV protocols.
    Fixed: SVM RBF (C=1, gamma=scale), no augmentation.
    """
    print_header("Experiment 1: Feature Extractor Comparison")

    feature_configs = [
        ("Eigenfaces-20",  "eigenfaces", {"n_components": 20}),
        ("Eigenfaces-30",  "eigenfaces", {"n_components": 30}),
        ("Eigenfaces-50",  "eigenfaces", {"n_components": 50}),
        ("Eigenfaces-80",  "eigenfaces", {"n_components": 80}),
        ("LBP-2x2",        "lbp",        {"grid_x": 2, "grid_y": 2}),
        ("LBP-3x3",        "lbp",        {"grid_x": 3, "grid_y": 3}),
        ("LBP-4x4",        "lbp",        {"grid_x": 4, "grid_y": 4}),
        ("Gabor",          "gabor",       {}),
        ("HOG",            "hog",         {}),
        ("Combined-50+100","combined",    {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3}),
        ("Combined-80+150","combined",    {"eigen_components": 80, "lbp_components": 150, "lbp_grid": 3}),
    ]

    # CV protocols per dataset
    cv_protocols = {
        "FERET": 7,
        "Yale": 10,
        "Olivetti": 10,
    }

    all_results = {}  # dataset_name -> list of (name, summary)

    for ds_name, (images, labels) in datasets.items():
        n_folds = cv_protocols.get(ds_name, 10)
        print(f"\n  --- {ds_name} ({n_folds}-fold CV) ---")
        ds_results = []
        total = len(feature_configs)
        for idx, (feat_name, ext_name, ext_params) in enumerate(feature_configs):
            progress_bar(idx, total, prefix=f"  {feat_name:<20s}")
            t0 = time.time()
            folds = cross_validate(
                images, labels, n_folds,
                ext_name, ext_params,
                "svm_rbf", {"C": 1.0, "gamma": "scale"},
                augment_cfg_name="none",
                desc="",
            )
            summary = summarize_folds(folds)
            elapsed = time.time() - t0
            ds_results.append((feat_name, summary, elapsed))
            progress_bar(total, total, prefix=f"  {feat_name:<20s}")
            print(f"    Rank-1={fmt_metric(summary, 'test_acc')}, "
                  f"Rank-5={fmt_metric(summary, 'rank5_acc')}, "
                  f"F1={fmt_metric(summary, 'macro_f1')}, "
                  f"FPS={summary.get('fps_mean', 0):.0f}")
        all_results[ds_name] = ds_results

    # Print tables
    for ds_name, results in all_results.items():
        print(f"\n  ** {ds_name} **")
        headers = ["Feature", "Rank-1 Acc (95% CI)", "Rank-5 Acc (95% CI)", "Macro F1 (95% CI)", "FPS"]
        rows = []
        for name, s, t in results:
            rows.append([
                name,
                fmt_metric(s, "test_acc"),
                fmt_metric(s, "rank5_acc"),
                fmt_metric(s, "macro_f1"),
                f"{s.get('fps_mean', 0):.0f} +/- {s.get('fps_std', 0):.0f}",
            ])
        print_md_table(headers, rows)

    return all_results


# ── Experiment 2: Classifier Comparison ───────────────────────────────────

def run_classifier_comparison(datasets):
    """
    Compare classifiers with hyperparameter sweeps.
    Feature: Combined (50+100), no augmentation.
    """
    print_header("Experiment 2: Classifier Comparison (Combined features)")

    classifier_configs = [
        # SVM Linear
        ("SVM-Linear C=0.01", "svm_linear", {"C": 0.01}),
        ("SVM-Linear C=0.1",  "svm_linear", {"C": 0.1}),
        ("SVM-Linear C=1",    "svm_linear", {"C": 1.0}),
        ("SVM-Linear C=10",   "svm_linear", {"C": 10.0}),
        # SVM RBF
        ("SVM-RBF C=0.1,g=scale",  "svm_rbf", {"C": 0.1, "gamma": "scale"}),
        ("SVM-RBF C=1,g=scale",    "svm_rbf", {"C": 1.0, "gamma": "scale"}),
        ("SVM-RBF C=10,g=scale",   "svm_rbf", {"C": 10.0, "gamma": "scale"}),
        ("SVM-RBF C=1,g=0.001",    "svm_rbf", {"C": 1.0, "gamma": 0.001}),
        ("SVM-RBF C=1,g=0.01",     "svm_rbf", {"C": 1.0, "gamma": 0.01}),
        # KNN
        ("KNN k=1 cosine",  "knn", {"k": 1, "metric": "cosine"}),
        ("KNN k=3 cosine",  "knn", {"k": 3, "metric": "cosine"}),
        ("KNN k=5 cosine",  "knn", {"k": 5, "metric": "cosine"}),
        ("KNN k=3 euclid",  "knn", {"k": 3, "metric": "euclidean"}),
        ("KNN k=5 euclid",  "knn", {"k": 5, "metric": "euclidean"}),
        # Random Forest
        ("RF n=50",   "rf", {"n_estimators": 50}),
        ("RF n=100",  "rf", {"n_estimators": 100}),
        ("RF n=200",  "rf", {"n_estimators": 200}),
        # Logistic Regression
        ("LR L2 C=1",    "logistic", {"penalty": "l2", "C": 1.0}),
        ("LR L2 C=0.1",  "logistic", {"penalty": "l2", "C": 0.1}),
        ("LR L2 C=10",   "logistic", {"penalty": "l2", "C": 10.0}),
    ]

    cv_protocols = {"FERET": 7, "Yale": 10, "Olivetti": 10}
    all_results = {}

    for ds_name, (images, labels) in datasets.items():
        n_folds = cv_protocols.get(ds_name, 10)
        print(f"\n  --- {ds_name} ({n_folds}-fold CV) ---")
        ds_results = []
        total = len(classifier_configs)
        for idx, (clf_name, clf_key, clf_params) in enumerate(classifier_configs):
            progress_bar(idx, total, prefix=f"  {clf_name:<25s}")
            t0 = time.time()
            folds = cross_validate(
                images, labels, n_folds,
                "combined", {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
                clf_key, clf_params,
                augment_cfg_name="none",
            )
            summary = summarize_folds(folds)
            elapsed = time.time() - t0
            ds_results.append((clf_name, summary, elapsed))
            progress_bar(total, total, prefix=f"  {clf_name:<25s}")
            print(f"    Rank-1={fmt_metric(summary, 'test_acc')}, "
                  f"Rank-5={fmt_metric(summary, 'rank5_acc')}, "
                  f"F1={fmt_metric(summary, 'macro_f1')}, "
                  f"FPS={summary.get('fps_mean', 0):.0f}")
        all_results[ds_name] = ds_results

    for ds_name, results in all_results.items():
        print(f"\n  ** {ds_name} **")
        headers = ["Classifier", "Rank-1 Acc (95% CI)", "Rank-5 Acc (95% CI)", "Macro F1 (95% CI)", "FPS"]
        rows = []
        for name, s, t in results:
            rows.append([
                name,
                fmt_metric(s, "test_acc"),
                fmt_metric(s, "rank5_acc"),
                fmt_metric(s, "macro_f1"),
                f"{s.get('fps_mean', 0):.0f} +/- {s.get('fps_std', 0):.0f}",
            ])
        print_md_table(headers, rows)

    return all_results


# ── Experiment 3: Augmentation Ablation ───────────────────────────────────

def run_augmentation_ablation(datasets):
    """
    Compare augmentation strategies with proper per-fold augmentation.
    Feature: Combined, Classifier: SVM-RBF(C=1).
    """
    print_header("Experiment 3: Augmentation Ablation")

    aug_configs = [
        "none",
        "rot5",
        "rot5+10",
        "rot5+10+15",
        "flip+rot5+10",
    ]
    aug_labels = {
        "none": "No augmentation",
        "rot5": "Rotation +/-5",
        "rot5+10": "Rotation +/-5,10",
        "rot5+10+15": "Rotation +/-5,10,15",
        "flip+rot5+10": "Flip + Rotation +/-5,10",
    }

    cv_protocols = {"FERET": 7, "Yale": 10, "Olivetti": 10}
    all_results = {}

    for ds_name, (images, labels) in datasets.items():
        n_folds = cv_protocols.get(ds_name, 10)
        print(f"\n  --- {ds_name} ({n_folds}-fold CV) ---")
        ds_results = []
        total = len(aug_configs)
        for idx, aug_name in enumerate(aug_configs):
            progress_bar(idx, total, prefix=f"  {aug_labels[aug_name]:<30s}")
            t0 = time.time()
            folds = cross_validate(
                images, labels, n_folds,
                "combined", {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
                "svm_rbf", {"C": 1.0, "gamma": "scale"},
                augment_cfg_name=aug_name,
            )
            summary = summarize_folds(folds)
            elapsed = time.time() - t0
            ds_results.append((aug_labels[aug_name], summary, elapsed))
            progress_bar(total, total, prefix=f"  {aug_labels[aug_name]:<30s}")
            print(f"    Rank-1={fmt_metric(summary, 'test_acc')}, "
                  f"Rank-5={fmt_metric(summary, 'rank5_acc')}, "
                  f"F1={fmt_metric(summary, 'macro_f1')}, "
                  f"FPS={summary.get('fps_mean', 0):.0f}")
        all_results[ds_name] = ds_results

    for ds_name, results in all_results.items():
        print(f"\n  ** {ds_name} **")
        headers = ["Augmentation", "Rank-1 Acc (95% CI)", "Rank-5 Acc (95% CI)", "Macro F1 (95% CI)", "FPS"]
        rows = []
        for name, s, t in results:
            rows.append([
                name,
                fmt_metric(s, "test_acc"),
                fmt_metric(s, "rank5_acc"),
                fmt_metric(s, "macro_f1"),
                f"{s.get('fps_mean', 0):.0f} +/- {s.get('fps_std', 0):.0f}",
            ])
        print_md_table(headers, rows)

    return all_results


# ── Experiment 4: Overfitting Analysis ────────────────────────────────────

def run_overfitting_analysis(datasets):
    """
    Sweep feature dimensions and classifier complexity.
    Track train vs test accuracy and generalization gap.
    """
    print_header("Experiment 4: Overfitting Analysis")

    cv_protocols = {"FERET": 7, "Yale": 10, "Olivetti": 10}
    all_results = {}

    # Configs designed to probe overfitting
    configs = [
        ("Eigenfaces-20 + SVM-RBF C=10",  "eigenfaces", {"n_components": 20},
         "svm_rbf", {"C": 10.0, "gamma": "scale"}),
        ("Eigenfaces-50 + SVM-RBF C=10",  "eigenfaces", {"n_components": 50},
         "svm_rbf", {"C": 10.0, "gamma": "scale"}),
        ("Eigenfaces-80 + SVM-RBF C=10",  "eigenfaces", {"n_components": 80},
         "svm_rbf", {"C": 10.0, "gamma": "scale"}),
        ("Eigenfaces-50 + SVM-RBF C=0.1", "eigenfaces", {"n_components": 50},
         "svm_rbf", {"C": 0.1, "gamma": "scale"}),
        ("Eigenfaces-50 + SVM-RBF C=1",   "eigenfaces", {"n_components": 50},
         "svm_rbf", {"C": 1.0, "gamma": "scale"}),
        ("Eigenfaces-50 + SVM-RBF C=10",  "eigenfaces", {"n_components": 50},
         "svm_rbf", {"C": 10.0, "gamma": "scale"}),
        ("Eigenfaces-50 + SVM-RBF C=10 g=0.01", "eigenfaces", {"n_components": 50},
         "svm_rbf", {"C": 10.0, "gamma": 0.01}),
        ("LBP-3x3 + SVM-RBF C=1",        "lbp",        {"grid_x": 3, "grid_y": 3},
         "svm_rbf", {"C": 1.0, "gamma": "scale"}),
        ("LBP-3x3 + KNN k=1",            "lbp",        {"grid_x": 3, "grid_y": 3},
         "knn", {"k": 1, "metric": "cosine"}),
        ("LBP-3x3 + KNN k=5",            "lbp",        {"grid_x": 3, "grid_y": 3},
         "knn", {"k": 5, "metric": "cosine"}),
        ("Combined + SVM-RBF C=1",       "combined",   {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
         "svm_rbf", {"C": 1.0, "gamma": "scale"}),
        ("Combined + SVM-RBF C=10",      "combined",   {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
         "svm_rbf", {"C": 10.0, "gamma": "scale"}),
        ("Combined + RF n=200",           "combined",   {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
         "rf", {"n_estimators": 200}),
        ("Combined + KNN k=1",            "combined",   {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
         "knn", {"k": 1, "metric": "cosine"}),
        # Augmented versions for gap comparison
        ("Combined + SVM-RBF C=1 + aug",  "combined",   {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
         "svm_rbf", {"C": 1.0, "gamma": "scale"}, ),  # will use augmented
    ]

    for ds_name, (images, labels) in datasets.items():
        n_folds = cv_protocols.get(ds_name, 10)
        print(f"\n  --- {ds_name} ({n_folds}-fold CV) ---")
        ds_results = []
        total = len(configs)
        for idx, cfg in enumerate(configs):
            name, ext_name, ext_params, clf_name, clf_params = cfg[:5]
            aug_name = "flip+rot5+10" if "aug" in name else "none"
            progress_bar(idx, total, prefix=f"  {name:<40s}")
            folds = cross_validate(
                images, labels, n_folds,
                ext_name, ext_params,
                clf_name, clf_params,
                augment_cfg_name=aug_name,
            )
            summary = summarize_folds(folds)
            ds_results.append((name, summary))
            progress_bar(total, total, prefix=f"  {name:<40s}")
            print(f"    Rank-1={fmt_metric(summary, 'test_acc')}, "
                  f"Rank-5={fmt_metric(summary, 'rank5_acc')}, "
                  f"F1={fmt_metric(summary, 'macro_f1')}, "
                  f"gap={summary['gap']*100:.1f}%")
        all_results[ds_name] = ds_results

    # Print gap analysis table (sorted by gap ascending)
    for ds_name, results in all_results.items():
        print(f"\n  ** {ds_name} - Generalization Gap Analysis (sorted by gap) **")
        headers = ["Configuration", "Rank-1 Acc (95% CI)", "Rank-5 Acc (95% CI)", "Macro F1 (95% CI)", "Gap (%)"]
        sorted_results = sorted(results, key=lambda x: x[1]["gap"])
        rows = []
        for name, s in sorted_results:
            rows.append([
                name,
                fmt_metric(s, "test_acc"),
                fmt_metric(s, "rank5_acc"),
                fmt_metric(s, "macro_f1"),
                f"{s['gap']*100:.1f}",
            ])
        print_md_table(headers, rows)

    return all_results


# ── Experiment 5: Best Model Selection + Statistical Tests ────────────────

def run_statistical_comparison(datasets):
    """
    Compare top-performing method pairs with paired t-test on fold-level accuracy.
    Uses the best augmentation setting found in Experiment 3.
    """
    print_header("Experiment 5: Statistical Comparison (Paired t-tests)")

    cv_protocols = {"FERET": 7, "Yale": 10, "Olivetti": 10}

    # Method pairs to compare
    method_pairs = [
        (
            "Combined + SVM-RBF",
            "combined", {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
            "svm_rbf", {"C": 1.0, "gamma": "scale"},
        ),
        (
            "Combined + SVM-Linear",
            "combined", {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
            "svm_linear", {"C": 1.0},
        ),
        (
            "Eigenfaces-80 + SVM-RBF",
            "eigenfaces", {"n_components": 80},
            "svm_rbf", {"C": 1.0, "gamma": "scale"},
        ),
        (
            "LBP-3x3 + SVM-RBF",
            "lbp", {"grid_x": 3, "grid_y": 3},
            "svm_rbf", {"C": 1.0, "gamma": "scale"},
        ),
        (
            "HOG + SVM-RBF",
            "hog", {},
            "svm_rbf", {"C": 1.0, "gamma": "scale"},
        ),
        (
            "Combined + SVM-RBF + aug",
            "combined", {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3},
            "svm_rbf", {"C": 1.0, "gamma": "scale"},
        ),
    ]

    all_results = {}

    for ds_name, (images, labels) in datasets.items():
        n_folds = cv_protocols.get(ds_name, 10)
        print(f"\n  --- {ds_name} ---")

        # Run all methods, collect per-fold test accuracies
        method_folds = {}
        for m_name, ext_name, ext_params, clf_name, clf_params in method_pairs:
            aug_name = "flip+rot5+10" if "aug" in m_name else "none"
            folds = cross_validate(
                images, labels, n_folds,
                ext_name, ext_params,
                clf_name, clf_params,
                augment_cfg_name=aug_name,
            )
            test_accs = [f["test_acc"] for f in folds]
            f1_scores = [f["macro_f1"] for f in folds]
            method_folds[m_name] = {"test_accs": test_accs, "f1_scores": f1_scores}
            print(f"    {m_name}: Rank-1={np.mean(test_accs)*100:.2f}% +/- {np.std(test_accs, ddof=1)*100:.2f}, "
                  f"F1={np.mean(f1_scores)*100:.2f}%")

        # Find best and second-best by mean test accuracy
        ranked = sorted(method_folds.items(), key=lambda x: np.mean(x[1]["test_accs"]), reverse=True)
        if len(ranked) >= 2:
            best_name, best_data = ranked[0]
            second_name, second_data = ranked[1]
            best_accs = best_data["test_accs"]
            second_accs = second_data["test_accs"]
            t_stat, p_value = stats.ttest_rel(best_accs, second_accs)
            # Cohen's d effect size
            diff = np.array(best_accs) - np.array(second_accs)
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            print(f"\n    Best:      {best_name} ({np.mean(best_accs)*100:.2f}%)")
            print(f"    Second:    {second_name} ({np.mean(second_accs)*100:.2f}%)")
            print(f"    Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
            print(f"    Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
            sig = "YES" if p_value < 0.05 else "NO"
            print(f"    Significant at alpha=0.05? {sig}")

        all_results[ds_name] = {
            "method_folds": method_folds,
            "ranking": [(name, np.mean(d["test_accs"]), np.std(d["test_accs"], ddof=1),
                         np.mean(d["f1_scores"])) for name, d in ranked],
        }

    # Summary table
    for ds_name, data in all_results.items():
        print(f"\n  ** {ds_name} - Method Ranking **")
        headers = ["Rank", "Method", "Rank-1 Acc (%)", "Std (%)", "Macro F1 (%)"]
        rows = []
        for rank, (name, mean, std, f1) in enumerate(data["ranking"], 1):
            rows.append([str(rank), name, f"{mean*100:.2f}", f"{std*100:.2f}", f"{f1*100:.2f}"])
        print_md_table(headers, rows)

    return all_results


# ── Experiment 6: Ensemble Methods ────────────────────────────────────────

def run_ensemble_experiment(datasets):
    """
    Compare individual classifiers vs. a VotingClassifier ensemble.
    """
    print_header("Experiment 6: Ensemble Methods")

    cv_protocols = {"FERET": 7, "Yale": 10, "Olivetti": 10}
    all_results = {}

    for ds_name, (images, labels) in datasets.items():
        n_folds = cv_protocols.get(ds_name, 10)
        print(f"\n  --- {ds_name} ({n_folds}-fold CV) ---")

        # Individual classifiers to ensemble
        base_classifiers = [
            ("SVM-RBF C=1", "svm_rbf", {"C": 1.0, "gamma": "scale"}),
            ("SVM-Linear C=1", "svm_linear", {"C": 1.0}),
            ("KNN k=3", "knn", {"k": 3, "metric": "cosine"}),
            ("RF n=100", "rf", {"n_estimators": 100}),
        ]

        ext_name = "combined"
        ext_params = {"eigen_components": 50, "lbp_components": 100, "lbp_grid": 3}

        # Run individual classifiers
        individual_results = {}
        for clf_name, clf_key, clf_params in base_classifiers:
            folds = cross_validate(
                images, labels, n_folds,
                ext_name, ext_params,
                clf_key, clf_params,
                augment_cfg_name="none",
            )
            summary = summarize_folds(folds)
            individual_results[clf_name] = summary
            print(f"    {clf_name}: Rank-1={fmt_metric(summary, 'test_acc')}, F1={fmt_metric(summary, 'macro_f1')}")

        # Ensemble (VotingClassifier) with custom fold evaluation
        print("    Running VotingClassifier ensemble...")
        actual_folds = min(n_folds, min(np.unique(labels, return_counts=True)[1]))
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        ensemble_folds = []

        for train_idx, test_idx in cv.split(images, labels):
            X_tr, y_tr = images[train_idx], labels[train_idx]
            X_te, y_te = images[test_idx], labels[test_idx]

            extractor = create_extractor(ext_name, **ext_params)
            X_tr_feat = extractor.fit_transform(X_tr)
            X_te_feat = extractor.transform(X_te)

            scaler = StandardScaler()
            X_tr_feat = scaler.fit_transform(X_tr_feat)
            X_te_feat = scaler.transform(X_te_feat)

            estimators = [
                ("svm_rbf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True)),
                ("svm_linear", SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True)),
                ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance", metric="cosine")),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ]
            ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
            ensemble.fit(X_tr_feat, y_tr)
            y_pred = ensemble.predict(X_te_feat)
            train_acc = ensemble.score(X_tr_feat, y_tr)
            test_acc = accuracy_score(y_te, y_pred)
            macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
            ensemble_folds.append({
                'train_acc': train_acc, 'test_acc': test_acc,
                'rank5_acc': None, 'rank10_acc': None,
                'macro_f1': macro_f1, 'weighted_f1': macro_f1,
                'macro_precision': macro_f1, 'macro_recall': macro_f1,
            })

        ensemble_summary = summarize_folds(ensemble_folds)
        print(f"    Ensemble (Voting): Rank-1={fmt_metric(ensemble_summary, 'test_acc')}, F1={fmt_metric(ensemble_summary, 'macro_f1')}")

        all_results[ds_name] = {
            "individual": individual_results,
            "ensemble": ensemble_summary,
        }

    # Print comparison table
    for ds_name, data in all_results.items():
        print(f"\n  ** {ds_name} **")
        headers = ["Method", "Rank-1 Acc (95% CI)", "Rank-5 Acc (95% CI)", "Macro F1 (95% CI)", "Gap (%)"]
        rows = []
        for clf_name, s in data["individual"].items():
            rows.append([
                clf_name,
                fmt_metric(s, "test_acc"),
                fmt_metric(s, "rank5_acc"),
                fmt_metric(s, "macro_f1"),
                f"{s['gap']*100:.1f}",
            ])
        s = data["ensemble"]
        rows.append([
            "Ensemble (Voting)",
            fmt_metric(s, "test_acc"),
            fmt_metric(s, "rank5_acc"),
            fmt_metric(s, "macro_f1"),
            f"{s['gap']*100:.1f}",
        ])
        print_md_table(headers, rows)

    return all_results


# ── Experiment 7: Cross-Dataset Generalization ───────────────────────────

def run_cross_dataset_generalization(datasets):
    """
    Train on one dataset, test on another.
    Tests cross-dataset generalization ability.
    """
    print_header("Experiment 7: Cross-Dataset Generalization")

    if len(datasets) < 2:
        print("  Skipping: need at least 2 datasets.")
        return {}

    results = {}
    ds_pairs = list(datasets.keys())

    for train_ds in ds_pairs:
        for test_ds in ds_pairs:
            if train_ds == test_ds:
                continue
            print(f"\n  Train on {train_ds}, Test on {test_ds}...")

            train_imgs, train_lbls = datasets[train_ds]
            test_imgs, test_lbls = datasets[test_ds]

            # Find common labels (if any)
            common_labels = set(np.unique(train_lbls)) & set(np.unique(test_lbls))
            if len(common_labels) == 0:
                # No common labels - test on all (will get near-zero accuracy)
                print("    No common labels between datasets. Testing with all test labels.")
                common_labels = set(np.unique(test_lbls))

            # Filter to common labels
            train_mask = np.isin(train_lbls, list(common_labels))
            test_mask = np.isin(test_lbls, list(common_labels))
            X_tr, y_tr = train_imgs[train_mask], train_lbls[train_mask]
            X_te, y_te = test_imgs[test_mask], test_lbls[test_mask]

            if len(X_tr) == 0 or len(X_te) == 0:
                print("    Not enough data after filtering. Skipping.")
                continue

            # Train
            extractor = create_extractor("combined", eigen_components=50, lbp_components=100, lbp_grid=3)
            X_tr_feat = extractor.fit_transform(X_tr)
            X_te_feat = extractor.transform(X_te)

            scaler = StandardScaler()
            X_tr_feat = scaler.fit_transform(X_tr_feat)
            X_te_feat = scaler.transform(X_te_feat)

            clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
            clf.fit(X_tr_feat, y_tr)

            train_acc = clf.score(X_tr_feat, y_tr)
            test_acc = clf.score(X_te_feat, y_te)
            results[(train_ds, test_ds)] = (train_acc, test_acc)
            print(f"    Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%, "
                  f"Common classes: {len(common_labels)}")

    # Print table
    if results:
        print(f"\n  ** Cross-Dataset Generalization Results **")
        headers = ["Train Dataset", "Test Dataset", "Train Acc (%)", "Test Acc (%)", "Common Classes"]
        rows = []
        for (tr_ds, te_ds), (tr_acc, te_acc) in results.items():
            common = len(set(np.unique(datasets[tr_ds][1])) & set(np.unique(datasets[te_ds][1])))
            rows.append([tr_ds, te_ds, f"{tr_acc*100:.2f}", f"{te_acc*100:.2f}", str(common)])
        print_md_table(headers, rows)

    return results


# ── Final Summary ─────────────────────────────────────────────────────────

def print_final_summary(exp1, exp2, exp3, exp4, exp5, exp6, exp7):
    """Print a consolidated summary of best results across all experiments."""
    print_header("FINAL SUMMARY: Best Configurations per Dataset")

    for ds_name in exp1:
        print(f"\n  === {ds_name} ===")

        # Best feature
        best_feat = max(exp1[ds_name], key=lambda x: x[1]["test_mean"])
        print(f"  Best feature:     {best_feat[0]} -> "
              f"{fmt_pct(best_feat[1]['test_mean'], ci=best_feat[1]['test_ci95'])}")

        # Best classifier
        if ds_name in exp2:
            best_clf = max(exp2[ds_name], key=lambda x: x[1]["test_mean"])
            print(f"  Best classifier:  {best_clf[0]} -> "
                  f"{fmt_pct(best_clf[1]['test_mean'], ci=best_clf[1]['test_ci95'])}")

        # Best augmentation
        if ds_name in exp3:
            best_aug = max(exp3[ds_name], key=lambda x: x[1]["test_mean"])
            print(f"  Best augmentation: {best_aug[0]} -> "
                  f"{fmt_pct(best_aug[1]['test_mean'], ci=best_aug[1]['test_ci95'])}")

        # Ensemble
        if ds_name in exp6:
            ens = exp6[ds_name]["ensemble"]
            print(f"  Ensemble (Voting): "
                  f"{fmt_pct(ens['test_mean'], ci=ens['test_ci95'])}")

        # Smallest gap
        if ds_name in exp4:
            smallest_gap = min(exp4[ds_name], key=lambda x: x[1]["gap"])
            print(f"  Smallest gap:     {smallest_gap[0]} -> "
                  f"gap={smallest_gap[1]['gap']*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 72)
    print("  ACADEMIC FACE RECOGNITION EXPERIMENTS")
    print("  Comprehensive evaluation with statistical rigor")
    print("=" * 72)

    # Load datasets
    print("\n[Step 0] Loading datasets...")
    datasets = load_all_datasets()
    if not datasets:
        print("ERROR: No datasets found. Check paths.")
        return

    # Run experiments 5, 6, and 7 only
    # print("\n[Step 1/7] Feature Extractor Comparison...")
    # exp1 = run_feature_comparison(datasets)

    # print("\n[Step 2/7] Classifier Comparison...")
    # exp2 = run_classifier_comparison(datasets)

    # print("\n[Step 3/7] Augmentation Ablation...")
    # exp3 = run_augmentation_ablation(datasets)

    # print("\n[Step 4/7] Overfitting Analysis...")
    # exp4 = run_overfitting_analysis(datasets)

    print("\n[Step 5/7] Statistical Comparison (paired t-tests)...")
    exp5 = run_statistical_comparison(datasets)

    print("\n[Step 6/7] Ensemble Methods...")
    exp6 = run_ensemble_experiment(datasets)

    print("\n[Step 7/7] Cross-Dataset Generalization...")
    exp7 = run_cross_dataset_generalization(datasets)

    # Final summary (skipped - only running experiments 5-7)
    # print_final_summary(exp1, exp2, exp3, exp4, exp5, exp6, exp7)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"  Total experiment time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
