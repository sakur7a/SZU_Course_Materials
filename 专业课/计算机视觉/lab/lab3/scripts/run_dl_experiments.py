"""Deep learning face recognition experiments from PLAN.md.

Usage:
    conda run -n cv-deep python run_dl_experiments.py [--dataset FERET|Yale|Olivetti] [--experiment all|kshot|robustness|cross]

Requires: conda env 'cv-deep' with torch, numpy, scikit-learn, opencv-python, pillow
"""

import argparse
import os
import sys
import time

# Add project root to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _PROJECT_ROOT)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import torch

from deep_learning import (
    SmallCNN, SmallCNNEmbedding,
    train_classifier, evaluate_classifier,
    extract_embeddings, predict_embedding,
    k_shot_split, get_dataloaders,
)

# ── Dataset loading ──────────────────────────────────────────────────────────

IMG_SIZE = (80, 80)

def load_feret():
    data_dir = os.path.join(_DATA_DIR, '数据库-feret_k175_s7_w80_h80', 'feret_k175_s7_w80_h80')
    if not os.path.exists(data_dir):
        return None, None
    from PIL import Image as PILImage
    images, labels = [], []
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.bmp'):
            continue
        parts = f[:-4].split('_')
        if len(parts) != 2:
            continue
        img = np.array(PILImage.open(os.path.join(data_dir, f)).convert('L'))
        if img.shape != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(int(parts[0]))
    return np.array(images), np.array(labels)


def load_yale():
    img_path = os.path.join(_DATA_DIR, 'yale_images.npy')
    lbl_path = os.path.join(_DATA_DIR, 'yale_labels.npy')
    if not os.path.exists(img_path):
        return None, None
    images, labels = np.load(img_path), np.load(lbl_path)
    if images.shape[1:] != IMG_SIZE:
        images = np.array([cv2.resize(img, IMG_SIZE) for img in images])
    return images, labels


def load_olivetti():
    img_path = os.path.join(_DATA_DIR, 'olivetti_images.npy')
    lbl_path = os.path.join(_DATA_DIR, 'olivetti_labels.npy')
    if not os.path.exists(img_path):
        return None, None
    images, labels = np.load(img_path), np.load(lbl_path)
    if images.shape[1:] != IMG_SIZE:
        images = np.array([cv2.resize(img, IMG_SIZE) for img in images])
    return images, labels


LOADERS = {'FERET': load_feret, 'Yale': load_yale, 'Olivetti': load_olivetti}


# ── Traditional Feature Extractors ───────────────────────────────────────────

class HOGExtractor:
    """HOG features using cv2.HOGDescriptor."""
    def __init__(self, win_size=(80, 80), block_size=(16, 16), block_stride=(8, 8),
                 cell_size=(8, 8), nbins=9):
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def transform(self, X):
        return np.array([self.hog.compute(img).flatten() for img in X])


def extract_traditional_features(X_train, X_test, method='pca'):
    """Extract traditional features and return (train_feat, test_feat)."""
    if method == 'pca':
        flat_tr = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
        flat_te = X_test.reshape(X_test.shape[0], -1).astype(np.float64)
        n_comp = min(50, flat_tr.shape[0] - 1)
        pca = PCA(n_components=n_comp, whiten=True)
        scaler = StandardScaler()
        feat_tr = scaler.fit_transform(pca.fit_transform(flat_tr))
        feat_te = scaler.transform(pca.transform(flat_te))
        return feat_tr, feat_te
    elif method == 'lbp':
        from face_recognition_system import LBPExtractor
        lbp = LBPExtractor(grid_x=3, grid_y=3)
        raw_tr = lbp.transform(X_train)
        raw_te = lbp.transform(X_test)
        scaler = StandardScaler()
        scaled_tr = scaler.fit_transform(raw_tr)
        scaled_te = scaler.transform(raw_te)
        n_comp = min(50, scaled_tr.shape[0] - 1, scaled_tr.shape[1])
        pca = PCA(n_components=n_comp)
        return pca.fit_transform(scaled_tr), pca.transform(scaled_te)
    elif method == 'hog':
        hog = HOGExtractor()
        raw_tr = hog.transform(X_train)
        raw_te = hog.transform(X_test)
        scaler = StandardScaler()
        scaled_tr = scaler.fit_transform(raw_tr)
        scaled_te = scaler.transform(raw_te)
        n_comp = min(100, scaled_tr.shape[0] - 1, scaled_tr.shape[1])
        pca = PCA(n_components=n_comp)
        return pca.fit_transform(scaled_tr), pca.transform(scaled_te)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_svm_classifier(feat_tr, y_tr, feat_te, y_te):
    """Train SVM and return accuracy."""
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    svm.fit(feat_tr, y_tr)
    return accuracy_score(y_te, svm.predict(feat_te)), svm


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def resize_images(images, size=IMG_SIZE):
    if images.shape[1:] == size:
        return images
    return np.array([cv2.resize(img, size) for img in images])


def add_gaussian_noise(images, sigma=25):
    noisy = []
    for img in images:
        noise = np.random.normal(0, sigma, img.shape)
        noisy.append(np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8))
    return np.array(noisy)


def add_blur(images, ksize=5):
    return np.array([cv2.GaussianBlur(img, (ksize, ksize), 0) for img in images])


def simulate_occlusion(images, ratio=0.2):
    occ = []
    h, w = images.shape[1], images.shape[2]
    occ_h, occ_w = int(h * np.sqrt(ratio)), int(w * np.sqrt(ratio))
    for img in images:
        y = np.random.randint(0, h - occ_h)
        x = np.random.randint(0, w - occ_w)
        o = img.copy()
        o[y:y+occ_h, x:x+occ_w] = 0
        occ.append(o)
    return np.array(occ)


def change_brightness(images, gamma=0.5):
    result = []
    for img in images:
        f = img.astype(np.float64) / 255.0
        result.append((np.power(f, gamma) * 255).astype(np.uint8))
    return np.array(result)


def downscale(images, scale=0.5):
    h, w = images.shape[1], images.shape[2]
    nh, nw = int(h * scale), int(w * scale)
    result = []
    for img in images:
        small = cv2.resize(img, (nw, nh))
        result.append(cv2.resize(small, (w, h)))
    return np.array(result)


# ── Experiment 0: Closed-set 1:N full dataset evaluation ─────────────────────

def run_closedset_experiment(images, labels, dataset_name, device='cpu', epochs=50):
    """Full dataset closed-set 1:N evaluation with all methods."""
    print_header(f"Closed-set 1:N ({dataset_name})")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # 70/30 split
    X_tr, X_te, y_tr, y_te = train_test_split(images, y, test_size=0.3,
                                                random_state=42, stratify=y)

    results = []

    # Traditional methods
    for method in ['pca', 'lbp', 'hog']:
        feat_tr, feat_te = extract_traditional_features(X_tr, X_te, method)
        acc, svm = train_svm_classifier(feat_tr, y_tr, feat_te, y_te)
        y_pred = svm.predict(feat_te)
        f1_macro = f1_score(y_te, y_pred, average='macro')
        f1_weighted = f1_score(y_te, y_pred, average='weighted')
        results.append((method.upper() + '+SVM', acc, f1_macro, f1_weighted))

    # SmallCNN
    model = SmallCNN(n_classes)
    train_classifier(model, X_tr, y_tr, X_te, y_te, epochs=epochs, device=device)
    test_loader = get_dataloaders(X_tr, y_tr, X_te, y_te)[1]
    acc = evaluate_classifier(model, test_loader, device)
    y_pred = []
    model.eval()
    for X_batch, _ in test_loader:
        y_pred.extend(model(X_batch.to(device)).argmax(1).cpu().numpy())
    f1_macro = f1_score(y_te, y_pred, average='macro')
    f1_weighted = f1_score(y_te, y_pred, average='weighted')
    results.append(('SmallCNN', acc, f1_macro, f1_weighted))

    # SmallCNN + Augmentation
    from face_recognition_system import augment_dataset
    X_tr_aug, y_tr_aug = augment_dataset(X_tr, y_tr)
    model_aug = SmallCNN(n_classes)
    train_classifier(model_aug, X_tr_aug, y_tr_aug, X_te, y_te, epochs=epochs, device=device)
    acc_aug = evaluate_classifier(model_aug, test_loader, device)
    y_pred_aug = []
    model_aug.eval()
    for X_batch, _ in test_loader:
        y_pred_aug.extend(model_aug(X_batch.to(device)).argmax(1).cpu().numpy())
    f1_macro_aug = f1_score(y_te, y_pred_aug, average='macro')
    f1_weighted_aug = f1_score(y_te, y_pred_aug, average='weighted')
    results.append(('CNN+Aug', acc_aug, f1_macro_aug, f1_weighted_aug))

    # Print results
    print(f"\n  {'Method':<12} | {'Rank-1':>7} | {'MacroF1':>8} | {'WeightF1':>8}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
    for name, acc, f1m, f1w in results:
        print(f"  {name:<12} | {acc:7.3f} | {f1m:8.3f} | {f1w:8.3f}")

    return results


# ── Experiment 1: k-shot learning curve ──────────────────────────────────────

def run_kshot_experiment(images, labels, dataset_name, device='cpu', epochs=50, quick=False):
    """Compare traditional methods vs CNN at different training sample sizes."""
    print_header(f"k-shot Learning Curve ({dataset_name})")

    n_people = len(np.unique(labels))
    max_k = min(50, min(np.bincount(labels)) - 1)

    # k values to test
    if quick:
        k_values = [1, 3, 5]
        if max_k >= 10:
            k_values.append(10)
    else:
        k_values = [1, 2, 3, 5]
        if max_k >= 7:
            k_values.append(7)
        if max_k >= 10:
            k_values.append(10)
        if max_k >= 20:
            k_values.append(20)
        if max_k >= 30:
            k_values.append(30)
        if max_k >= 50:
            k_values.append(50)

    print(f"  Dataset: {dataset_name}, {len(images)} images, {n_people} people")
    print(f"  k values: {k_values}")

    # Traditional baseline (PCA + SVM)
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    results = {k: {} for k in k_values}

    for k in k_values:
        X_tr, y_tr, X_te, y_te = k_shot_split(images, labels, k)
        # Encode labels to 0-based for CNN
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_te_enc = le.transform(y_te)
        n_classes = len(le.classes_)

        # --- Traditional: PCA + SVM ---
        flat_tr = X_tr.reshape(X_tr.shape[0], -1).astype(np.float64)
        flat_te = X_te.reshape(X_te.shape[0], -1).astype(np.float64)
        n_comp = min(50, flat_tr.shape[0] - 1)
        pca = PCA(n_components=n_comp, whiten=True)
        scaler = StandardScaler()
        feat_tr = scaler.fit_transform(pca.fit_transform(flat_tr))
        feat_te = scaler.transform(pca.transform(flat_te))
        svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
        svm.fit(feat_tr, y_tr)
        pca_acc = accuracy_score(y_te, svm.predict(feat_te))
        results[k]['PCA+SVM'] = pca_acc

        # --- Traditional: LBP + SVM (with PCA to handle high-dimensional features) ---
        from face_recognition_system import LBPExtractor
        from sklearn.decomposition import PCA as PCA_LBP
        lbp = LBPExtractor(grid_x=3, grid_y=3)
        lbp_raw_tr = lbp.transform(X_tr)
        lbp_raw_te = lbp.transform(X_te)
        lbp_scaler = StandardScaler()
        lbp_scaled_tr = lbp_scaler.fit_transform(lbp_raw_tr)
        lbp_scaled_te = lbp_scaler.transform(lbp_raw_te)
        n_lbp_comp = min(50, lbp_scaled_tr.shape[0] - 1, lbp_scaled_tr.shape[1])
        lbp_pca = PCA_LBP(n_components=n_lbp_comp)
        lbp_feat_tr = lbp_pca.fit_transform(lbp_scaled_tr)
        lbp_feat_te = lbp_pca.transform(lbp_scaled_te)
        svm_lbp = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
        svm_lbp.fit(lbp_feat_tr, y_tr)
        lbp_acc = accuracy_score(y_te, svm_lbp.predict(lbp_feat_te))
        results[k]['LBP+SVM'] = lbp_acc

        # --- Small CNN ---
        model = SmallCNN(n_classes)
        _, test_accs = train_classifier(model, X_tr, y_tr_enc, X_te, y_te_enc,
                                        epochs=epochs, lr=1e-3, batch_size=64, device=device)
        results[k]['SmallCNN'] = max(test_accs)

        # --- Small CNN + Augmentation ---
        from face_recognition_system import augment_dataset
        X_tr_aug, y_tr_aug_enc = augment_dataset(X_tr, y_tr_enc)
        model_aug = SmallCNN(n_classes)
        _, test_accs_aug = train_classifier(model_aug, X_tr_aug, y_tr_aug_enc, X_te, y_te_enc,
                                            epochs=epochs, lr=1e-3, batch_size=64, device=device)
        results[k]['CNN+Aug'] = max(test_accs_aug)

        print(f"  k={k:2d}: PCA+SVM={pca_acc:.3f}  LBP+SVM={lbp_acc:.3f}  "
              f"CNN={results[k]['SmallCNN']:.3f}  CNN+Aug={results[k]['CNN+Aug']:.3f}")

    # Print summary table
    print(f"\n  {'k':>4} | {'PCA+SVM':>8} | {'LBP+SVM':>8} | {'SmallCNN':>8} | {'CNN+Aug':>8}")
    print(f"  {'----':>4}-+-{'--------':>8}-+-{'--------':>8}-+-{'--------':>8}-+-{'--------':>8}")
    for k in k_values:
        r = results[k]
        print(f"  {k:4d} | {r['PCA+SVM']:8.3f} | {r['LBP+SVM']:8.3f} | "
              f"{r['SmallCNN']:8.3f} | {r['CNN+Aug']:8.3f}")

    return results


# ── Experiment 2: Robustness ─────────────────────────────────────────────────

def run_robustness_experiment(images, labels, dataset_name, device='cpu', epochs=50):
    """Test robustness to noise, blur, occlusion, brightness, low resolution."""
    print_header(f"Robustness Experiment ({dataset_name})")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(images, y, test_size=0.3,
                                                random_state=42, stratify=y)
    n_classes = len(np.unique(y))

    # Train traditional SVM classifiers
    models = {}
    clean_accs = {}

    for method in ['pca', 'lbp', 'hog']:
        print(f"  Training {method.upper()}+SVM...")
        feat_tr, feat_te = extract_traditional_features(X_tr, X_te, method)
        acc, svm = train_svm_classifier(feat_tr, y_tr, feat_te, y_te)
        models[method] = (svm, feat_te)
        clean_accs[method] = acc

    # Train CNN models
    print("  Training SmallCNN...")
    model_cnn = SmallCNN(n_classes)
    train_classifier(model_cnn, X_tr, y_tr, X_te, y_te, epochs=epochs, device=device)

    print("  Training SmallCNNEmbedding...")
    model_emb = SmallCNNEmbedding(n_classes)
    train_classifier(model_emb, X_tr, y_tr, X_te, y_te, epochs=epochs, device=device)

    # Clean baseline
    test_loader = get_dataloaders(X_tr, y_tr, X_te, y_te)[1]
    clean_accs['cnn'] = evaluate_classifier(model_cnn, test_loader, device)
    clean_accs['cnn_emb'] = evaluate_classifier(model_emb, test_loader, device)

    print(f"\n  Clean: " + "  ".join(f"{k.upper()}={v:.3f}" for k, v in clean_accs.items()))

    # Degradation tests
    degradations = [
        ("Gaussian σ=15", lambda imgs: add_gaussian_noise(imgs, 15)),
        ("Gaussian σ=25", lambda imgs: add_gaussian_noise(imgs, 25)),
        ("Blur k=3", lambda imgs: add_blur(imgs, 3)),
        ("Blur k=5", lambda imgs: add_blur(imgs, 5)),
        ("Occlusion 10%", lambda imgs: simulate_occlusion(imgs, 0.1)),
        ("Occlusion 20%", lambda imgs: simulate_occlusion(imgs, 0.2)),
        ("Dark γ=0.3", lambda imgs: change_brightness(imgs, 0.3)),
        ("Dark γ=0.5", lambda imgs: change_brightness(imgs, 0.5)),
        ("Low-res 0.5x", lambda imgs: downscale(imgs, 0.5)),
        ("Low-res 0.25x", lambda imgs: downscale(imgs, 0.25)),
    ]

    header = f"  {'Degradation':<16} | {'PCA':>5} | {'LBP':>5} | {'HOG':>5} | {'CNN':>5} | {'CEmb':>5}"
    print(f"\n{header}")
    print(f"  {'-'*16}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")

    for name, degrade_fn in degradations:
        X_te_deg = degrade_fn(X_te)
        accs = {}

        # Traditional methods
        for method in ['pca', 'lbp', 'hog']:
            svm_model, _ = models[method]
            _, feat_deg = extract_traditional_features(X_tr, X_te_deg, method)
            accs[method] = accuracy_score(y_te, svm_model.predict(feat_deg))

        # CNN methods
        deg_loader = get_dataloaders(X_tr, y_tr, X_te_deg, y_te)[1]
        accs['cnn'] = evaluate_classifier(model_cnn, deg_loader, device)
        accs['cnn_emb'] = evaluate_classifier(model_emb, deg_loader, device)

        print(f"  {name:<16} | {accs['pca']:5.3f} | {accs['lbp']:5.3f} | {accs['hog']:5.3f} | "
              f"{accs['cnn']:5.3f} | {accs['cnn_emb']:5.3f}")

    # Print deltas
    print(f"\n  Performance drop from clean:")
    print(f"  {'Degradation':<16} | {'PCA Δ':>6} | {'LBP Δ':>6} | {'HOG Δ':>6} | {'CNN Δ':>6} | {'CEmb Δ':>6}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for name, degrade_fn in degradations:
        X_te_deg = degrade_fn(X_te)
        deltas = {}
        for method in ['pca', 'lbp', 'hog']:
            svm_model, _ = models[method]
            _, feat_deg = extract_traditional_features(X_tr, X_te_deg, method)
            deltas[method] = accuracy_score(y_te, svm_model.predict(feat_deg)) - clean_accs[method]
        deg_loader = get_dataloaders(X_tr, y_tr, X_te_deg, y_te)[1]
        deltas['cnn'] = evaluate_classifier(model_cnn, deg_loader, device) - clean_accs['cnn']
        deltas['cnn_emb'] = evaluate_classifier(model_emb, deg_loader, device) - clean_accs['cnn_emb']

        print(f"  {name:<16} | {deltas['pca']:+6.3f} | {deltas['lbp']:+6.3f} | {deltas['hog']:+6.3f} | "
              f"{deltas['cnn']:+6.3f} | {deltas['cnn_emb']:+6.3f}")


# ── Experiment 3: Pretrained model upper bound (InsightFace/ArcFace) ─────────

def run_pretrained_experiment(images, labels, dataset_name):
    """Exp 5: InsightFace (ResNet50@WebFace600K) pretrained model upper bound."""
    print_header(f"Pretrained Model Upper Bound ({dataset_name})")

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("  SKIP: insightface not installed. Install with: pip install insightface onnxruntime-gpu")
        return None

    # Convert grayscale to BGR for InsightFace
    import cv2 as cv2_mod
    bgr_images = np.array([cv2_mod.cvtColor(img, cv2_mod.COLOR_GRAY2BGR) for img in images])

    print("  Loading InsightFace model (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rec_model = app.models.get('recognition')
    if rec_model is None:
        print("  ERROR: recognition model not found")
        return None

    # Extract embeddings
    print(f"  Extracting embeddings ({len(bgr_images)} images)...")
    embeddings, valid_idx = [], []
    for i, img in enumerate(bgr_images):
        img_resized = cv2_mod.resize(img, (112, 112))
        try:
            emb = rec_model.get_feat(img_resized)
            if emb is not None:
                embeddings.append(emb.flatten())
                valid_idx.append(i)
        except Exception:
            pass

    if len(embeddings) == 0:
        print("  ERROR: No embeddings extracted")
        return None

    embeddings = np.array(embeddings)
    # L2 normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)
    valid_labels = labels[valid_idx]
    print(f"  Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # 70/30 split
    le = LabelEncoder()
    y = le.fit_transform(valid_labels)
    X_tr, X_te, y_tr, y_te = train_test_split(embeddings, y, test_size=0.3,
                                                random_state=42, stratify=y)

    results = []

    # SVM (linear)
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_tr, y_tr)
    svm_acc = accuracy_score(y_te, svm.predict(X_te))
    y_pred_svm = svm.predict(X_te)
    f1_macro_svm = f1_score(y_te, y_pred_svm, average='macro')
    f1_weighted_svm = f1_score(y_te, y_pred_svm, average='weighted')
    results.append(('SVM(linear)', svm_acc, f1_macro_svm, f1_weighted_svm))

    # KNN cosine
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(X_tr, y_tr)
    knn_acc = accuracy_score(y_te, knn.predict(X_te))
    y_pred_knn = knn.predict(X_te)
    f1_macro_knn = f1_score(y_te, y_pred_knn, average='macro')
    f1_weighted_knn = f1_score(y_te, y_pred_knn, average='weighted')
    results.append(('KNN(cosine)', knn_acc, f1_macro_knn, f1_weighted_knn))

    print(f"\n  {'Method':<12} | {'Rank-1':>7} | {'MacroF1':>8} | {'WeightF1':>8}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
    for name, acc, f1m, f1w in results:
        print(f"  {name:<12} | {acc:7.3f} | {f1m:8.3f} | {f1w:8.3f}")

    return results


# ── Experiment 3: Cross-dataset embedding transfer ───────────────────────────

def run_cross_dataset_experiment(device='cpu', epochs=50):
    """Train embedding model on dataset A, test on dataset B via nearest-neighbor."""
    print_header("Cross-Dataset Embedding Transfer")

    datasets = {}
    for name, loader in LOADERS.items():
        imgs, lbls = loader()
        if imgs is not None:
            datasets[name] = (imgs, lbls)

    if len(datasets) < 2:
        print("  Need at least 2 datasets. Skipping.")
        return

    print(f"  Available datasets: {list(datasets.keys())}")

    for train_name, (train_imgs, train_lbls) in datasets.items():
        for test_name, (test_imgs, test_lbls) in datasets.items():
            if train_name == test_name:
                continue

            print(f"\n  Train: {train_name} -> Test: {test_name}")

            le = LabelEncoder()
            y_tr_enc = le.fit_transform(train_lbls)
            n_classes = len(le.classes_)
            # Use 80/20 split from training data for validation during training
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                train_imgs, y_tr_enc, test_size=0.2, random_state=42, stratify=y_tr_enc)
            model = SmallCNNEmbedding(n_classes)
            train_classifier(model, X_tr, y_tr, X_val, y_val,
                             epochs=epochs, device=device)

            # Build gallery from test set (one per identity)
            gallery_emb, gallery_labels = [], []
            for label in np.unique(test_lbls):
                idxs = np.where(test_lbls == label)[0]
                # Use first image as gallery
                emb = extract_embeddings(model, test_imgs[idxs[:1]], device=device)
                gallery_emb.append(emb[0])
                gallery_labels.append(label)
            gallery_emb = np.array(gallery_emb)

            # Query with remaining images
            correct, total = 0, 0
            for label in np.unique(test_lbls):
                idxs = np.where(test_lbls == label)[0]
                query_imgs = test_imgs[idxs[1:]]  # skip gallery image
                if len(query_imgs) == 0:
                    continue
                for img in query_imgs:
                    pred, conf = predict_embedding(model, img, gallery_emb,
                                                   np.array(gallery_labels), device)
                    if pred == label:
                        correct += 1
                    total += 1

            acc = correct / total if total > 0 else 0
            print(f"    Rank-1 Accuracy: {acc:.3f} ({correct}/{total})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep learning face recognition experiments")
    parser.add_argument('--dataset', choices=['FERET', 'Yale', 'Olivetti', 'all'], default='all')
    parser.add_argument('--experiment', choices=['closedset', 'kshot', 'robustness', 'pretrained', 'cross', 'all'], default='all')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per model')
    parser.add_argument('--quick', action='store_true', help='Quick mode: fewer k values, fewer epochs')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20

    device = args.device
    print(f"Device: {device}")

    # Load datasets
    if args.dataset == 'all':
        dataset_names = list(LOADERS.keys())
    else:
        dataset_names = [args.dataset]

    datasets = {}
    for name in dataset_names:
        imgs, lbls = LOADERS[name]()
        if imgs is not None:
            datasets[name] = (imgs, lbls)
            print(f"Loaded {name}: {len(imgs)} images, {len(np.unique(lbls))} people")

    if not datasets:
        print("ERROR: No datasets found.")
        return

    # Run experiments
    if args.experiment in ('closedset', 'all'):
        for name, (imgs, lbls) in datasets.items():
            run_closedset_experiment(imgs, lbls, name, device, epochs=args.epochs)

    if args.experiment in ('kshot', 'all'):
        for name, (imgs, lbls) in datasets.items():
            run_kshot_experiment(imgs, lbls, name, device, epochs=args.epochs, quick=args.quick)

    if args.experiment in ('robustness', 'all'):
        # Use largest dataset for robustness
        best = max(datasets.items(), key=lambda x: len(x[1][0]))
        run_robustness_experiment(best[1][0], best[1][1], best[0], device, epochs=args.epochs)

    if args.experiment in ('pretrained', 'all'):
        for name, (imgs, lbls) in datasets.items():
            run_pretrained_experiment(imgs, lbls, name)

    if args.experiment in ('cross', 'all'):
        run_cross_dataset_experiment(device, epochs=args.epochs)


if __name__ == '__main__':
    main()
