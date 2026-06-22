"""
Run final candidate verification - lightweight version without progress bars.
Each dataset runs independently. Use --dataset to select.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from face_recognition_system import (
    EigenfacesExtractor, LBPExtractor, GaborExtractor, HOGExtractor, CombinedExtractor,
    augment_dataset, load_feret_dataset,
)
from scripts.results_io import save_result, save_summary, clear_results

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
IMG_SIZE = (80, 80)
RANDOM_SEED = 42


def load_ds(name):
    import cv2
    if name == "FERET":
        d = os.path.join(DATA_DIR, "数据库-feret_k175_s7_w80_h80", "feret_k175_s7_w80_h80")
        return load_feret_dataset(d, img_size=IMG_SIZE)
    elif name == "Yale":
        imgs, lbls = np.load(os.path.join(DATA_DIR, "yale_images.npy")), np.load(os.path.join(DATA_DIR, "yale_labels.npy"))
        if imgs.shape[1:] != IMG_SIZE:
            imgs = np.array([cv2.resize(im, IMG_SIZE) for im in imgs])
        return imgs, lbls
    elif name == "Olivetti":
        imgs, lbls = np.load(os.path.join(DATA_DIR, "olivetti_images.npy")), np.load(os.path.join(DATA_DIR, "olivetti_labels.npy"))
        if imgs.shape[1:] != IMG_SIZE:
            imgs = np.array([cv2.resize(im, IMG_SIZE) for im in imgs])
        return imgs, lbls


def make_ext(name, params):
    if name == "eigenfaces": return EigenfacesExtractor(n_components=params.get("n_components", 50))
    if name == "lbp": return LBPExtractor(grid_x=params.get("grid_x", 3), grid_y=params.get("grid_y", 3))
    if name == "hog": return HOGExtractor()
    if name == "combined": return CombinedExtractor(eigen_components=params.get("eigen_components", 50), lbp_components=params.get("lbp_components", 100))
    if name == "gabor": return GaborExtractor()


def make_clf(cfg):
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    t = cfg["type"]
    if t == "svm_rbf": return SVC(kernel="rbf", C=cfg.get("C",1), gamma=cfg.get("gamma","scale"), class_weight="balanced", probability=True)
    if t == "svm_linear": return SVC(kernel="linear", C=cfg.get("C",1), class_weight="balanced", probability=True)
    if t == "knn": return KNeighborsClassifier(n_neighbors=cfg.get("n_neighbors",3), weights="distance", metric=cfg.get("metric","cosine"))
    if t == "logistic": return LogisticRegression(C=cfg.get("C",1), max_iter=2000, solver="saga", random_state=RANDOM_SEED)
    if t == "rf": return RandomForestClassifier(n_estimators=cfg.get("n_estimators",100), random_state=RANDOM_SEED, n_jobs=-1)


def run_candidate(images, labels, n_folds, feat_name, feat_params, clf_cfg, aug_cfg=None):
    unique, counts = np.unique(labels, return_counts=True)
    actual_folds = min(n_folds, min(counts))
    if actual_folds < 2: actual_folds = 2
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_accs = []
    for tr_idx, te_idx in cv.split(images, labels):
        X_tr, y_tr = images[tr_idx], labels[tr_idx]
        X_te, y_te = images[te_idx], labels[te_idx]
        if aug_cfg:
            angles = aug_cfg.get("angles")
            flip = aug_cfg.get("flip", False)
            if angles or flip:
                X_tr, y_tr = augment_dataset(X_tr, y_tr, flips=flip, rotations=angles is not None, angles=angles or (-10,-5,5,10))
        ext = make_ext(feat_name, feat_params)
        X_tr_f = ext.fit_transform(X_tr)
        X_te_f = ext.transform(X_te)
        scaler = StandardScaler()
        X_tr_f = scaler.fit_transform(X_tr_f)
        X_te_f = scaler.transform(X_te_f)
        clf = make_clf(clf_cfg)
        clf.fit(X_tr_f, y_tr)
        y_pred = clf.predict(X_te_f)
        fold_accs.append(accuracy_score(y_te, y_pred))
    m, s = np.mean(fold_accs), np.std(fold_accs, ddof=1) if len(fold_accs) > 1 else 0
    t_val = stats.t.ppf(0.975, df=len(fold_accs)-1) if len(fold_accs) > 1 else 1.96
    ci = t_val * s / np.sqrt(len(fold_accs)) if len(fold_accs) > 1 else 0
    return m, ci, fold_accs


CANDIDATES = {
    "FERET": [
        ("combined_rbf_c1", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_rbf","C":1,"gamma":"scale"}, None),
        ("combined_rbf_c10", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_rbf","C":10,"gamma":"scale"}, None),
        ("combined_linear_c1", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_linear","C":1}, None),
        ("combined_linear_c10", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_linear","C":10}, None),
        ("combined_rbf_c10_aug", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_rbf","C":10,"gamma":"scale"}, {"angles":(-10,-5,5,10),"flip":False}),
        ("eigenfaces100_linear", "eigenfaces", {"n_components":100}, {"type":"svm_linear","C":1}, None),
        ("lbp3x3_linear", "lbp", {"grid_x":3,"grid_y":3}, {"type":"svm_linear","C":1}, None),
    ],
    "Yale": [
        ("hog_rbf_c1", "hog", {}, {"type":"svm_rbf","C":1,"gamma":"scale"}, None),
        ("hog_linear_c1", "hog", {}, {"type":"svm_linear","C":1}, None),
        ("hog_logistic", "hog", {}, {"type":"logistic","C":1}, None),
        ("eigenfaces150_linear", "eigenfaces", {"n_components":150}, {"type":"svm_linear","C":1}, None),
        ("combined_rbf_c1", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_rbf","C":1,"gamma":"scale"}, None),
    ],
    "Olivetti": [
        ("hog_rbf_c1", "hog", {}, {"type":"svm_rbf","C":1,"gamma":"scale"}, None),
        ("hog_linear_c1", "hog", {}, {"type":"svm_linear","C":1}, None),
        ("hog_rbf_aug", "hog", {}, {"type":"svm_rbf","C":1,"gamma":"scale"}, {"angles":(-10,-5,5,10),"flip":False}),
        ("hog_rbf_flip_aug", "hog", {}, {"type":"svm_rbf","C":1,"gamma":"scale"}, {"angles":(-10,-5,5,10),"flip":True}),
        ("eigenfaces100_rbf", "eigenfaces", {"n_components":100}, {"type":"svm_rbf","C":1,"gamma":"scale"}, None),
        ("combined_rbf_c1", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_rbf","C":1,"gamma":"scale"}, None),
        ("combined_rbf_flip_aug", "combined", {"eigen_components":50,"lbp_components":100}, {"type":"svm_rbf","C":1,"gamma":"scale"}, {"angles":(-10,-5,5,10),"flip":True}),
    ],
}


def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "Olivetti"
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Loading {ds_name}...")
    images, labels = load_ds(ds_name)
    print(f"  {len(images)} images, {len(np.unique(labels))} classes")

    candidates = CANDIDATES.get(ds_name, [])
    if not candidates:
        print(f"No candidates for {ds_name}")
        return

    clear_results("final_candidates", ds_name)
    results = []

    for name, feat, fparams, clf_cfg, aug in candidates:
        t0 = time.perf_counter()
        m, ci, fold_accs = run_candidate(images, labels, n_folds, feat, fparams, clf_cfg, aug)
        elapsed = time.perf_counter() - t0
        print(f"  {name:30s} acc={m*100:.2f}% +/- {ci*100:.2f}  ({elapsed:.1f}s)")
        results.append({"name": name, "test_acc": m, "ci95": ci, "fold_test_accs": fold_accs})

        for fi, acc in enumerate(fold_accs):
            save_result({
                "feature": feat, "feature_params": fparams,
                "preprocess": "none", "classifier": clf_cfg["type"],
                "classifier_params": clf_cfg, "augmentation": str(aug),
                "fold": fi, "test_acc": acc, "train_acc": None,
                "rank5_acc": None, "macro_f1": None, "precision": None,
                "recall": None, "fps": None, "train_time": elapsed,
                "feature_dim": None,
            }, "final_candidates", ds_name)

    # Sort and rank
    results.sort(key=lambda r: r["test_acc"], reverse=True)
    print(f"\n  Ranking on {ds_name}:")
    for i, r in enumerate(results, 1):
        print(f"    #{i}: {r['name']} ({r['test_acc']*100:.2f}%)")

    # Pairwise t-tests
    sig = []
    print(f"\n  Pairwise t-tests:")
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            a, b = np.array(results[i]["fold_test_accs"]), np.array(results[j]["fold_test_accs"])
            if len(a) == len(b) and len(a) > 1:
                t_stat, p_val = stats.ttest_rel(a, b)
                diff = results[i]["test_acc"] - results[j]["test_acc"]
                s = "*" if p_val < 0.05 else ""
                sig.append({"a": results[i]["name"], "b": results[j]["name"], "diff": diff, "p": p_val, "sig": p_val < 0.05})
                print(f"    {results[i]['name']:25s} vs {results[j]['name']:25s} diff={diff*100:+.2f}% p={p_val:.4f} {s}")

    save_summary({
        "ranking": [{"rank": i+1, "name": r["name"], "accuracy": r["test_acc"], "ci95": r["ci95"]} for i, r in enumerate(results)],
        "pairwise_tests": sig,
        "fold_accuracies": {r["name"]: [float(v) for v in r["fold_test_accs"]] for r in results},
    }, "final_candidates", ds_name)
    print(f"\n  Results saved to results/summary_final_candidates_{ds_name}.json")


if __name__ == "__main__":
    main()
