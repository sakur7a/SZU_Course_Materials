"""
Centralized configuration matrices for all traditional face recognition experiments.
All experiment parameters are defined here to avoid scattering hardcoded configs across scripts.
"""

import math

# ─── Dataset Configs ─────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "FERET": {
        "cv_folds": 7,
        "image_size": (80, 80),
    },
    "Yale": {
        "cv_folds": 10,
        "image_size": (80, 80),
    },
    "Olivetti": {
        "cv_folds": 10,
        "image_size": (80, 80),
    },
}

# ─── Exp 1: Feature Extraction Comparison ────────────────────────────────────

FEATURE_CONFIGS = {
    # Eigenfaces with different n_components
    "eigenfaces_20": {"type": "eigenfaces", "n_components": 20},
    "eigenfaces_50": {"type": "eigenfaces", "n_components": 50},
    "eigenfaces_80": {"type": "eigenfaces", "n_components": 80},
    "eigenfaces_100": {"type": "eigenfaces", "n_components": 100},
    "eigenfaces_150": {"type": "eigenfaces", "n_components": 150},
    # LBP with different grid sizes
    "lbp_2x2": {"type": "lbp", "grid_x": 2, "grid_y": 2},
    "lbp_3x3": {"type": "lbp", "grid_x": 3, "grid_y": 3},
    "lbp_4x4": {"type": "lbp", "grid_x": 4, "grid_y": 4},
    "lbp_5x5": {"type": "lbp", "grid_x": 5, "grid_y": 5},
    # Gabor variants
    "gabor_basic": {"type": "gabor", "frequencies": (0.1, 0.2, 0.3), "orientations": (0, math.pi/4, math.pi/2, 3*math.pi/4)},
    "gabor_more_orient": {"type": "gabor", "frequencies": (0.1, 0.2, 0.3), "orientations": (0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6)},
    # HOG variants
    "hog_default": {"type": "hog", "cell_size": (8, 8), "block_size": (16, 16), "nbins": 9},
    "hog_small_cell": {"type": "hog", "cell_size": (10, 10), "block_size": (20, 20), "nbins": 9},
    # Combined (Eigenfaces + LBP)
    "combined_50_100": {"type": "combined", "eigen_components": 50, "lbp_components": 100},
    "combined_80_150": {"type": "combined", "eigen_components": 80, "lbp_components": 150},
    "combined_100_100": {"type": "combined", "eigen_components": 100, "lbp_components": 100},
}

# Shorter list for quick tests
FEATURE_CONFIGS_QUICK = {
    "eigenfaces_50": {"type": "eigenfaces", "n_components": 50},
    "lbp_3x3": {"type": "lbp", "grid_x": 3, "grid_y": 3},
    "hog_default": {"type": "hog", "cell_size": (8, 8), "block_size": (16, 16), "nbins": 9},
    "combined_50_100": {"type": "combined", "eigen_components": 50, "lbp_components": 100},
}

# ─── Exp 2: Preprocessing Methods ────────────────────────────────────────────

PREPROCESS_CONFIGS = [
    "none",
    "hist_eq",
    "clahe",
    "gaussian",
    "median",
    "bilateral",
    "nlm",
    "gradient",
    "illuminant_norm",
]

PREPROCESS_CONFIGS_QUICK = ["none", "hist_eq", "clahe"]

# Features to test with each preprocessing method
PREPROCESS_FEATURE_GRID = ["eigenfaces", "lbp", "hog", "gabor", "combined"]

# ─── Exp 3: Classifier Comparison ────────────────────────────────────────────

CLASSIFIER_CONFIGS = {
    # SVM Linear
    "svm_linear_C0.01": {"type": "svm_linear", "C": 0.01},
    "svm_linear_C0.1": {"type": "svm_linear", "C": 0.1},
    "svm_linear_C1": {"type": "svm_linear", "C": 1},
    "svm_linear_C10": {"type": "svm_linear", "C": 10},
    # SVM RBF
    "svm_rbf_C0.1_scale": {"type": "svm_rbf", "C": 0.1, "gamma": "scale"},
    "svm_rbf_C1_scale": {"type": "svm_rbf", "C": 1, "gamma": "scale"},
    "svm_rbf_C10_scale": {"type": "svm_rbf", "C": 10, "gamma": "scale"},
    "svm_rbf_C100_scale": {"type": "svm_rbf", "C": 100, "gamma": "scale"},
    "svm_rbf_C1_001": {"type": "svm_rbf", "C": 1, "gamma": 0.001},
    "svm_rbf_C1_01": {"type": "svm_rbf", "C": 1, "gamma": 0.01},
    "svm_rbf_C10_01": {"type": "svm_rbf", "C": 10, "gamma": 0.01},
    # KNN
    "knn_k1_cosine": {"type": "knn", "n_neighbors": 1, "metric": "cosine"},
    "knn_k3_cosine": {"type": "knn", "n_neighbors": 3, "metric": "cosine"},
    "knn_k5_cosine": {"type": "knn", "n_neighbors": 5, "metric": "cosine"},
    "knn_k7_cosine": {"type": "knn", "n_neighbors": 7, "metric": "cosine"},
    "knn_k1_euclid": {"type": "knn", "n_neighbors": 1, "metric": "euclidean"},
    "knn_k3_euclid": {"type": "knn", "n_neighbors": 3, "metric": "euclidean"},
    "knn_k5_euclid": {"type": "knn", "n_neighbors": 5, "metric": "euclidean"},
    # Logistic Regression
    "logistic_C0.1": {"type": "logistic", "C": 0.1},
    "logistic_C1": {"type": "logistic", "C": 1},
    "logistic_C10": {"type": "logistic", "C": 10},
    # Random Forest
    "rf_n50": {"type": "rf", "n_estimators": 50},
    "rf_n100": {"type": "rf", "n_estimators": 100},
    "rf_n200": {"type": "rf", "n_estimators": 200},
    # Gradient Boosting
    "gbdt_n50": {"type": "gbdt", "n_estimators": 50},
    "gbdt_n100": {"type": "gbdt", "n_estimators": 100},
}

CLASSIFIER_CONFIGS_QUICK = {
    "svm_linear_C1": {"type": "svm_linear", "C": 1},
    "svm_rbf_C1_scale": {"type": "svm_rbf", "C": 1, "gamma": "scale"},
    "knn_k3_cosine": {"type": "knn", "n_neighbors": 3, "metric": "cosine"},
}

# ─── Exp 4: Feature Parameter Ablation ───────────────────────────────────────

PARAM_CONFIGS = {
    "eigenfaces": {
        "n_components": [10, 20, 30, 50, 80, 100, 150, 200],
        "whiten": [True, False],
    },
    "lbp": {
        "radius": [1, 2],
        "n_points": [8, 16],
        "grid": [(2, 2), (3, 3), (4, 4), (5, 5)],
    },
    "gabor": {
        "orientations_count": [4, 6, 8],
        "frequencies_count": [2, 3, 4],
        "grid": [(2, 2), (3, 3)],
        "stats": ["mean_std", "mean_std_energy"],
    },
    "hog": {
        "cell_size": [(8, 8), (10, 10), (16, 16)],
        "block_size": [(16, 16), (24, 24), (32, 32)],
        "nbins": [9, 12],
    },
    "combined": {
        "eigen_components": [30, 50, 80, 100],
        "lbp_components": [50, 100, 150],
        "lbp_grid": [2, 3, 4],
    },
}

# ─── Exp 5: Data Augmentation ────────────────────────────────────────────────

AUGMENT_CONFIGS = {
    "none": {"angles": None, "flip": False},
    "rot5": {"angles": (-5, 5), "flip": False},
    "rot10": {"angles": (-10, 10), "flip": False},
    "rot5+10": {"angles": (-10, -5, 5, 10), "flip": False},
    "rot5+10+15": {"angles": (-15, -10, -5, 5, 10, 15), "flip": False},
    "flip_only": {"angles": None, "flip": True},
    "flip+rot5+10": {"angles": (-10, -5, 5, 10), "flip": True},
}

AUGMENT_CONFIGS_QUICK = {
    "none": {"angles": None, "flip": False},
    "rot5+10": {"angles": (-10, -5, 5, 10), "flip": False},
}

# ─── Exp 6: Robustness / Degradation ─────────────────────────────────────────

ROBUSTNESS_CONFIGS = {
    "gaussian_sigma10": {"type": "gaussian_noise", "sigma": 10},
    "gaussian_sigma25": {"type": "gaussian_noise", "sigma": 25},
    "gaussian_sigma50": {"type": "gaussian_noise", "sigma": 50},
    "salt_pepper_02": {"type": "salt_pepper", "amount": 0.02},
    "salt_pepper_05": {"type": "salt_pepper", "amount": 0.05},
    "salt_pepper_10": {"type": "salt_pepper", "amount": 0.10},
    "blur_k3": {"type": "blur", "kernel_size": 3},
    "blur_k5": {"type": "blur", "kernel_size": 5},
    "blur_k7": {"type": "blur", "kernel_size": 7},
    "occlusion_10": {"type": "occlusion", "ratio": 0.1},
    "occlusion_20": {"type": "occlusion", "ratio": 0.2},
    "occlusion_30": {"type": "occlusion", "ratio": 0.3},
    "dark_gamma03": {"type": "gamma", "gamma": 0.3},
    "dark_gamma05": {"type": "gamma", "gamma": 0.5},
    "bright_gamma15": {"type": "gamma", "gamma": 1.5},
    "bright_gamma20": {"type": "gamma", "gamma": 2.0},
    "lowres_05x": {"type": "low_res", "scale": 0.5},
    "lowres_025x": {"type": "low_res", "scale": 0.25},
    "rotation_5": {"type": "rotation", "angle": 5},
    "rotation_10": {"type": "rotation", "angle": 10},
    "rotation_15": {"type": "rotation", "angle": 15},
}

ROBUSTNESS_CONFIGS_QUICK = {
    "gaussian_sigma25": {"type": "gaussian_noise", "sigma": 25},
    "blur_k5": {"type": "blur", "kernel_size": 5},
    "occlusion_20": {"type": "occlusion", "ratio": 0.2},
}

# Representative methods to test robustness on (per dataset)
ROBUSTNESS_METHODS = ["eigenfaces_50", "lbp_3x3", "hog_default", "combined_50_100"]

# ─── Exp 7: Fusion & Ensemble ────────────────────────────────────────────────

FUSION_CONFIGS = {
    # Single features (baselines)
    "eigenfaces_only": {"features": ["eigenfaces_50"], "strategy": "single"},
    "lbp_only": {"features": ["lbp_3x3"], "strategy": "single"},
    "hog_only": {"features": ["hog_default"], "strategy": "single"},
    # Early fusion (feature concatenation)
    "eigen_lbp": {"features": ["eigenfaces_50", "lbp_3x3"], "strategy": "early"},
    "eigen_hog": {"features": ["eigenfaces_50", "hog_default"], "strategy": "early"},
    "lbp_hog": {"features": ["lbp_3x3", "hog_default"], "strategy": "early"},
    "eigen_lbp_hog": {"features": ["eigenfaces_50", "lbp_3x3", "hog_default"], "strategy": "early"},
    "gabor_hog": {"features": ["gabor_basic", "hog_default"], "strategy": "early"},
}

FUSION_ENSEMBLE_CONFIGS = {
    # Late fusion (classifier ensemble)
    "svm_rbf": {"type": "svm_rbf", "C": 1, "gamma": "scale"},
    "svm_linear": {"type": "svm_linear", "C": 1},
    "knn": {"type": "knn", "n_neighbors": 3, "metric": "cosine"},
    "logistic": {"type": "logistic", "C": 1},
    "rf": {"type": "rf", "n_estimators": 100},
    "voting_soft": {"type": "voting", "voting": "soft"},
}

# ─── Exp 8: Final Top-K Configs ──────────────────────────────────────────────

TOP_K = 5  # Number of top configurations to report per dataset

# Statistical significance
SIGNIFICANCE_ALPHA = 0.05

# ─── Common Defaults ─────────────────────────────────────────────────────────

DEFAULT_CLASSIFIER = {"type": "svm_rbf", "C": 1, "gamma": "scale"}
DEFAULT_PREPROCESS = "none"
DEFAULT_AUGMENT = {"angles": None, "flip": False}
RANDOM_SEED = 42
