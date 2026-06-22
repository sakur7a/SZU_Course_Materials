"""
人脸识别系统 - Face Recognition System
基于特征脸(Eigenfaces)和LBP特征的人脸识别，支持噪声/光照鲁棒性增强
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def augment_dataset(images, labels, flips=False, rotations=True, angles=(-10, -5, 5, 10)):
    """数据增强: 小角度旋转 (默认不翻转, 避免人脸不对称导致身份混淆)"""
    aug_images, aug_labels = [images], [labels]
    if flips:
        flipped = np.array([np.fliplr(img) for img in images])
        aug_images.append(flipped)
        aug_labels.append(labels)
    if rotations:
        for angle in angles:
            rotated = []
            for img in images:
                pil_img = PILImage.fromarray(img)
                rot = pil_img.rotate(angle, fillcolor=0)
                rotated.append(np.array(rot))
            aug_images.append(np.array(rotated))
            aug_labels.append(labels)
    return np.concatenate(aug_images), np.concatenate(aug_labels)


def _imread_gray(path):
    """读取灰度图 - 兼容中文路径"""
    try:
        pil_img = PILImage.open(str(path)).convert('L')
        return np.array(pil_img)
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


# ============================================================
# 数据加载
# ============================================================

def load_feret_dataset(data_dir, img_size=(80, 80)):
    """加载FERET数据集: 175人, 每人7张, 80x80灰度"""
    images, labels = [], []
    data_path = Path(data_dir)
    for img_file in sorted(data_path.glob("*.bmp")):
        parts = img_file.stem.split("_")
        if len(parts) != 2:
            continue
        label = int(parts[0])
        img = _imread_gray(img_file)
        if img is None:
            continue
        if img.shape != img_size:
            img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def load_dataset(data_dir, img_size=(80, 80)):
    """通用数据集加载"""
    images, labels = [], []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    for person_dir in sorted(data_path.iterdir()):
        if not person_dir.is_dir():
            continue
        try:
            label = int(person_dir.name)
        except ValueError:
            continue
        for img_file in sorted(person_dir.glob("*")):
            if img_file.suffix.lower() not in ('.bmp', '.jpg', '.jpeg', '.png', '.pgm'):
                continue
            img = _imread_gray(img_file)
            if img is None:
                continue
            if img.shape != img_size:
                img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


# ============================================================
# 图像预处理与增强
# ============================================================

def preprocess_histogram_eq(images):
    """直方图均衡化 - 增强光照鲁棒性"""
    return np.array([cv2.equalizeHist(img) for img in images])


def preprocess_clahe(images, clip_limit=2.0, grid_size=(8, 8)):
    """自适应直方图均衡化(CLAHE)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return np.array([clahe.apply(img) for img in images])


def preprocess_gaussian_filter(images, ksize=5):
    """高斯滤波 - 去噪"""
    return np.array([cv2.GaussianBlur(img, (ksize, ksize), 0) for img in images])


def preprocess_median_filter(images, ksize=5):
    """中值滤波 - 去噪"""
    return np.array([cv2.medianBlur(img, ksize) for img in images])


def preprocess_bilateral_filter(images, d=9, sigma_color=75, sigma_space=75):
    """双边滤波 - 保边去噪"""
    return np.array([cv2.bilateralFilter(img, d, sigma_color, sigma_space) for img in images])


def preprocess_nlm_denoising(images, h=10, template_window=7, search_window=21):
    """非局部均值去噪"""
    return np.array([cv2.fastNlMeansDenoising(img, None, h, template_window, search_window) for img in images])


def preprocess_gradient_magnitude(images):
    """梯度幅度特征 - 对光照变化具有不变性"""
    result = []
    for img in images:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mag = np.clip(grad_mag / grad_mag.max() * 255, 0, 255).astype(np.uint8)
        result.append(grad_mag)
    return np.array(result)


def preprocess_illuminant_normalized(images):
    """光照归一化"""
    result = []
    for img in images:
        img_f = img.astype(np.float64) + 1.0
        illuminant = cv2.GaussianBlur(img_f, (0, 0), sigmaX=20)
        normalized = img_f / illuminant * 128.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        result.append(normalized)
    return np.array(result)


def add_gaussian_noise(images, mean=0, sigma=25, seed=42):
    """添加高斯噪声(用于测试鲁棒性)"""
    rng = np.random.RandomState(seed)
    noisy = []
    for img in images:
        noise = rng.normal(mean, sigma, img.shape).astype(np.float64)
        noisy_img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        noisy.append(noisy_img)
    return np.array(noisy)


def add_salt_pepper_noise(images, amount=0.05, seed=42):
    """添加椒盐噪声(用于测试鲁棒性)"""
    rng = np.random.RandomState(seed)
    noisy = []
    for img in images:
        noisy_img = img.copy()
        num_salt = int(amount * img.size / 2)
        coords = tuple(rng.randint(0, d, num_salt) for d in img.shape)
        noisy_img[coords] = 255
        coords = tuple(rng.randint(0, d, num_salt) for d in img.shape)
        noisy_img[coords] = 0
        noisy.append(noisy_img)
    return np.array(noisy)


def simulate_occlusion(images, occlusion_ratio=0.2, position='random', seed=42):
    """模拟遮挡"""
    rng = np.random.RandomState(seed)
    occluded = []
    for img in images:
        h, w = img.shape
        occ_h = int(h * np.sqrt(occlusion_ratio))
        occ_w = int(w * np.sqrt(occlusion_ratio))
        if position == 'random':
            y = rng.randint(0, h - occ_h)
            x = rng.randint(0, w - occ_w)
        elif position == 'top':
            y, x = 0, (w - occ_w) // 2
        elif position == 'bottom':
            y, x = h - occ_h, (w - occ_w) // 2
        elif position == 'left':
            y, x = (h - occ_h) // 2, 0
        elif position == 'right':
            y, x = (h - occ_h) // 2, w - occ_w
        else:
            y, x = (h - occ_h) // 2, (w - occ_w) // 2
        occ_img = img.copy()
        occ_img[y:y+occ_h, x:x+occ_w] = 0
        occluded.append(occ_img)
    return np.array(occluded)


def simulate_illumination_change(images, gamma=0.5):
    """模拟光照变化"""
    result = []
    for img in images:
        img_float = img.astype(np.float64) / 255.0
        img_gamma = np.power(img_float, gamma)
        result.append((img_gamma * 255).astype(np.uint8))
    return np.array(result)


def occlusion_aware_predict(system, image, img_size=(80, 80), grid_size=3):
    """遮挡感知识别: 将图像分块, 对每块独立识别后投票"""
    h, w = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    votes = {}
    for i in range(grid_size):
        for j in range(grid_size):
            cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            if cell.size == 0:
                continue
            cell_resized = cv2.resize(cell, img_size)
            try:
                label, conf = system.predict(cell_resized)
                votes[label] = votes.get(label, 0) + conf
            except Exception:
                continue
    if not votes:
        return None, 0
    best_label = max(votes, key=votes.get)
    total_conf = sum(votes.values())
    return best_label, votes[best_label] / total_conf if total_conf > 0 else 0


def occlusion_aware_predict_v2(system, image, img_size=(80, 80), grid_sizes=(2, 3, 4)):
    """改进的遮挡感知识别: 多尺度分块投票"""
    h, w = image.shape
    votes = {}
    for grid_size in grid_sizes:
        cell_h, cell_w = h // grid_size, w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                if cell.size == 0:
                    continue
                cell_resized = cv2.resize(cell, img_size)
                try:
                    label, conf = system.predict(cell_resized)
                    weight = conf / len(grid_sizes)
                    votes[label] = votes.get(label, 0) + weight
                except Exception:
                    continue
    if not votes:
        return None, 0
    best_label = max(votes, key=votes.get)
    total_conf = sum(votes.values())
    return best_label, votes[best_label] / total_conf if total_conf > 0 else 0


# ============================================================
# 特征提取
# ============================================================

class EigenfacesExtractor:
    """特征脸(Eigenfaces)特征提取 - PCA降维"""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None
        self.mean_face = None

    def fit(self, X):
        n_samples = X.shape[0]
        flat = X.reshape(n_samples, -1).astype(np.float64)
        self.mean_face = flat.mean(axis=0)
        flat -= self.mean_face
        n_comp = min(self.n_components, n_samples - 1, flat.shape[1])
        self.pca = PCA(n_components=n_comp, whiten=True, svd_solver='full')
        self.pca.fit(flat)
        return self

    def transform(self, X):
        flat = X.reshape(X.shape[0], -1).astype(np.float64)
        flat -= self.mean_face
        return self.pca.transform(flat)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LBPExtractor:
    """LBP(Local Binary Pattern)特征提取 - 向量化实现"""

    def __init__(self, radius=1, n_points=8, grid_x=3, grid_y=3):
        self.radius = radius
        self.n_points = n_points
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                        (1, 1), (1, 0), (1, -1), (0, -1)]

    def _compute_lbp_fast(self, img):
        """向量化LBP计算"""
        h, w = img.shape
        r = self.radius
        center = img[r:h-r, r:w-r].astype(np.int32)
        code = np.zeros_like(center, dtype=np.uint8)
        for k, (dy, dx) in enumerate(self.offsets[:self.n_points]):
            neighbor = img[r+dy:h-r+dy, r+dx:w-r+dx].astype(np.int32)
            code |= ((neighbor >= center).astype(np.uint8) << k)
        return code

    def _lbp_histogram(self, img):
        """计算单张图像分块LBP直方图"""
        lbp = self._compute_lbp_fast(img)
        h, w = lbp.shape
        cell_h, cell_w = h // self.grid_y, w // self.grid_x
        histograms = []
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256), density=True)
                histograms.append(hist)
        return np.concatenate(histograms)

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([self._lbp_histogram(img) for img in X])

    def fit_transform(self, X):
        return self.transform(X)


class GaborExtractor:
    """Gabor滤波器特征提取 - 分块统计"""

    def __init__(self, frequencies=(0.1, 0.2, 0.3), orientations=(0, np.pi/4, np.pi/2, 3*np.pi/4),
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
                kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F)
                self.kernels.append(kernel)

    def _extract_features(self, img):
        features = []
        h, w = img.shape
        cell_h, cell_w = h // self.grid_y, w // self.grid_x
        for kernel in self.kernels:
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            for i in range(self.grid_y):
                for j in range(self.grid_x):
                    cell = filtered[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    features.append(cell.mean())
                    features.append(cell.std())
        return np.array(features)

    def fit(self, X):
        self._create_kernels()
        return self

    def transform(self, X):
        if not self.kernels:
            self._create_kernels()
        return np.array([self._extract_features(img) for img in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class HOGExtractor:
    """HOG (Histogram of Oriented Gradients) 特征提取"""

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

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('hog', None)  # cv2.HOGDescriptor is not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.hog = None  # will be recreated lazily


class CombinedExtractor:
    """组合特征: Eigenfaces + LBP (LBP先PCA降维再拼接)"""

    def __init__(self, eigen_components=50, lbp_components=100, lbp_grid=3):
        self.eigen = EigenfacesExtractor(n_components=eigen_components)
        self.lbp = LBPExtractor(grid_x=lbp_grid, grid_y=lbp_grid)
        self.lbp_scaler = StandardScaler()
        self.lbp_pca = PCA(n_components=lbp_components, svd_solver='full')

    def fit(self, X):
        self.eigen.fit(X)
        lbp_feat_raw = self.lbp.transform(X)
        lbp_feat_scaled = self.lbp_scaler.fit_transform(lbp_feat_raw)
        n_comp = min(self.lbp_pca.n_components, lbp_feat_scaled.shape[0] - 1, lbp_feat_scaled.shape[1])
        self.lbp_pca = PCA(n_components=n_comp, svd_solver='full')
        self.lbp_pca.fit(lbp_feat_scaled)
        return self

    def transform(self, X):
        eigen_feat = self.eigen.transform(X)
        lbp_feat_raw = self.lbp.transform(X)
        lbp_feat_scaled = self.lbp_scaler.transform(lbp_feat_raw)
        lbp_feat = self.lbp_pca.transform(lbp_feat_scaled)
        return np.hstack([eigen_feat, lbp_feat])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# 人脸识别系统核心
# ============================================================

class FaceRecognitionSystem:
    """人脸识别系统"""

    FEATURE_METHODS = {
        'eigenfaces': EigenfacesExtractor,
        'lbp': LBPExtractor,
        'gabor': GaborExtractor,
        'hog': HOGExtractor,
        'combined': CombinedExtractor,
    }

    CLASSIFIER_METHODS = {
        'svm': lambda: SVC(kernel='rbf', C=1, gamma='scale', probability=True, class_weight='balanced'),
        'knn': lambda: KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine'),
    }

    PREPROCESS_METHODS = {
        'none': lambda imgs: imgs,
        'hist_eq': preprocess_histogram_eq,
        'clahe': preprocess_clahe,
        'gaussian': preprocess_gaussian_filter,
        'median': preprocess_median_filter,
        'bilateral': preprocess_bilateral_filter,
        'nlm': preprocess_nlm_denoising,
        'gradient': preprocess_gradient_magnitude,
        'illuminant_norm': preprocess_illuminant_normalized,
    }

    def __init__(self, feature_method='eigenfaces', classifier_method='svm',
                 preprocess='none', n_components=50):
        self.feature_method = feature_method
        self.classifier_method = classifier_method
        self.preprocess = preprocess
        self.n_components = n_components
        self.img_size = (80, 80)

        self.extractor = self._create_extractor()
        self.classifier = self.CLASSIFIER_METHODS[classifier_method]()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.test_images = None
        self.test_labels = None

    @staticmethod
    def _plain_label(label):
        return label.item() if hasattr(label, "item") else label

    def _create_extractor(self):
        if self.feature_method == 'eigenfaces':
            return EigenfacesExtractor(n_components=self.n_components)
        elif self.feature_method == 'lbp':
            return LBPExtractor()
        elif self.feature_method == 'gabor':
            return GaborExtractor()
        elif self.feature_method == 'hog':
            return HOGExtractor(win_size=self.img_size)
        elif self.feature_method == 'combined':
            return CombinedExtractor(eigen_components=self.n_components)
        else:
            raise ValueError(f"未知特征方法: {self.feature_method}")

    def preprocess_images(self, images):
        return self.PREPROCESS_METHODS[self.preprocess](images)

    def train(self, images, labels, test_size=0.3, augment=False, cv_folds=0):
        """训练人脸识别模型

        Args:
            cv_folds: >0 时在原始图像上做全数据集 CV 评估 (增强在每个折内部进行, 无泄漏)
        """
        if images.ndim == 3:
            self.img_size = (images.shape[2], images.shape[1])
            self.extractor = self._create_extractor()

        y = self.label_encoder.fit_transform(labels)

        # 划分训练/测试集(增强前)
        X_train_img, X_test_img, y_train, y_test = train_test_split(
            images, y, test_size=test_size, random_state=42, stratify=y
        )

        # 保存测试集供鲁棒性测试使用
        self.test_images = X_test_img
        self.test_labels = y_test

        # 全数据集 CV 评估 (在原始图像上, 增强在折内进行)
        if cv_folds > 0 and self.classifier_method == 'svm':
            n_folds = min(cv_folds, min(np.bincount(y)))
            if n_folds < 2:
                n_folds = 2
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            c_values = [0.1, 1, 10]
            best_score = -1
            best_c = 1.0

            for c in c_values:
                fold_scores = []
                for train_idx, val_idx in cv.split(images, y):
                    X_tr_img, X_val_img = images[train_idx], images[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    # 只增强训练折
                    X_tr_aug, y_tr_aug = augment_dataset(X_tr_img, y_tr)
                    X_tr_aug = self.preprocess_images(X_tr_aug)
                    X_val_img = self.preprocess_images(X_val_img)
                    ext = self._create_extractor()
                    X_tr_feat = ext.fit_transform(X_tr_aug)
                    X_val_feat = ext.transform(X_val_img)
                    scaler = StandardScaler()
                    X_tr_feat = scaler.fit_transform(X_tr_feat)
                    X_val_feat = scaler.transform(X_val_feat)
                    clf = SVC(kernel='rbf', C=c, gamma='scale', class_weight='balanced')
                    clf.fit(X_tr_feat, y_tr_aug)
                    fold_scores.append(clf.score(X_val_feat, y_val))
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_c = c

            self.cv_accuracy = best_score
            self.best_params = {'C': best_c, 'gamma': 'scale'}

        # 训练最终模型 (增强全部训练数据)
        if augment or cv_folds > 0:
            X_train_img, y_train = augment_dataset(X_train_img, y_train)

        # 预处理
        X_train_img = self.preprocess_images(X_train_img)
        X_test_img = self.preprocess_images(X_test_img)

        # 特征提取
        self.extractor.fit(X_train_img)
        X_train_feat = self.extractor.transform(X_train_img)
        X_test_feat = self.extractor.transform(X_test_img)

        # 归一化
        X_train_feat = self.scaler.fit_transform(X_train_feat)
        X_test_feat = self.scaler.transform(X_test_feat)

        # 用最优参数训练
        if cv_folds > 0 and hasattr(self, 'best_params'):
            self.classifier = SVC(kernel='rbf', probability=True, class_weight='balanced',
                                  C=self.best_params['C'], gamma=self.best_params['gamma'])
        self.classifier.fit(X_train_feat, y_train)

        # 评估
        self.train_accuracy = self.classifier.score(X_train_feat, y_train)
        self.test_accuracy = self.classifier.score(X_test_feat, y_test)
        self.is_trained = True

        # Top-5 准确率
        n_classes = len(self.label_encoder.classes_)
        top5_acc = None
        if n_classes >= 5 and hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(X_test_feat)
            top5_acc = top_k_accuracy_score(y_test, proba, k=5, labels=range(n_classes))

        y_pred = self.classifier.predict(X_test_feat)
        report = classification_report(y_test, y_pred,
                                       target_names=[str(c) for c in self.label_encoder.classes_],
                                       output_dict=True, zero_division=0)

        result = {
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'top5_accuracy': top5_acc,
            'report': report,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_classes': n_classes,
        }
        if cv_folds > 0 and hasattr(self, 'cv_accuracy'):
            result['cv_accuracy'] = self.cv_accuracy
            result['best_params'] = self.best_params
        return result

    def predict(self, image):
        """预测单张图像"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        image = self.preprocess_images(image)
        feat = self.extractor.transform(image)
        feat = self.scaler.transform(feat)
        label_idx = self.classifier.predict(feat)[0]
        label = self._plain_label(self.label_encoder.inverse_transform([label_idx])[0])
        if hasattr(self.classifier, "predict_proba"):
            prob = self.classifier.predict_proba(feat)[0]
            confidence = prob.max()
        else:
            confidence = 1.0
        return label, confidence

    def predict_top_k(self, image, k=3):
        """预测单张图像, 返回 Top-k 结果"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        image = self.preprocess_images(image)
        feat = self.extractor.transform(image)
        feat = self.scaler.transform(feat)
        if hasattr(self.classifier, "predict_proba"):
            prob = self.classifier.predict_proba(feat)[0]
            top_k_idx = np.argsort(prob)[::-1][:k]
            results = []
            for idx in top_k_idx:
                label = self._plain_label(self.label_encoder.inverse_transform([idx])[0])
                confidence = float(prob[idx])
                results.append((label, confidence))
            return results

        label_idx = self.classifier.predict(feat)[0]
        label = self._plain_label(self.label_encoder.inverse_transform([label_idx])[0])
        return [(label, 1.0)]

    def evaluate_robustness(self, test_images, test_labels, noise_type='gaussian', noise_params=None):
        """评估鲁棒性: 添加噪声后测试"""
        if noise_params is None:
            noise_params = {}
        if noise_type == 'gaussian':
            noisy = add_gaussian_noise(test_images, **noise_params)
        elif noise_type == 'salt_pepper':
            noisy = add_salt_pepper_noise(test_images, **noise_params)
        elif noise_type == 'occlusion':
            noisy = simulate_occlusion(test_images, **noise_params)
        elif noise_type == 'illumination':
            noisy = simulate_illumination_change(test_images, **noise_params)
        elif noise_type == 'blur':
            ksize = noise_params.get('ksize', 5)
            noisy = np.array([cv2.GaussianBlur(img, (ksize, ksize), 0) for img in test_images])
        else:
            raise ValueError(f"未知噪声类型: {noise_type}")

        noisy = self.preprocess_images(noisy)
        feat = self.extractor.transform(noisy)
        feat = self.scaler.transform(feat)
        y = self.label_encoder.transform(test_labels)
        return self.classifier.score(feat, y)


# ============================================================
# 便捷函数
# ============================================================

def run_experiment(data_dir, feature='eigenfaces', classifier='svm',
                   preprocess='none', n_components=50, test_size=0.3, augment=False):
    """运行完整实验"""
    print(f"加载数据集: {data_dir}")
    images, labels = load_feret_dataset(data_dir)
    print(f"  样本数: {len(images)}, 人数: {len(np.unique(labels))}")

    system = FaceRecognitionSystem(
        feature_method=feature,
        classifier_method=classifier,
        preprocess=preprocess,
        n_components=n_components,
    )

    print(f"训练模型: {feature} + {classifier}, 预处理: {preprocess}, 增强: {augment}")
    result = system.train(images, labels, test_size=test_size, augment=augment)
    print(f"  训练集准确率: {result['train_accuracy']:.4f}")
    print(f"  测试集准确率: {result['test_accuracy']:.4f}")
    if result.get('top5_accuracy') is not None:
        print(f"  Top-5 准确率: {result['top5_accuracy']:.4f}")
    return system, result


if __name__ == '__main__':
    feret_dir = os.path.join(os.path.dirname(__file__),
                             '数据库-feret_k175_s7_w80_h80', 'feret_k175_s7_w80_h80')
    if os.path.exists(feret_dir):
        system, result = run_experiment(feret_dir, feature='eigenfaces', classifier='svm')
    else:
        print(f"数据集目录不存在: {feret_dir}")
