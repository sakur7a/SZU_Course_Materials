import numpy as np
import scipy.io as sio
from pathlib import Path
import time
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


class SimpleKMeansModel:
    def __init__(self, centers, inertia, n_iter):
        self.cluster_centers_ = centers
        self.inertia_ = inertia
        self.n_iter_ = n_iter


def _euclidean_distances_squared(X, centers):
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    C_sq = np.sum(centers ** 2, axis=1)
    distances = X_sq + C_sq - 2 * X @ centers.T
    np.maximum(distances, 0, out=distances)
    return distances


def _kmeans_plusplus_init(X, k, rng, n_local_trials=None):
    n_samples = X.shape[0]
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(k))

    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    # 选取第一个中心
    first_idx = rng.integers(0, n_samples)
    centers[0] = X[first_idx]

    # 到最近中心的距离平方
    closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

    for c in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        candidate_ids = rng.choice(n_samples, size=n_local_trials, replace=False, p=probs)

        best_candidate = None
        best_cost = None
        best_dist_sq = None

        for candidate in candidate_ids:
            dist_sq = np.sum((X - X[candidate]) ** 2, axis=1)
            new_dist_sq = np.minimum(closest_dist_sq, dist_sq)
            cost = new_dist_sq.sum()
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_candidate = candidate
                best_dist_sq = new_dist_sq

        centers[c] = X[best_candidate]
        closest_dist_sq = best_dist_sq

    return centers


def _pca_init(X, k):
    pca = PCA(n_components=k, random_state=42)
    pca.fit(X)
    mean = pca.mean_
    components = pca.components_
    variances = pca.explained_variance_

    centers = []
    for i in range(k):
        scale = float(np.sqrt(max(variances[i], 0.0)))
        center = mean + scale * components[i]
        centers.append(np.clip(center, 0.0, 1.0))
    return np.vstack(centers)


def _run_single_kmeans(X, k, max_iter, tol, rng, *, init_mode='random', init_centers=None):
    n_samples = X.shape[0]
    if n_samples < k:
        raise ValueError(f"样本数 {n_samples} 小于聚类数 {k}")

    if init_centers is not None:
        centers = init_centers.copy()
    elif init_mode == 'kmeans++':
        centers = _kmeans_plusplus_init(X, k, rng)
    else:
        indices = rng.choice(n_samples, size=k, replace=False)
        centers = X[indices].copy()

    for iteration in range(1, max_iter + 1):
        distances = _euclidean_distances_squared(X, centers)
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for cluster_id in range(k):
            members = X[labels == cluster_id]
            if len(members) == 0:
                new_centers[cluster_id] = X[rng.integers(0, n_samples)]
            else:
                new_centers[cluster_id] = members.mean(axis=0)

        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        centers = new_centers
        if shift <= tol:
            break

    distances = _euclidean_distances_squared(X, centers)
    labels = np.argmin(distances, axis=1)
    inertia = float(np.sum(distances[np.arange(n_samples), labels]))
    return centers, labels, inertia, iteration


def kmeans_manual(
    X,
    k,
    *,
    n_init=10,
    max_iter=300,
    tol=1e-4,
    random_state=42,
    init_mode='random',
    init_centers=None,
):
    best_inertia = None
    best_centers = None
    best_labels = None
    best_iter = None

    for init_idx in range(n_init):
        rng = np.random.default_rng(random_state + init_idx)
        centers, labels, inertia, n_iter = _run_single_kmeans(
            X,
            k,
            max_iter=max_iter,
            tol=tol,
            rng=rng,
            init_mode=init_mode,
            init_centers=init_centers,
        )

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels
            best_iter = n_iter

    return SimpleKMeansModel(best_centers, best_inertia, best_iter), best_labels

# 设置中文字体，避免图像中文显示为乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 结果输出目录（当前脚本所在目录）
BASE_DIR = Path(__file__).resolve().parent

# ----------------------
# 1. 读取数据（支持不同样本数）
# ----------------------
def load_data(n_samples: int | None = None):
    """读取 MNIST 数据并展平成 (样本数, 784)。

    参数
    ------
    n_samples: int | None
        若指定，则返回前 n_samples 个样本；否则返回全部样本。
    """
    img_path = BASE_DIR / 'train_images.mat'
    label_path = BASE_DIR / 'train_labels.mat'
    if not img_path.exists() or not label_path.exists():
        img_path = BASE_DIR.parent / 'train_images.mat'
        label_path = BASE_DIR.parent / 'train_labels.mat'

    images_mat = sio.loadmat(str(img_path))
    labels_mat = sio.loadmat(str(label_path))

    train_images = images_mat['train_images']  # 形状 (28, 28, N)
    train_labels = labels_mat['train_labels']

    # 展平所有图像
    X = train_images.reshape(28 * 28, -1).T  # (N, 784)
    # 标签展平
    y = train_labels.reshape(-1)
    if y.size < X.shape[0]:
        y = np.squeeze(train_labels, axis=0).reshape(-1)

    # 将 1-10 映射到 0-9（若数据如此存储）
    if y.min() >= 1 and y.max() <= 10:
        y = y - 1

    if n_samples is not None:
        X = X[:n_samples]
        y = y[:n_samples]

    return X, y

# ----------------------
# 2. 数据预处理（归一化）
# ----------------------
def preprocess_data(X):
    # 将像素值从[0,255]缩放到[0,1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# ----------------------
# 3. 执行k-Means聚类（10类）
# ----------------------
def perform_clustering(
    X,
    *,
    random_state: int = 42,
    max_iter: int = 300,
    n_trials: int = 10,
    init_mode: str = 'random',
    init_centers: np.ndarray | None = None,
    method_label: str | None = None,
):
    """执行多次随机初始化的标准 k-Means，返回目标函数最小的一次。"""

    best_model = None
    best_inertia = None
    best_labels = None

    for trial in range(n_trials):
        seed = random_state + trial
        model, labels = kmeans_manual(
            X,
            10,
            n_init=1,
            max_iter=max_iter,
            tol=1e-4,
            random_state=seed,
            init_mode=init_mode,
            init_centers=init_centers,
        )
        inertia = model.inertia_

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_model = model
            best_labels = labels

    if method_label is not None:
        method_name = method_label
    else:
        mode_map = {
            'random': 'KMeans (manual, random init)',
            'kmeans++': 'KMeans (manual, k-means++ init)',
            'pca': 'KMeans (manual, PCA init)',
        }
        method_name = mode_map.get(init_mode, 'KMeans (manual)')
    return best_labels, best_model, method_name, best_inertia

# ----------------------
# 4. 评估聚类性能（ARI、NMI、纯度）
# ----------------------
def evaluate_results(y_true, y_pred, X_features=None, *, silhouette_sample_size=2000):
    # 同质性、完整性、V 度量
    homo = homogeneity_score(y_true, y_pred)
    compl = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)

    # 调整兰德指数、调整互信息
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    # 纯度（Purity）
    cm = confusion_matrix(y_true, y_pred)
    purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)

    # 轮廓系数（可能采样以降低计算量）
    silhouette = float('nan')
    if X_features is not None and len(np.unique(y_pred)) > 1 and len(X_features) > 1:
        sample_size = None
        if len(X_features) > silhouette_sample_size:
            sample_size = silhouette_sample_size
        try:
            silhouette = silhouette_score(
                X_features,
                y_pred,
                sample_size=sample_size,
                random_state=42,
            )
        except Exception:
            silhouette = float('nan')

    # 打印评估结果
    print("聚类性能评估：")
    print(f"同质性（homo）：{homo:.4f}")
    print(f"完整性（compl）：{compl:.4f}")
    print(f"V 度量（v-meas）：{v_measure:.4f}")
    print(f"调整兰德指数（ARI）：{ari:.4f}")
    print(f"调整互信息（AMI）：{ami:.4f}")
    if not np.isnan(silhouette):
        print(f"轮廓系数（silhouette）：{silhouette:.4f}")
    else:
        print("轮廓系数（silhouette）：NaN（簇数量不足或计算失败）")
    print(f"纯度（Purity）：{purity:.4f}")

    return {
        'Homogeneity': homo,
        'Completeness': compl,
        'VMeasure': v_measure,
        'ARI': ari,
        'AMI': ami,
        'Silhouette': silhouette,
        'Purity': purity,
    }


def save_metrics(metrics, filepath):
    lines = [
        '指标,数值',
        f"算法,{metrics['Method']}",
        f"同质性(homo),{metrics['Homogeneity']:.6f}",
        f"完整性(compl),{metrics['Completeness']:.6f}",
        f"V度量(v-meas),{metrics['VMeasure']:.6f}",
        f"调整兰德指数(ARI),{metrics['ARI']:.6f}",
        f"调整互信息(AMI),{metrics['AMI']:.6f}",
        f"轮廓系数(silhouette),{metrics['Silhouette']:.6f}",
        f"纯度(Purity),{metrics['Purity']:.6f}",
        f"目标函数J(惯性),{metrics['Inertia']:.6f}",
        f"运行时间(秒),{metrics['RuntimeSeconds']:.6f}"
    ]
    filepath.write_text('\n'.join(lines), encoding='utf-8')

# ----------------------
# 5. 可视化聚类结果
# ----------------------
def visualize_results(X, y_true, y_pred, output_dir: Path, *, suffix: str = ''):
    # 可视化1：每个簇的样本示例（每个簇显示5张图）
    rng = np.random.default_rng(42)
    plt.figure(figsize=(15, 10))
    for cluster_id in range(10):
        # 找到该簇的所有样本索引
        cluster_samples = X[y_pred == cluster_id]
        if len(cluster_samples) == 0:
            continue
        # 随机选5张（若不足5张则全选）
        n_samples = min(5, len(cluster_samples))
        selected = rng.choice(len(cluster_samples), n_samples, replace=False)
        
        for i, idx in enumerate(selected):
            plt.subplot(10, 5, cluster_id*5 + i + 1)
            # 还原为28x28图像
            plt.imshow(cluster_samples[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                # 标题显示簇ID和该簇中最多的真实数字
                true_labels_in_cluster = y_true[y_pred == cluster_id]
                if len(true_labels_in_cluster) > 0:
                    most_common = np.bincount(true_labels_in_cluster).argmax()
                    plt.title(f"簇{cluster_id}\n(主要是{most_common})")
    plt.tight_layout()
    sample_fig_path = output_dir / f'cluster_samples{suffix}.png'
    sample_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(sample_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 可视化2：混淆矩阵（真实标签 vs 聚类结果）
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'簇{i}' for i in range(10)],
                yticklabels=[f'数字{i}' for i in range(10)])
    plt.xlabel('聚类结果（簇）')
    plt.ylabel('真实标签（数字）')
    plt.title(f'前{len(y_true)}张图像的聚类混淆矩阵')
    conf_fig_path = output_dir / f'confusion_matrix{suffix}.png'
    conf_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(conf_fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_scatter(X_features, labels, output_dir: Path, *, suffix: str = '', sample_size: int | None = 5000):
    if len(X_features) == 0:
        return

    rng = np.random.default_rng(42)
    if sample_size is not None and len(X_features) > sample_size:
        idx = rng.choice(len(X_features), sample_size, replace=False)
        X_plot = X_features[idx]
        labels_plot = labels[idx]
        title_suffix = f'(采样 {sample_size} / {len(X_features)})'
    else:
        X_plot = X_features
        labels_plot = labels
        title_suffix = f'(全部 {len(X_features)} 样本)'

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_plot)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels_plot, palette='tab10', s=12, linewidth=0)
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.title(f'PCA 2D 聚类可视化 {title_suffix}')
    plt.legend(title='簇', loc='best', fontsize='small', markerscale=1.5)
    plt.tight_layout()

    pca_fig_path = output_dir / f'pca_scatter{suffix}.png'
    pca_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pca_fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_comparison(
    X_features,
    y_true,
    y_pred,
    output_dir: Path,
    *,
    suffix: str = '',
    sample_size: int | None = 5000,
):
    if len(X_features) == 0:
        return

    rng = np.random.default_rng(42)
    if sample_size is not None and len(X_features) > sample_size:
        idx = rng.choice(len(X_features), sample_size, replace=False)
        X_plot = X_features[idx]
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
        title_suffix = f'(采样 {sample_size} / {len(X_features)})'
    else:
        X_plot = X_features
        y_true_plot = y_true
        y_pred_plot = y_pred
        title_suffix = f'(全部 {len(X_features)} 样本)'

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_plot)

    palette = sns.color_palette('tab10', 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(
        ax=axes[0],
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=y_true_plot,
        palette=palette,
        s=12,
        linewidth=0,
        legend='brief',
    )
    axes[0].set_title(f'真实标签分布 {title_suffix}')
    axes[0].set_xlabel('主成分 1')
    axes[0].set_ylabel('主成分 2')
    axes[0].legend(title='数字', fontsize='small', markerscale=1.5)

    sns.scatterplot(
        ax=axes[1],
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=y_pred_plot,
        palette=palette,
        s=12,
        linewidth=0,
        legend='brief',
    )
    axes[1].set_title(f'聚类结果分布 {title_suffix}')
    axes[1].set_xlabel('主成分 1')
    axes[1].set_ylabel('主成分 2')
    axes[1].legend(title='簇', fontsize='small', markerscale=1.5)

    fig.suptitle('PCA 2D 聚类前后对比', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    compare_path = output_dir / f'pca_comparison{suffix}.png'
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ----------------------
# 指标汇总与主函数
# ----------------------
def save_summary(metrics_list, filepath: Path):
    header = ['N', 'Method', 'Homogeneity', 'Completeness', 'VMeasure', 'ARI', 'AMI', 'Silhouette', 'Purity', 'Inertia', 'RuntimeSeconds']
    lines = [','.join(header)]
    for item in metrics_list:
        lines.append(
            f"{item['N']},{item['Method']},{item['Homogeneity']:.6f},{item['Completeness']:.6f},{item['VMeasure']:.6f},{item['ARI']:.6f},{item['AMI']:.6f},{item['Silhouette']:.6f},{item['Purity']:.6f},{item['Inertia']:.6f},{item['RuntimeSeconds']:.6f}"
        )
    filepath.write_text('\n'.join(lines), encoding='utf-8')


def main():
    experiment_configs = [
        {'N': 100, 'n_trials': 10},
        {'N': 1000, 'n_trials': 10},
    {'N': None, 'n_trials': 2},  # 全量 60000，仅少量随机重启以控制耗时
    ]
    metrics_collection = []

    # 读取全部数据
    X_all, y_all = load_data()
    total_samples = X_all.shape[0]
    print(f"数据加载完成：共{total_samples}个样本，每个样本{X_all.shape[1]}维特征")

    for config in experiment_configs:
        N = config['N'] if config['N'] is not None else total_samples
        if N > total_samples:
            print(f"警告：指定样本数 {N} 超过数据量 {total_samples}，自动截断为 {total_samples}")
            N = total_samples

        print(f"\n===== 聚类实验：前 {N} 张图像 =====")
        X = X_all[:N]
        y_true = y_all[:N]

        # 数据归一化
        X_scaled = preprocess_data(X)

        run_variants = [
            {
                'suffix': '',
                'init_mode': 'random',
                'init_centers': None,
                'method_label': 'KMeans (manual, random init)',
                'n_trials': config['n_trials'],
                'generate_outputs': True,
            }
        ]

        if N == total_samples:
            pca_init_centers = _pca_init(X_scaled, 10)
            run_variants.append(
                {
                    'suffix': '_kpp',
                    'init_mode': 'kmeans++',
                    'init_centers': None,
                    'method_label': 'KMeans (manual, k-means++ init)',
                    'n_trials': config['n_trials'],
                    'generate_outputs': False,
                }
            )
            run_variants.append(
                {
                    'suffix': '_pcaInit',
                    'init_mode': 'pca',
                    'init_centers': pca_init_centers,
                    'method_label': 'KMeans (manual, PCA init)',
                    'n_trials': 1,
                    'generate_outputs': False,
                }
            )

        for variant in run_variants:
            print(f"\n--- 方案：{variant['method_label']} ---")
            start_time = time.perf_counter()
            y_pred, model, method_name, best_inertia = perform_clustering(
                X_scaled,
                n_trials=variant['n_trials'],
                init_mode=variant['init_mode'],
                init_centers=variant.get('init_centers'),
                method_label=variant.get('method_label'),
            )

            metrics = evaluate_results(
                y_true,
                y_pred,
                X_scaled,
                silhouette_sample_size=2000 if N < 20000 else 3000,
            )
            metrics['N'] = N
            metrics['Method'] = method_name
            metrics['Inertia'] = best_inertia
            elapsed = time.perf_counter() - start_time
            metrics['RuntimeSeconds'] = elapsed
            metrics_collection.append(metrics)

            suffix = f"_N{N}{variant['suffix']}"
            metrics_path = BASE_DIR / f'clustering_metrics{suffix}.csv'
            save_metrics(metrics, metrics_path)

            print("结果已保存：")
            print(f" - 指标文件: {metrics_path}")

            if variant.get('generate_outputs', True):
                visualize_results(X, y_true, y_pred, BASE_DIR, suffix=suffix)
                plot_pca_scatter(
                    X_scaled,
                    y_pred,
                    BASE_DIR,
                    suffix=suffix,
                    sample_size=3000 if N > 3000 else None,
                )
                plot_pca_comparison(
                    X_scaled,
                    y_true,
                    y_pred,
                    BASE_DIR,
                    suffix=suffix,
                    sample_size=3000 if N > 3000 else None,
                )
                print(f" - 簇样本图: {BASE_DIR / f'cluster_samples{suffix}.png'}")
                print(f" - 混淆矩阵: {BASE_DIR / f'confusion_matrix{suffix}.png'}")
                print(f" - PCA 可视化: {BASE_DIR / f'pca_scatter{suffix}.png'}")
                print(f" - PCA 前后对比: {BASE_DIR / f'pca_comparison{suffix}.png'}")
            else:
                print(" - 可视化: 已跳过（仅记录指标）")

            print(f" - 运行时间: {elapsed:.2f} 秒")
            print(f" - 算法: {method_name} (最优J={best_inertia:.2f}, 重复次数={variant['n_trials']})")

    # 汇总对比
    summary_path = BASE_DIR / 'clustering_metrics_summary.csv'
    save_summary(metrics_collection, summary_path)
    print(f"\n指标对比汇总已保存至: {summary_path}")

if __name__ == "__main__":
    main()
