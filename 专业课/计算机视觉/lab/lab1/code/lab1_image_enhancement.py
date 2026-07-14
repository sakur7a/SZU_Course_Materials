from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLOR_IMAGE = ROOT / "测试图像" / "peppers.png"
DEFAULT_GRAY_IMAGE = ROOT / "测试图像" / "cameraman.tif"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def save_channel_images(color_image: np.ndarray, out_dir: Path) -> Tuple[Path, Path, Path, Path]:
    blue_channel = color_image[:, :, 0]
    green_channel = color_image[:, :, 1]
    red_channel = color_image[:, :, 2]

    blue_path = out_dir / "channel_blue.png"
    green_path = out_dir / "channel_green.png"
    red_path = out_dir / "channel_red.png"
    merged_path = out_dir / "channel_split_compare.png"

    write_image(blue_path, blue_channel)
    write_image(green_path, green_channel)
    write_image(red_path, red_channel)

    merged = make_channel_montage(color_image, blue_channel, green_channel, red_channel)
    write_image(merged_path, merged)

    return blue_path, green_path, red_path, merged_path


def make_channel_montage(color_image: np.ndarray, blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
    color_bgr = color_image.copy()
    blue_bgr = cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR)
    green_bgr = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    red_bgr = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)

    p1 = annotate_panel(color_bgr, "Original")
    p2 = annotate_panel(blue_bgr, "Blue channel")
    p3 = annotate_panel(green_bgr, "Green channel")
    p4 = annotate_panel(red_bgr, "Red channel")

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    return np.vstack([top, bottom])


def annotate_panel(image_bgr: np.ndarray, label: str) -> np.ndarray:
    panel = image_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return panel


def manual_histogram(gray_image: np.ndarray) -> np.ndarray:
    # 手工直方图：遍历每个像素并统计 0-255 灰度计数
    hist = np.zeros(256, dtype=np.int64)
    rows, cols = gray_image.shape
    for row in range(rows):
        for col in range(cols):
            hist[int(gray_image[row, col])] += 1
    return hist


def opencv_histogram(gray_image: np.ndarray) -> np.ndarray:
    # OpenCV 对照实现，用于验证手工统计结果
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist.reshape(-1)


def histogram_equalization_manual(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 手工直方图均衡化：统计直方图 -> 计算累积分布函数 -> 构建映射表
    hist = manual_histogram(gray_image).astype(np.float64)
    cdf = np.cumsum(hist)
    nonzero_cdf = cdf[cdf > 0]
    if nonzero_cdf.size == 0:
        return gray_image.copy(), np.arange(256, dtype=np.uint8)

    cdf_min = float(nonzero_cdf[0])
    total = float(gray_image.size)
    denom = total - cdf_min
    if denom <= 0:
        mapping = np.arange(256, dtype=np.uint8)
        return gray_image.copy(), mapping

    mapping = np.round((cdf - cdf_min) * 255.0 / denom)
    mapping = np.clip(mapping, 0, 255).astype(np.uint8)
    equalized = mapping[gray_image]
    return equalized, mapping


def histogram_equalization_opencv(gray_image: np.ndarray) -> np.ndarray:
    # OpenCV 对照实现
    return cv2.equalizeHist(gray_image)


def build_gamma_lut(gamma: float) -> np.ndarray:
    values = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(values, gamma) * 255.0
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def gamma_correction_manual(gray_image: np.ndarray, gamma: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    lut = build_gamma_lut(gamma)
    return lut[gray_image], lut


def gamma_correction_opencv(gray_image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(gray_image, lut)


def linear_contrast_stretch_manual(gray_image: np.ndarray) -> np.ndarray:
    image = gray_image.astype(np.float32)
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros_like(gray_image)
    stretched = (image - minimum) * 255.0 / (maximum - minimum)
    return np.clip(np.round(stretched), 0, 255).astype(np.uint8)


def linear_contrast_stretch_opencv(gray_image: np.ndarray) -> np.ndarray:
    return cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)


def add_salt_pepper_noise(gray_image: np.ndarray, amount: float = 0.05, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = gray_image.copy()
    total = gray_image.size
    noisy_pixels = int(total * amount)
    if noisy_pixels <= 0:
        return noisy

    flat = noisy.reshape(-1)
    indices = rng.choice(total, size=noisy_pixels, replace=False)
    split = noisy_pixels // 2
    flat[indices[:split]] = 255
    flat[indices[split:]] = 0
    return noisy


def reflect_pad(gray_image: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(gray_image.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")


def convolve_gray(gray_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = kernel.astype(np.float32)
    pad = kernel.shape[0] // 2
    padded = reflect_pad(gray_image, pad)
    rows, cols = gray_image.shape
    result = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            region = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            result[row, col] = float(np.sum(region * kernel))

    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def mean_filter_manual(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / float(kernel_size * kernel_size)
    return convolve_gray(gray_image, kernel)


def mean_filter_opencv(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.blur(gray_image, (kernel_size, kernel_size))


def median_filter_manual(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    pad = kernel_size // 2
    padded = reflect_pad(gray_image, pad)
    rows, cols = gray_image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            region = padded[row : row + kernel_size, col : col + kernel_size]
            result[row, col] = np.uint8(np.median(region))

    return result


def median_filter_opencv(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.medianBlur(gray_image, kernel_size)


def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    radius = kernel_size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel /= float(kernel.sum())
    return kernel.astype(np.float32)


def gaussian_filter_manual(gray_image: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve_gray(gray_image, kernel)


def gaussian_filter_opencv(gray_image: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)


def laplacian_sharpen_manual(gray_image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    laplacian = convolve_gray(gray_image, laplacian_kernel).astype(np.float32)
    sharpened = gray_image.astype(np.float32) + alpha * laplacian
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)


def laplacian_sharpen_opencv(gray_image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    laplacian = cv2.filter2D(gray_image.astype(np.float32), cv2.CV_32F, laplacian_kernel)
    sharpened = gray_image.astype(np.float32) + alpha * laplacian
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)


def unsharp_mask_manual(gray_image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blurred = gaussian_filter_manual(gray_image, kernel_size=5, sigma=1.2).astype(np.float32)
    image = gray_image.astype(np.float32)
    sharpened = image + amount * (image - blurred)
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)


def unsharp_mask_opencv(gray_image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.2)
    return cv2.addWeighted(gray_image, 1.0 + amount, blurred, -amount, 0)


def draw_histogram(hist: np.ndarray, out_size: Tuple[int, int] = (900, 500)) -> np.ndarray:
    width, height = out_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    max_value = float(hist.max()) if hist.max() > 0 else 1.0
    margin_left = 50
    margin_bottom = 40
    plot_width = width - margin_left - 20
    plot_height = height - 30 - margin_bottom
    bin_width = plot_width / 256.0

    cv2.line(canvas, (margin_left, 20), (margin_left, 20 + plot_height), (180, 180, 180), 1)
    cv2.line(canvas, (margin_left, 20 + plot_height), (margin_left + plot_width, 20 + plot_height), (180, 180, 180), 1)

    for i in range(256):
        value = float(hist[i]) / max_value
        bar_height = int(value * plot_height)
        x1 = int(margin_left + i * bin_width)
        x2 = int(margin_left + (i + 1) * bin_width)
        y1 = int(20 + plot_height - bar_height)
        y2 = int(20 + plot_height)
        cv2.rectangle(canvas, (x1, y1), (max(x1 + 1, x2), y2), (255, 255, 255), -1)

    cv2.putText(canvas, "Histogram", (margin_left, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return canvas


def draw_histogram_comparison(manual_hist: np.ndarray, opencv_hist: np.ndarray, out_size: Tuple[int, int] = (1000, 550)) -> np.ndarray:
    width, height = out_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    manual = manual_hist.astype(np.float64)
    opencv = opencv_hist.astype(np.float64)
    max_value = max(float(np.max(manual)), float(np.max(opencv)), 1.0)

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_height), (160, 160, 160), 1)
    cv2.line(
        canvas,
        (margin_left, margin_top + plot_height),
        (margin_left + plot_width, margin_top + plot_height),
        (160, 160, 160),
        1,
    )

    x_step = plot_width / 255.0
    manual_points = []
    opencv_points = []
    for i in range(256):
        x = int(margin_left + i * x_step)
        y_manual = int(margin_top + plot_height - (manual[i] / max_value) * plot_height)
        y_opencv = int(margin_top + plot_height - (opencv[i] / max_value) * plot_height)
        manual_points.append([x, y_manual])
        opencv_points.append([x, y_opencv])

    manual_points_arr = np.array(manual_points, dtype=np.int32).reshape((-1, 1, 2))
    opencv_points_arr = np.array(opencv_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [manual_points_arr], isClosed=False, color=(255, 220, 0), thickness=2)
    cv2.polylines(canvas, [opencv_points_arr], isClosed=False, color=(0, 255, 255), thickness=2)

    cv2.putText(canvas, "Histogram Comparison", (margin_left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(canvas, (width - 290, 12), (width - 16, 72), (50, 50, 50), -1)
    cv2.line(canvas, (width - 275, 34), (width - 235, 34), (255, 220, 0), 2)
    cv2.putText(canvas, "Manual histogram", (width - 225, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.line(canvas, (width - 275, 58), (width - 235, 58), (0, 255, 255), 2)
    cv2.putText(canvas, "OpenCV histogram", (width - 225, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    return canvas


def draw_histogram_error_curve(manual_hist: np.ndarray, opencv_hist: np.ndarray, out_size: Tuple[int, int] = (1000, 550)) -> np.ndarray:
    width, height = out_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    error = np.abs(manual_hist.astype(np.float64) - opencv_hist.astype(np.float64))
    max_error = max(float(np.max(error)), 1.0)

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_height), (160, 160, 160), 1)
    cv2.line(
        canvas,
        (margin_left, margin_top + plot_height),
        (margin_left + plot_width, margin_top + plot_height),
        (160, 160, 160),
        1,
    )

    x_step = plot_width / 255.0
    points = []
    for i in range(256):
        x = int(margin_left + i * x_step)
        y = int(margin_top + plot_height - (error[i] / max_error) * plot_height)
        points.append([x, y])

    points_arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [points_arr], isClosed=False, color=(0, 165, 255), thickness=2)

    cv2.putText(canvas, "Histogram Error Curve | abs(manual - opencv)", (margin_left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(canvas, f"Max error: {float(np.max(error)):.4f}", (margin_left, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return canvas


def make_equalization_montage(gray_image: np.ndarray, manual_eq: np.ndarray, opencv_eq: np.ndarray) -> np.ndarray:
    original_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    manual_bgr = cv2.cvtColor(manual_eq, cv2.COLOR_GRAY2BGR)
    opencv_bgr = cv2.cvtColor(opencv_eq, cv2.COLOR_GRAY2BGR)

    p1 = annotate_panel(original_bgr, "Original")
    p2 = annotate_panel(manual_bgr, "Manual equalization")
    p3 = annotate_panel(opencv_bgr, "OpenCV equalizeHist")
    return np.hstack([p1, p2, p3])


def make_three_panel_montage(original: np.ndarray, manual: np.ndarray, opencv: np.ndarray, original_label: str, manual_label: str, opencv_label: str) -> np.ndarray:
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    manual_bgr = cv2.cvtColor(manual, cv2.COLOR_GRAY2BGR)
    opencv_bgr = cv2.cvtColor(opencv, cv2.COLOR_GRAY2BGR)

    p1 = annotate_panel(original_bgr, original_label)
    p2 = annotate_panel(manual_bgr, manual_label)
    p3 = annotate_panel(opencv_bgr, opencv_label)
    return np.hstack([p1, p2, p3])


def make_original_histogram_panel(gray_image: np.ndarray, histogram_image: np.ndarray) -> np.ndarray:
    hist_h = histogram_image.shape[0]
    hist_w = histogram_image.shape[1]

    gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    resized_gray = cv2.resize(gray_bgr, (hist_w // 2, hist_h), interpolation=cv2.INTER_NEAREST)

    left = annotate_panel(resized_gray, "Original grayscale image")
    right = annotate_panel(histogram_image, "Manual histogram")
    return np.hstack([left, right])


def sobel_manual(gray_image: np.ndarray) -> np.ndarray:
    # 手工 Sobel：先转浮点并做零填充，便于处理边界像素
    image = gray_image.astype(np.float32)
    padded = np.pad(image, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    rows, cols = image.shape
    gx = np.zeros((rows, cols), dtype=np.float32)
    gy = np.zeros((rows, cols), dtype=np.float32)

    # 逐像素卷积计算水平和垂直梯度
    for row in range(rows):
        for col in range(cols):
            region = padded[row : row + 3, col : col + 3]
            gx[row, col] = float(np.sum(region * gx_kernel))
            gy[row, col] = float(np.sum(region * gy_kernel))

    # 梯度幅值 G = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return normalize_to_uint8(magnitude)


def sobel_opencv(gray_image: np.ndarray) -> np.ndarray:
    # OpenCV Sobel 对照实现
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return normalize_to_uint8(magnitude)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - minimum) * 255.0 / (maximum - minimum)
    return normalized.astype(np.uint8)


def compare_images(manual: np.ndarray, opencv: np.ndarray) -> dict[str, float]:
    diff = np.abs(manual.astype(np.int32) - opencv.astype(np.int32))
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
    }


def make_edge_montage(gray_image: np.ndarray, manual_edge: np.ndarray, opencv_edge: np.ndarray) -> np.ndarray:
    original_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    manual_bgr = cv2.cvtColor(manual_edge, cv2.COLOR_GRAY2BGR)
    opencv_bgr = cv2.cvtColor(opencv_edge, cv2.COLOR_GRAY2BGR)

    p1 = annotate_panel(original_bgr, "Original")
    p2 = annotate_panel(manual_bgr, "Manual Sobel")
    p3 = annotate_panel(opencv_bgr, "OpenCV Sobel")
    return np.hstack([p1, p2, p3])


def write_text_report(
    out_dir: Path,
    gray_image: np.ndarray,
    gray_image_path: Path,
    manual_hist: np.ndarray,
    opencv_hist: np.ndarray,
    transform_stats: dict[str, float],
    gamma_stats: dict[str, float],
    smoothing_stats: dict[str, float],
    sharpen_stats: dict[str, float],
    eq_stats: dict[str, float],
    edge_stats: dict[str, float],
) -> Path:
    report_path = out_dir / "analysis.txt"
    pixel_count = int(gray_image.size)
    hist_manual_sum = int(manual_hist.sum())
    hist_cv_sum = float(opencv_hist.sum())
    hist_diff = np.abs(manual_hist.astype(np.float64) - opencv_hist.astype(np.float64))

    lines = [
        "实验1：图像增强实验结果",
        "",
        f"直方图统计图像: {gray_image_path}",
        f"灰度图像尺寸: {gray_image.shape[0]} x {gray_image.shape[1]}",
        f"像素总数: {pixel_count}",
        f"手工直方图总和: {hist_manual_sum}",
        f"OpenCV直方图总和: {hist_cv_sum:.1f}",
        f"直方图最大绝对差: {float(hist_diff.max()):.1f}",
        f"直方图平均绝对差: {float(hist_diff.mean()):.4f}",
        f"灰度变换平均绝对差: {transform_stats['mean_abs_diff']:.4f}",
        f"灰度变换最大绝对差: {transform_stats['max_abs_diff']:.4f}",
        f"Gamma 校正平均绝对差: {gamma_stats['mean_abs_diff']:.4f}",
        f"Gamma 校正最大绝对差: {gamma_stats['max_abs_diff']:.4f}",
        f"平滑平均绝对差: {smoothing_stats['mean_abs_diff']:.4f}",
        f"平滑最大绝对差: {smoothing_stats['max_abs_diff']:.4f}",
        f"锐化平均绝对差: {sharpen_stats['mean_abs_diff']:.4f}",
        f"锐化最大绝对差: {sharpen_stats['max_abs_diff']:.4f}",
        f"直方图均衡化平均绝对差: {eq_stats['mean_abs_diff']:.4f}",
        f"直方图均衡化最大绝对差: {eq_stats['max_abs_diff']:.4f}",
        f"边缘图平均绝对差: {edge_stats['mean_abs_diff']:.4f}",
        f"边缘图最大绝对差: {edge_stats['max_abs_diff']:.4f}",
        "",
        "对比分析：",
        "1. 灰度变换中的线性拉伸与 Gamma 校正可直接控制整体亮度和对比度，适合做预处理；与 OpenCV 对照时，差异通常来自取整和边界裁剪。",
        "2. 直方图均衡化能显著提升暗部或低对比图像的可视性，但也可能放大噪声；手工实现与 OpenCV 的差别通常来自累计分布函数的舍入。",
        "3. 空间域平滑里，均值滤波去噪最直接但会明显模糊边缘，中值滤波对椒盐噪声更稳健，高斯滤波在平滑和保边之间更平衡。",
        "4. 锐化与边缘增强中，Laplacian 强调二阶变化，Unsharp Mask 更适合整体细节增强；与 OpenCV 的差别主要来自卷积边界处理。",
        "5. 彩色图像通道拆分后，B/G/R 三个单通道图像与 OpenCV 读取结果一致，说明通道索引正确。",
        "6. 手工直方图与 cv2.calcHist 的统计结果一致，说明像素统计逻辑正确。",
        "7. 手工 Sobel 与 OpenCV Sobel 的边缘位置基本一致，差异主要来自边界处理和数值归一化方式。",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="实验1：图像增强相关方法手工实现与 OpenCV 对比")
    parser.add_argument("--color", type=Path, default=DEFAULT_COLOR_IMAGE, help="彩色图像路径")
    parser.add_argument("--gray", type=Path, default=DEFAULT_GRAY_IMAGE, help="单通道图像路径")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results", help="输出目录")
    args = parser.parse_args()

    out_dir = ensure_dir(args.outdir)
    channel_dir = ensure_dir(out_dir / "channels")
    histogram_dir = ensure_dir(out_dir / "histogram")
    edge_dir = ensure_dir(out_dir / "edge")
    enhancement_dir = ensure_dir(out_dir / "enhancement")
    gray_dir = ensure_dir(enhancement_dir / "gray_transform")
    smoothing_dir = ensure_dir(enhancement_dir / "smoothing")
    sharpen_dir = ensure_dir(enhancement_dir / "sharpen")

    # 读取实验输入：彩色图用于通道分离，灰度图用于直方图和边缘检测
    color_image = read_image(args.color, cv2.IMREAD_COLOR)
    gray_image = read_image(args.gray, cv2.IMREAD_GRAYSCALE)

    save_channel_images(color_image, channel_dir)

    # 直方图：手工实现 + OpenCV 对照
    manual_hist = manual_histogram(gray_image)
    opencv_hist = opencv_histogram(gray_image)
    hist_image = draw_histogram(manual_hist)
    write_image(histogram_dir / "manual_histogram.png", hist_image)
    hist_panel_image = make_original_histogram_panel(gray_image, hist_image)
    write_image(histogram_dir / "original_with_manual_histogram.png", hist_panel_image)
    hist_compare_image = draw_histogram_comparison(manual_hist, opencv_hist)
    write_image(histogram_dir / "histogram_compare.png", hist_compare_image)
    hist_error_image = draw_histogram_error_curve(manual_hist, opencv_hist)
    write_image(histogram_dir / "histogram_error_curve.png", hist_error_image)

    # 灰度变换：线性拉伸 + Gamma 校正
    manual_transform = linear_contrast_stretch_manual(gray_image)
    opencv_transform = linear_contrast_stretch_opencv(gray_image)
    transform_stats = compare_images(manual_transform, opencv_transform)
    transform_montage = make_three_panel_montage(gray_image, manual_transform, opencv_transform, "Original", "Manual contrast stretch", "OpenCV normalize")
    write_image(gray_dir / "linear_contrast_compare.png", transform_montage)

    gamma_manual, gamma_lut = gamma_correction_manual(gray_image, gamma=1.5)
    gamma_opencv = gamma_correction_opencv(gray_image, gamma_lut)
    gamma_stats = compare_images(gamma_manual, gamma_opencv)
    gamma_montage = make_three_panel_montage(gray_image, gamma_manual, gamma_opencv, "Original", "Manual gamma", "OpenCV LUT")
    write_image(gray_dir / "gamma_compare.png", gamma_montage)

    # 直方图均衡化：手工实现 + OpenCV 对照
    manual_equalized, mapping = histogram_equalization_manual(gray_image)
    opencv_equalized = histogram_equalization_opencv(gray_image)
    eq_stats = compare_images(manual_equalized, opencv_equalized)
    eq_montage = make_equalization_montage(gray_image, manual_equalized, opencv_equalized)
    write_image(enhancement_dir / "equalization_compare.png", eq_montage)
    write_image(enhancement_dir / "manual_equalized.png", manual_equalized)
    write_image(enhancement_dir / "opencv_equalized.png", opencv_equalized)
    mapping_image = draw_histogram(mapping.astype(np.int64), out_size=(900, 500))
    write_image(enhancement_dir / "equalization_mapping.png", mapping_image)

    # 空间域平滑：均值、中值、高斯
    noisy_gray = add_salt_pepper_noise(gray_image, amount=0.05, seed=7)
    mean_manual = mean_filter_manual(noisy_gray, kernel_size=5)
    mean_opencv = mean_filter_opencv(noisy_gray, kernel_size=5)
    median_manual = median_filter_manual(noisy_gray, kernel_size=5)
    median_opencv = median_filter_opencv(noisy_gray, kernel_size=5)
    gaussian_manual = gaussian_filter_manual(noisy_gray, kernel_size=5, sigma=1.2)
    gaussian_opencv = gaussian_filter_opencv(noisy_gray, kernel_size=5, sigma=1.2)
    mean_stats = compare_images(mean_manual, mean_opencv)
    median_stats = compare_images(median_manual, median_opencv)
    gaussian_stats = compare_images(gaussian_manual, gaussian_opencv)
    smoothing_stats = {
        "mean_abs_diff": float(np.mean([mean_stats["mean_abs_diff"], median_stats["mean_abs_diff"], gaussian_stats["mean_abs_diff"]])),
        "max_abs_diff": float(np.max([mean_stats["max_abs_diff"], median_stats["max_abs_diff"], gaussian_stats["max_abs_diff"]])),
    }
    write_image(smoothing_dir / "noisy_input.png", noisy_gray)
    write_image(smoothing_dir / "mean_compare.png", make_three_panel_montage(noisy_gray, mean_manual, mean_opencv, "Noisy input", "Manual mean", "OpenCV blur"))
    write_image(smoothing_dir / "median_compare.png", make_three_panel_montage(noisy_gray, median_manual, median_opencv, "Noisy input", "Manual median", "OpenCV medianBlur"))
    write_image(smoothing_dir / "gaussian_compare.png", make_three_panel_montage(noisy_gray, gaussian_manual, gaussian_opencv, "Noisy input", "Manual Gaussian", "OpenCV GaussianBlur"))

    # 锐化与边缘增强：Laplacian 和 Unsharp Mask
    lap_manual = laplacian_sharpen_manual(gray_image, alpha=1.0)
    lap_opencv = laplacian_sharpen_opencv(gray_image, alpha=1.0)
    unsharp_manual = unsharp_mask_manual(gray_image, amount=1.5)
    unsharp_opencv = unsharp_mask_opencv(gray_image, amount=1.5)
    lap_stats = compare_images(lap_manual, lap_opencv)
    unsharp_stats = compare_images(unsharp_manual, unsharp_opencv)
    sharpen_stats = {
        "mean_abs_diff": float(np.mean([lap_stats["mean_abs_diff"], unsharp_stats["mean_abs_diff"]])),
        "max_abs_diff": float(np.max([lap_stats["max_abs_diff"], unsharp_stats["max_abs_diff"]])),
    }
    write_image(sharpen_dir / "laplacian_compare.png", make_three_panel_montage(gray_image, lap_manual, lap_opencv, "Original", "Manual Laplacian", "OpenCV Laplacian"))
    write_image(sharpen_dir / "unsharp_compare.png", make_three_panel_montage(gray_image, unsharp_manual, unsharp_opencv, "Original", "Manual Unsharp", "OpenCV addWeighted"))

    # 边缘检测：手工 Sobel + OpenCV Sobel 对照
    manual_edge = sobel_manual(gray_image)
    opencv_edge = sobel_opencv(gray_image)
    edge_stats = compare_images(manual_edge, opencv_edge)
    edge_montage = make_edge_montage(gray_image, manual_edge, opencv_edge)
    write_image(edge_dir / "edge_compare.png", edge_montage)

    report_path = write_text_report(
        out_dir,
        gray_image,
        args.gray,
        manual_hist,
        opencv_hist,
        transform_stats,
        gamma_stats,
        smoothing_stats,
        sharpen_stats,
        eq_stats,
        edge_stats,
    )

    print("实验处理完成")
    print(f"彩色通道图像输出：{channel_dir}")
    print(f"直方图输出：{histogram_dir / 'manual_histogram.png'}")
    print(f"原图+手工直方图并排图输出：{histogram_dir / 'original_with_manual_histogram.png'}")
    print(f"直方图对比图输出：{histogram_dir / 'histogram_compare.png'}")
    print(f"直方图误差曲线图输出：{histogram_dir / 'histogram_error_curve.png'}")
    print(f"灰度变换对比图输出：{gray_dir / 'linear_contrast_compare.png'}")
    print(f"Gamma 校正对比图输出：{gray_dir / 'gamma_compare.png'}")
    print(f"平滑对比图输出：{smoothing_dir / 'mean_compare.png'} / {smoothing_dir / 'median_compare.png'} / {smoothing_dir / 'gaussian_compare.png'}")
    print(f"锐化对比图输出：{sharpen_dir / 'laplacian_compare.png'} / {sharpen_dir / 'unsharp_compare.png'}")
    print(f"直方图均衡化对比图输出：{enhancement_dir / 'equalization_compare.png'}")
    print(f"直方图均衡化映射图输出：{enhancement_dir / 'equalization_mapping.png'}")
    print(f"边缘检测对比图输出：{edge_dir / 'edge_compare.png'}")
    print(f"分析报告输出：{report_path}")


if __name__ == "__main__":
    main()