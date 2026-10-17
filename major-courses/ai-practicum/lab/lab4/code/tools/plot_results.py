"""Rebuild report PNGs from measured results using SciencePlots and a CJK font."""
import csv
import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter, MaxNLocator
from fontTools.ttLib import TTCollection
import numpy as np
import scienceplots

ROOT=Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'reports/figures'
FONT_SOURCE = Path(str(ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'))
FONT_FILE = FONT_SOURCE
NAMES = {
    'teacher_pytorch_fp32': '教师 / PyTorch FP32',
    'teacher_pytorch_fp16': '教师 / PyTorch FP16',
    'teacher_tensorrt_16': '教师 / TensorRT FP16',
    'teacher_tensorrt_8': '教师 / TensorRT INT8',
    'student_control_pytorch_fp32': '独立训练学生 / FP32',
    'student_distilled_pytorch_fp32': '蒸馏学生 / PyTorch FP32',
    'student_distilled_pytorch_fp16': '蒸馏学生 / PyTorch FP16',
    'student_distilled_tensorrt_16': '蒸馏学生 / TensorRT FP16',
    'student_distilled_tensorrt_8': '蒸馏学生 / TensorRT INT8',
}


def read(path):
    return json.loads(path.read_text())


def configure_style():
    # Matplotlib indexes only the JP face of this installed TTC by default.
    # Extract its already-installed Simplified Chinese face into project cache.
    if not FONT_FILE.exists():
        FONT_FILE.parent.mkdir(parents=True, exist_ok=True)
        collection = TTCollection(FONT_SOURCE, lazy=True)
        face = next(f for f in collection.fonts if f['name'].getDebugName(1) == 'Noto Sans CJK SC')
        face.save(FONT_FILE)
        collection.close()
    font_manager.fontManager.addfont(str(FONT_FILE))
    font_name = font_manager.FontProperties(fname=str(FONT_FILE)).get_name()
    assert font_name == 'Noto Sans CJK SC', font_name
    plt.style.use(['science', 'no-latex', 'bright'])
    # These settings apply to this plotting process only; no matplotlibrc edits.
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [font_name, 'DejaVu Sans'],
        'font.size': 11.5, 'axes.labelsize': 12, 'axes.titlesize': 11,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10.5,
        'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': True,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.edgecolor': '#333333', 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.top': False, 'ytick.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'lines.linewidth': 1.7, 'legend.frameon': False,
        'figure.dpi': 120, 'savefig.dpi': 300,
    })
    warnings.filterwarnings('error', message=r'Glyph .* missing from font')
    return font_name


def generate_figures():
    FIGURES.mkdir(parents=True, exist_ok=True)
    font_name = configure_style()
    plan = read(ROOT / 'run_plan.json')
    sources = [ROOT / 'run_plan.json']
    paths, descriptions = {}, {}

    def save(fig, key, description):
        path = FIGURES / (key + '.png')
        fig.canvas.draw()  # Missing CJK glyphs fail here instead of being hidden.
        fig.savefig(path, dpi=300, metadata={'Description': description,
                    'Software': 'SciencePlots ' + importlib.metadata.version('SciencePlots')})
        plt.close(fig)
        paths[key] = path
        descriptions[key] = description

    def curve_axes(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.minorticks_off()
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#d5d9df', linewidth=0.6, alpha=0.8)
        ax.margins(x=0.025, y=0.12)
        ax.legend(loc='lower right')

    for key, models in [
        ('detector_validation', [('yolo26m_raw', '基础微调'), ('yolo26m_augmented', '预处理与增强')]),
        ('distillation_validation', [('yolo26n_control', '独立训练学生'), ('yolo26n_distilled', '特征蒸馏学生')]),
    ]:
        if key.startswith("detector"):
            continue
        path = ROOT / 'experiments/arithmetic_model_acceleration/runs' / plan['runs']['deployment_comparison'] / 'comparison.json'
    sources.append(path)
    results = read(path)['results']
    medians, p05, p95 = [], [], []
    for result in results:
        samples = np.asarray(result['samples_ms'], dtype=float)
        assert samples.shape == (200,) and np.isfinite(samples).all() and (samples > 0).all()
        lo, median, hi = np.percentile(samples, [5, 50, 95])
        assert np.isclose(median, result['forward_median_ms'])
        assert np.isclose(hi, result['forward_p95_ms'])
        p05.append(lo); medians.append(median); p95.append(hi)
    medians, p05, p95 = [np.array(values) for values in [medians, p05, p95]]
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [cycle[0] if r['label'].startswith('teacher') else
              (cycle[2] if r['label'].startswith('student_control') else cycle[1]) for r in results]
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    positions = np.arange(len(results))
    ax.barh(positions, medians, color=colors, height=0.62, zorder=2)
    ax.errorbar(medians, positions, xerr=[medians - p05, p95 - medians],
                fmt='none', ecolor='#262626', elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)
    for position, median, hi in zip(positions, medians, p95):
        ax.text(hi + max(p95) * 0.025, position, f'{median:.3f}', va='center', fontsize=10.5)
    ax.set_yticks(positions, [NAMES[r['label']] for r in results])
    ax.set_xlim(0, max(p95) * 1.2)
    ax.invert_yaxis()
    ax.set_xlabel('GPU 前向延迟（ms）')
    ax.set_title('条形：中位数；须线：200 次测量的 P5–P95', loc='left', pad=9)
    ax.minorticks_off()
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#d5d9df', linewidth=0.6, alpha=0.8)
    for boundary in [3.5, 4.5]:
        ax.axhline(boundary, color='#d5d9df', linewidth=0.6)
    fig.supxlabel('A100 40GB · batch=1 · 960×960；预热 30 次，测量 200 次', fontsize=9.5)
    save(fig, 'deployment_comparison', 'Bars: median GPU forward latency (ms). Whiskers: empirical P5-P95 of 200 timed forwards after 30 warmups; not a confidence interval across runs. Excludes decode, resize, H2D and NMS.')

    manifest = {
        'styles': ['science', 'no-latex', 'bright'], 'font_family': font_name,
        'font_source': str(FONT_SOURCE), 'font_file': str(FONT_FILE),
        'font_source_sha256': hashlib.sha256(FONT_SOURCE.read_bytes()).hexdigest(),
        'scienceplots': importlib.metadata.version('SciencePlots'), 'matplotlib': matplotlib.__version__,
        'export': {'format': 'PNG', 'dpi': 300, 'background': 'white'},
        'script': str(Path(__file__).resolve()), 'global_matplotlibrc_modified': False,
        'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        'figures': {key: {'path': str(path), 'description': descriptions[key]} for key, path in paths.items()},
        'latency_intervals_ms': [{'label': r['label'], 'p05': float(lo), 'median': float(med), 'p95': float(hi)}
                                 for r, lo, med, hi in zip(results, p05, medians, p95)],
    }
    (FIGURES / 'plot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return paths


if __name__ == '__main__':
    print(json.dumps({k: str(v) for k, v in generate_figures().items()}, ensure_ascii=False, indent=2))
