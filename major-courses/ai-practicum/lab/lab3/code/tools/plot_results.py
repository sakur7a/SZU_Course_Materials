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
        if key.startswith("distillation"):
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.2), layout='constrained')
        for i, (name, label) in enumerate(models):
            experiment = 'arithmetic_model_acceleration' if name.startswith('yolo26n') else 'arithmetic_model_optimization'
            path = ROOT / 'experiments' / experiment / 'runs' / plan['runs'][name] / 'results.csv'
            sources.append(path)
            with path.open() as stream:
                rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            metric = next(k for k in rows[0] if 'mAP50-95' in k)
            ax.plot([r['epoch'] for r in rows], [r[metric] for r in rows],
                    label=f'{label}（{len(rows)}轮）', linestyle=['-', '--'][i],
                    marker=['o', 's'][i], markersize=3, markevery=max(1, len(rows) // 8))
        curve_axes(ax, '训练轮次（epoch）', '验证集 mAP50–95')
        save(fig, key, 'Raw per-epoch validation mAP50-95 from one training run per condition; no smoothing or error bars.')

    fig, ax = plt.subplots(figsize=(6.4, 3.2), layout='constrained')
    for i, (variant, label) in enumerate([('raw', '基础微调'), ('augmented', '数据增强')]):
        path = ROOT / 'experiments/arithmetic_model_optimization/runs' / plan['runs']['ocrv5_' + variant] / 'history.jsonl'
        sources.append(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if r['phase'] == 'val' and 'acc' in r['metrics']]
        assert len(rows) == 15
        ax.plot([r['step'] for r in rows], [r['metrics']['acc'] for r in rows], label=label,
                linestyle=['-', '--'][i], marker=['o', 's'][i], markersize=3.5)
    curve_axes(ax, '优化器更新步数（step）', '验证集整串准确率')
    save(fig, 'ocr_validation', 'Official RecMetric validation accuracy at 15 evaluations per condition; one training run, no error bars.')

    results, p05, medians, p95 = [], [], [], []
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
