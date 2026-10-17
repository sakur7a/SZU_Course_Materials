"""Portable paths, frozen-input checks and serial per-lab execution."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'package_config.json').read_text())
LAB = CONFIG['lab']
EXPERIMENT = ROOT / 'experiments' / ('arithmetic_model_optimization' if LAB == 3 else 'arithmetic_model_acceleration')
PREVIOUS_ROOT = (ROOT / 'cache/initialized_root.txt').read_text().strip() if (ROOT / 'cache/initialized_root.txt').exists() else ''


def read(path):
    return json.loads(path.read_text())


def rebase(value):
    if isinstance(value, str):
        if PREVIOUS_ROOT and value.startswith(PREVIOUS_ROOT + '/') and not value.startswith(str(ROOT) + '/'):
            value = str(ROOT) + value[len(PREVIOUS_ROOT):]
        return value.replace('${LAB_ROOT}', str(ROOT))
    if isinstance(value, dict):
        return {k: rebase(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rebase(v) for v in value]
    return value


def initialize():
    import yaml
    for folder in ['lab3', 'lab4', 'cache/tmp', 'models/pretrained', 'checkpoints']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    # Rebase only explicit portable path fields; frozen IDs and labels stay intact.
    for pattern in ['run_plan.json', 'dataset/manifests/**/*.json', 'experiments/**/selected_model.json']:
        for path in ROOT.glob(pattern):
            value = rebase(read(path))
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    for path in (ROOT / 'experiments').glob('**/configuration.yaml'):
        path.write_text(yaml.safe_dump(rebase(yaml.safe_load(path.read_text())), allow_unicode=True, sort_keys=False))
    data = ROOT / 'dataset/processed/arithmetic_worksheet/yolo'
    if data.exists():
        for mode, train in [('raw', 'train'), ('augmented', 'train_preprocessed')]:
            (data / (mode + '.yaml')).write_text(yaml.safe_dump({'path': str(data), 'train': 'images/' + train,
                'val': 'images/val', 'test': 'images/test', 'names': {0: 'expression'}}, sort_keys=False))
    (ROOT / 'cache/initialized_root.txt').write_text(str(ROOT))


def check():
    manifest = read(ROOT / 'dataset/manifests/arithmetic_worksheet/split_manifest.json')
    assert len(manifest['records']) == 581
    assert manifest['split_sizes'] == {'train': 407, 'val': 87, 'test': 87}
    source = ROOT / 'lab3/实验3/train'
    assert hashlib.sha256((source / 'annotations.json').read_bytes()).hexdigest() == manifest['source_annotations_sha256'], 'Provide the original dataset in lab3/实验3/train first.'
    for record in manifest['records']:
        path = source / 'images' / record['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
        path = ROOT / 'dataset/processed/arithmetic_worksheet/yolo/images' / record['split'] / (Path(record['image']).stem + '.jpg')
        assert path.is_file(), path
        assert (path.parents[2] / 'labels' / record['split'] / (path.stem + '.txt')).is_file()
    for weight in read(ROOT / 'models/pretrained/manifest.json'):
        path = ROOT / 'models/pretrained' / weight['name']
        assert path.stat().st_size == weight['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == weight['sha256'], path
    if LAB == 3:
        for split, count in [('train', 13126), ('val', 2732), ('test', 2853)]:
            data = ROOT / 'dataset/processed/arithmetic_worksheet/ocr'
            rows = (data / (split + '.txt')).read_text().splitlines()
            assert len(rows) == count
            assert all((data / row.split('\t', 1)[0]).is_file() for row in rows)
    print('Frozen data, pretrained weights and portable paths: passed', flush=True)


def prepare():
    initialize()
    spec = importlib.util.spec_from_file_location('prepare_data', ROOT / 'experiments/arithmetic_model_optimization/scripts/prepare_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    check()


def launch():
    logdir = Path(os.environ['LOG_DIR'])
    command = '\n'.join([
        'set -euo pipefail',
        'source ' + shlex.quote(str(ROOT / 'tools/env.sh')),
        'export CUDA_VISIBLE_DEVICES=' + shlex.quote(os.environ['CUDA_VISIBLE_DEVICES']),
        'export RUN_ID=' + shlex.quote(os.environ['RUN_ID']),
        'export PYTHON_BIN=' + shlex.quote(sys.executable),
        'exec ' + shlex.join([sys.executable, str(ROOT / 'tools/portable.py'), os.environ['LAB_MODE']]) +
        ' > ' + shlex.quote(str(logdir / 'launcher.log')) + ' 2>&1',
    ])
    subprocess.run(['tmux', '-L', 'wjz_cloud', 'new-session', '-d', '-s', os.environ['SESSION'],
                    'bash', '-lc', command], check=True)
    (logdir / 'README.md').write_text('Mode: ' + os.environ['LAB_MODE'] + '\nSee launcher.log; managed by tmux socket wjz_cloud.\n')


def execute(mode):
    initialize()
    assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Use bash run.sh so GPU checks and tmux management apply.'
    prefix = os.environ.get('RUN_ID', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    state_path = EXPERIMENT / 'runs' / (prefix + '-orchestrator') / 'state.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {'mode': mode, 'status': 'running', 'completed': []}

    def run(name, script, *args):
        state['current_step'] = name
        state_path.write_text(json.dumps(state, indent=2))
        print('START', name, flush=True)
        subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)
        state['completed'].append(name)
    try:
        if mode == 'smoke':
            run('gpu_smoke', 'tools/smoke.py', '--output', state_path.parent / 'gpu_smoke.json')
        else:
            check()
            plan = read(ROOT / 'run_plan.json')
            original = read(ROOT / 'bundled_run_plan.json')
            train_script = 'experiments/arithmetic_model_optimization/scripts/train_yolo.py'
            if LAB == 3:
                for key in ['yolo26m_raw', 'yolo26m_augmented', 'ocrv5_raw', 'ocrv5_augmented']:
                    rid = prefix + '-' + key
                    plan['runs'][key] = rid
                    if mode == 'eval':
                        old = ROOT / 'experiments/arithmetic_model_optimization/runs' / original['runs'][key]
                        new = ROOT / 'experiments/arithmetic_model_optimization/runs' / rid
                        new.mkdir(parents=True, exist_ok=True)
                        for name in ['selected_model.json', 'configuration.yaml', 'results.csv', 'history.jsonl']:
                            if (old / name).is_file():
                                (new / name).write_bytes((old / name).read_bytes())
                    elif key.startswith('yolo'):
                        run(key, train_script, '--run-id', rid, '--variant', key.removeprefix('yolo26m_'), '--epochs', '50')
                    else:
                        run(key, 'experiments/arithmetic_model_optimization/scripts/train_ocr.py', '--run-id', rid,
                            '--variant', key.removeprefix('ocrv5_'), '--epochs', '15')
                (ROOT / 'run_plan.json').write_text(json.dumps(plan, indent=2))
                initialize()
                run('detection_test', 'experiments/arithmetic_model_optimization/scripts/evaluate_detection.py')
                for variant in ['raw', 'augmented']:
                    key = 'ocrv5_' + variant
                    args = ['--run-id', plan['runs'][key]]
                    if mode == 'eval':
                        args += ['--checkpoint', ROOT / 'checkpoints' / original['runs'][key] / 'best_accuracy.pdparams']
                    run(key + '_test', 'experiments/arithmetic_model_optimization/scripts/evaluate_ocr.py', *args)
            else:
                teacher = ROOT / CONFIG['teacher_checkpoint']
                if mode == 'train':
                    for key in ['yolo26n_control', 'yolo26n_distilled']:
                        rid = prefix + '-' + key
                        plan['runs'][key] = rid
                        args = ['--run-id', rid, '--variant', 'augmented', '--scale', 'n', '--epochs', '50', '--batch', '16']
                        if key.endswith('distilled'):
                            args += ['--teacher', teacher]
                        run(key, train_script, *args)
                def checkpoint(key):
                    selection = ROOT / 'experiments/arithmetic_model_acceleration/runs' / plan['runs'][key] / 'selected_model.json'
                    return rebase(read(selection))['best_checkpoint']
                rid = prefix + '-deployment_comparison'
                plan['runs']['deployment_comparison'] = rid
                (ROOT / 'run_plan.json').write_text(json.dumps(plan, indent=2))
                run('deployment_comparison', 'experiments/arithmetic_model_acceleration/scripts/evaluate_deployment.py',
                    '--run-id', rid, '--teacher', teacher, '--student', checkpoint('yolo26n_distilled'),
                    '--control-student', checkpoint('yolo26n_control'))
            run('plots', 'tools/plot_results.py')
        state.update(status='complete', current_step=None)
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['init', 'check', 'prepare', 'plot', 'launch', 'smoke', 'eval', 'train'])
    mode = parser.parse_args().mode
    if mode == 'init': initialize()
    elif mode == 'check': initialize(); check()
    elif mode == 'prepare': prepare()
    elif mode == 'plot': initialize(); subprocess.run([sys.executable, str(ROOT / 'tools/plot_results.py')], check=True)
    elif mode == 'launch': launch()
    else: execute(mode)
