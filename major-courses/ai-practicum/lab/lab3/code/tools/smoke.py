"""One-image GPU inference from bundled checkpoints; each backend is isolated."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['torch', 'paddle'])
    args = parser.parse_args()
    config = json.loads((ROOT / 'package_config.json').read_text())
    if args.backend is None:
        results = []
        for backend in ['torch'] + (['paddle'] if config['lab'] == 3 else []):
            path = args.output.with_name(backend + '_smoke.json')
            subprocess.run([sys.executable, __file__, '--output', str(path), '--backend', backend], check=True)
            results.append(json.loads(path.read_text()))
        args.output.write_text(json.dumps({'status': 'passed', 'backends': results}, ensure_ascii=False, indent=2))
        return
    if args.backend == 'torch':
        import torch
        from ultralytics import YOLO
        assert torch.cuda.is_available()
        torch.set_num_threads(8)
        image = ROOT / 'dataset/processed/arithmetic_worksheet/yolo/images/test'
        image = sorted(image.glob('*.jpg'))[0]
        models = [config['teacher_checkpoint']]
        if config['lab'] == 4:
            models += [config['student_checkpoint'], config['control_checkpoint']]
        results = []
        for path in models:
            model = YOLO(ROOT / path)
            output = model.predict(str(image), imgsz=960, device=0, verbose=False, save=False)
            assert len(output) == 1 and output[0].boxes is not None
            assert torch.isfinite(output[0].boxes.data).all()
            results.append({'checkpoint': path, 'detections': len(output[0].boxes)})
            del model, output
            torch.cuda.empty_cache()
        result = {'backend': 'torch', 'version': torch.__version__, 'device': torch.cuda.get_device_name(0),
                  'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'models': results}
    else:
        import cv2
        import numpy as np
        import paddle
        import yaml
        sys.path.insert(0, str(ROOT / 'official_repo/PaddleOCR_lab_source'))
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from ppocr.data.imaug.rec_img_aug import RecResizeImg
        plan = json.loads((ROOT / 'bundled_run_plan.json').read_text())
        rid = plan['runs']['ocrv5_augmented']
        cfg = yaml.safe_load((ROOT / 'experiments/arithmetic_model_optimization/runs' / rid / 'configuration.yaml').read_text())
        decode = build_post_process(cfg['PostProcess'], cfg['Global'])
        n = len(decode.character)
        cfg['Architecture']['Head']['out_channels_list'] = {'CTCLabelDecode': n, 'SARLabelDecode': n+2, 'NRTRLabelDecode': n+3}
        paddle.set_device('gpu:0')
        model = build_model(cfg['Architecture'])
        model.set_state_dict(paddle.load(str(ROOT / 'checkpoints' / rid / 'best_accuracy.pdparams')))
        model.eval()
        data = ROOT / 'dataset/processed/arithmetic_worksheet/ocr'
        rel = (data / 'test.txt').read_text().splitlines()[0].split('\t')[0]
        image = cv2.imread(str(data / rel))
        assert image is not None
        image = RecResizeImg(image_shape=[3,48,320])({'image': image})['image']
        with paddle.no_grad():
            output = model(paddle.to_tensor(np.stack([image])))
            text, score = decode(output)[0]
        result = {'backend': 'paddle', 'version': paddle.__version__, 'device': paddle.get_device(),
                  'crop': rel, 'prediction': text, 'confidence': float(score)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
