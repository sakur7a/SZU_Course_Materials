"""Export and compare trained models on the same A100, images and timing scope."""
import argparse
import copy
import csv
import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'dataset/processed/arithmetic_worksheet/yolo'


def benchmark(path, fp16=False, iterations=200):
    torch.cuda.empty_cache()
    torch.manual_seed(20261017)
    backend=AutoBackend(model=path,device=torch.device('cuda:0'),fp16=fp16,end2end=False,verbose=False)
    dtype=torch.float16 if backend.fp16 else torch.float32
    tensor=torch.rand(1,3,960,960,device='cuda',dtype=dtype)
    with torch.inference_mode():
        for _ in range(30): backend(tensor)
        torch.cuda.synchronize()
        times=[]
        for _ in range(iterations):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();backend(tensor);end.record();end.synchronize()
            times.append(start.elapsed_time(end))
    result={'forward_median_ms':float(np.median(times)),'forward_mean_ms':float(np.mean(times)),
            'forward_p95_ms':float(np.percentile(times,95)),'forward_fps':1000/float(np.mean(times)),
            'warmup_iterations':30,'timed_iterations':iterations,'batch':1,'input_shape':[1,3,960,960],
            'torch_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'torch_cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'timing_scope':'GPU forward pass; excludes image decode, resize, H2D transfer and NMS','samples_ms':times}
    del backend,tensor
    gc.collect();torch.cuda.empty_cache()
    return result


def evaluate(path, output, label, fp16=False):
    model=YOLO(path,task='detect')
    metrics=model.val(data=str(DATA/'raw.yaml'),split='test',imgsz=960,batch=1,device=0,workers=4,
                      rect=False,quantize=16 if fp16 else 32,nms=None,
                      project=str(output),name=label,plots=True,save_json=False,save_txt=True,save_conf=True,conf=0.001,iou=0.7,max_det=300)
    assert int(metrics.nt_per_class.sum())==3031, 'Detection GT coverage mismatch.'
    assert int(metrics.nt_per_image.sum())==87, 'Detection page coverage mismatch.'
    result={'label':label,'path':str(path),'artifact_bytes':Path(path).stat().st_size,
            'test_metrics':{k:float(v) for k,v in metrics.results_dict.items()},
            'validation_speed_ms':{k:float(v) for k,v in metrics.speed.items()},'test_pages':87,'test_boxes':3031,'failures':0}
    del model,metrics
    gc.collect();torch.cuda.empty_cache()
    result.update(benchmark(path,fp16))
    (output/f'{label}.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--teacher',required=True)
    parser.add_argument('--student',required=True)
    parser.add_argument('--control-student',required=True)
    args=parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'): raise SystemExit('Use bash run.sh to select one visible GPU.')
    torch.set_num_threads(8)
    output=ROOT/'experiments/arithmetic_model_acceleration/runs'/args.run_id
    output.mkdir(parents=True,exist_ok=True)
    (output/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,name,uuid,memory.used,utilization.gpu','--format=csv'],text=True))
    weights=ROOT/'checkpoints'/args.run_id
    weights.mkdir(parents=True,exist_ok=True)
    # Ultralytics' calibrator defaults to the validation field. Every field of
    # this calibration-only YAML deliberately points to the 407 TRAIN pages.
    calibration=output/'calibration_train_only.yaml'
    calibration.write_text(f'path: {DATA}\ntrain: images/train\nval: images/train\ntest: images/train\nnames:\n  0: expression\n')
    specifications=[]
    parameter_counts={}
    for label,source in [('teacher',args.teacher),('student_control',args.control_student),('student_distilled',args.student)]:
        yolo=YOLO(source)
        if hasattr(yolo.model,'student_model'): yolo.model=yolo.model.student_model
        yolo.model=yolo.model.float().eval()
        parameter_counts[label]=sum(p.numel() for p in yolo.model.parameters())
        destination=weights/f'{label}_fp32.pt'
        torch.save({'model':copy.deepcopy(yolo.model).cpu().float(),'train_args':dict(yolo.model.args) if isinstance(yolo.model.args,dict) else vars(yolo.model.args)},destination)
        specifications.append((label+'_pytorch_fp32',destination,False))
        if label=='student_control': continue
        specifications.append((label+'_pytorch_fp16',destination,True))
        for precision in [16,8]:
            exporter=YOLO(destination)
            exported=Path(exporter.export(format='engine',device=0,imgsz=960,batch=1,quantize=precision,nms=None,
                                          data=str(calibration),workspace=4,dynamic=False,simplify=False,opset=17,verbose=False))
            named=weights/f'{label}_int{precision}.engine' if precision==8 else weights/f'{label}_fp16.engine'
            exported.rename(named)
            onnx=destination.with_suffix('.onnx')
            if onnx.exists(): onnx.rename(weights/f'{label}_export_source_{precision}.onnx')
            specifications.append((label+f'_tensorrt_{precision}',named,precision==16))
            del exporter
            gc.collect();torch.cuda.empty_cache()
    results=[]
    for label,path,half in specifications:
        print('EVALUATING',label,str(path),flush=True)
        results.append(evaluate(path,output,label,half))
    columns=['label','artifact_bytes','forward_median_ms','forward_mean_ms','forward_p95_ms',
             'metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)']
    for destination in [output/'comparison.csv',ROOT/'lab4/deployment_comparison.csv']:
        with destination.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for result in results:
                row={**result,**result['test_metrics']}
                writer.writerow({k:row[k] for k in columns})
    (output/'comparison.json').write_text(json.dumps({'hardware':torch.cuda.get_device_name(0)+', visible GPU '+os.environ['CUDA_VISIBLE_DEVICES'],
        'calibration':'407 training pages only; no validation/test calibration','input_shape':[1,3,960,960],
        'parameter_counts':parameter_counts,'results':results},indent=2))
    print('DEPLOYMENT_EVALUATION_COMPLETE',str(output/'comparison.json'),flush=True)


if __name__=='__main__':
    main()
