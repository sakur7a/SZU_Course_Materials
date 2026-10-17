import gc
import json
import os
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[3]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')
    plan=json.loads((ROOT/'run_plan.json').read_text())
    results={}
    for variant in ['raw','augmented']:
        rid=plan['runs']['yolo26m_'+variant]
        run=ROOT/'experiments/arithmetic_model_optimization/runs'/rid
        selection=json.loads((run/'selected_model.json').read_text())
        model=YOLO(selection['best_checkpoint'])
        metric=model.val(data=str(ROOT/'dataset/processed/arithmetic_worksheet/yolo/raw.yaml'),split='test',
                         imgsz=960,batch=1,device=0,workers=4,rect=False,quantize=32,nms=None,
                         project=str(run),name='test_evaluation',plots=True,save_txt=True,save_conf=True,conf=0.001,iou=0.7,max_det=300)
        assert int(metric.nt_per_class.sum())==3031, 'Detection GT coverage mismatch.'
        assert int(metric.nt_per_image.sum())==87, 'Detection page coverage mismatch.'
        results[variant]={'run_id':rid,'checkpoint':selection['best_checkpoint'],'parameters':selection['parameters'],
                          'test_pages':87,'test_boxes':3031,'missing_pages':0,'failed_pages':0,
                          'metrics':{k:float(v) for k,v in metric.results_dict.items()},'speed_ms':metric.speed}
        (run/'test_metrics.json').write_text(json.dumps(results[variant],indent=2))
        del metric,model
        gc.collect();torch.cuda.empty_cache()
    (ROOT/'lab3/detection_comparison.json').write_text(json.dumps(results,indent=2))
    print('DETECTION_COMPARISON_COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__': main()
