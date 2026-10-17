"""Evaluate every frozen test crop, preserving predictions and exact error counts."""
import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml
from rapidfuzz.distance import Levenshtein

ROOT=Path(__file__).resolve().parents[3]
REPO=ROOT/'official_repo/PaddleOCR_lab_source'
DATA=ROOT/'dataset/processed/arithmetic_worksheet/ocr'


def normalize(text):
    return re.sub(r'\s+','',text.translate(str.maketrans({'×':'*','∗':'*','÷':'/','−':'-','（':'(', '）':')','。':'.'})))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--split',default='test',choices=['val','test'])
    args=parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'): raise SystemExit('Use bash run.sh to select one visible GPU.')
    sys.path.insert(0,str(REPO))
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from ppocr.data.imaug.rec_img_aug import RecResizeImg
    paddle.set_device('gpu:0')
    run=ROOT/'experiments/arithmetic_model_optimization/runs'/args.run_id
    config=yaml.safe_load((run/'configuration.yaml').read_text())
    decode=build_post_process(config['PostProcess'],config['Global'])
    n=len(decode.character)
    config['Architecture']['Head']['out_channels_list']={'CTCLabelDecode':n,'SARLabelDecode':n+2,'NRTRLabelDecode':n+3}
    model=build_model(config['Architecture'])
    checkpoint=args.checkpoint or ROOT/'checkpoints'/args.run_id/'best_accuracy.pdparams'
    state=pickle.loads(checkpoint.with_suffix('.states').read_bytes())
    selection_path=run/'selected_model.json'
    selection=json.loads(selection_path.read_text())
    selection.update(validation=state['best_model_dict'],selected_epoch=state['epoch'],selected_global_step=state['global_step'])
    selection_path.write_text(json.dumps(selection,indent=2))
    model.set_state_dict(paddle.load(str(checkpoint)))
    model.eval()
    resize=RecResizeImg(image_shape=[3,48,320])
    samples=[line.split('\t',1) for line in (DATA/f'{args.split}.txt').read_text().splitlines()]
    predictions=[]
    elapsed=0.0
    with paddle.no_grad():
        for start in range(0,len(samples),64):
            batch=samples[start:start+64]
            arrays=[]
            for rel,gt in batch:
                image=cv2.imread(str(DATA/rel))
                if image is None: raise RuntimeError(f'Missing or unreadable crop: {rel}')
                arrays.append(resize({'image':image})['image'])
            tensor=paddle.to_tensor(np.stack(arrays))
            paddle.device.cuda.synchronize()
            before=time.perf_counter()
            output=model(tensor)
            decoded=decode(output)
            paddle.device.cuda.synchronize()
            elapsed+=time.perf_counter()-before
            for (rel,gt),(raw_pred,score) in zip(batch,decoded):
                pred=normalize(raw_pred)
                dist=Levenshtein.distance(gt,pred)
                predictions.append({'crop':rel,'ground_truth':gt,'prediction':pred,'raw_prediction':raw_pred,
                                    'confidence':float(score),'exact_match':int(pred==gt),'edit_distance':dist})
            print('OCR_EVAL',len(predictions),'/',len(samples),flush=True)
    assert len(predictions)==len(samples)
    correct=sum(p['exact_match'] for p in predictions)
    chars=sum(len(p['ground_truth']) for p in predictions)
    edits=sum(p['edit_distance'] for p in predictions)
    metrics={'split':args.split,'samples':len(samples),'failures':0,'exact_matches':correct,'accuracy':correct/len(samples),
             'ground_truth_characters':chars,'edit_operations':edits,'character_error_rate':edits/chars,
             'normalized_edit_similarity':float(np.mean([1-p['edit_distance']/max(len(p['ground_truth']),len(p['prediction']),1) for p in predictions])),
             'inference_and_decode_seconds':elapsed,'batch_size':64,'checkpoint':str(checkpoint),
             'checkpoint_bytes':checkpoint.stat().st_size,'parameters':sum(int(np.prod(p.shape)) for p in model.parameters())}
    metrics.update(selected_epoch=state['epoch'],selected_validation=state['best_model_dict'])
    output_dir=run/f'{args.split}_evaluation'
    output_dir.mkdir(exist_ok=True)
    (output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    with (output_dir/'predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]))
        writer.writeheader();writer.writerows(predictions)
    print('OCR_EVALUATION_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':
    main()
