"""Fine-tune the official PP-OCRv5 recognizer without modifying upstream code."""
import argparse
import json
import logging
import os
import re
import sys
import subprocess
from pathlib import Path

import swanlab
import yaml

ROOT=Path(__file__).resolve().parents[3]
REPO=ROOT/'official_repo/PaddleOCR_lab_source'
DATA=ROOT/'dataset/processed/arithmetic_worksheet/ocr'


class SwanMetrics(logging.Handler):
    def __init__(self, run_dir, eval_interval):
        super().__init__()
        self.run_dir=run_dir
        self.step=0
        self.best={}
        self.eval_interval=eval_interval

    def emit(self, record):
        message=record.getMessage()
        is_train='global_step:' in message and 'loss:' in message
        is_validation='cur metric' in message
        if not (is_train or is_validation): return
        metrics={k:float(v) for k,v in re.findall(r'([A-Za-z_][A-Za-z0-9_/]*):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',message)}
        self.step=int(metrics.get('global_step',self.step))
        if is_validation:
            # Official validation lines omit global_step. Recover the exact
            # scheduled evaluation step, rather than the last print interval.
            self.step=max(1,(self.step+self.eval_interval-1)//self.eval_interval)*self.eval_interval
        prefix='train' if is_train else 'val'
        swanlab.log({f'{prefix}/{k}':v for k,v in metrics.items()},step=self.step)
        with (self.run_dir/'history.jsonl').open('a') as f:
            f.write(json.dumps({'step':self.step,'phase':prefix,'metrics':metrics})+'\n')
        if is_validation and metrics.get('acc',-1)>self.best.get('acc',-1):
            self.best=metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--variant',choices=['raw','augmented'],required=True)
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--batch',type=int,default=96)
    args=parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('Run with bash run.sh.')
    run_dir=ROOT/'experiments/arithmetic_model_optimization/runs'/args.run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,name,uuid,memory.used,utilization.gpu','--format=csv'],text=True))
    checkpoints=ROOT/'checkpoints'/args.run_id
    checkpoints.mkdir(parents=True,exist_ok=True)
    with (REPO/'configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml').open() as f:
        config=yaml.safe_load(f)
    config['Global'].update(epoch_num=args.epochs,save_model_dir=str(checkpoints),save_epoch_step=9999,
                            pretrained_model=str(ROOT/'models/pretrained/PP-OCRv5_server_rec_pretrained'),
                            checkpoints=None,eval_batch_step=[0,max(1,13126//args.batch)],print_batch_step=10,
                            distributed=False,use_gpu=True,use_visualdl=False,cal_metric_during_train=True,
                            character_dict_path=str(REPO/'ppocr/utils/dict/ppocrv5_dict.txt'),
                            save_res_path=str(run_dir/'predictions.txt'),seed=20261017,use_amp=True,amp_level='O1',scale_loss=1024.0,use_dynamic_loss_scaling=True)
    config['Optimizer']['lr'].update(learning_rate=0.0001,warmup_epoch=1)
    transform=[{'DecodeImage':{'img_mode':'BGR','channel_first':False}}]
    if args.variant=='augmented': transform.append({'RecAug':{'tia_prob':0.0,'crop_prob':0.0,'reverse_prob':0.1,'noise_prob':0.2,'jitter_prob':0.1,'blur_prob':0.1,'hsv_aug_prob':0.2}})
    encode=[{'MultiLabelEncode':{'gtc_encode':'NRTRLabelEncode'}},
            {'RecResizeImg':{'image_shape':[3,48,320]}},
            {'KeepKeys':{'keep_keys':['image','label_ctc','label_gtc','length','valid_ratio']}}]
    for split,section in [('train','Train'),('val','Eval')]:
        config[section]={'dataset':{'name':'SimpleDataSet','data_dir':str(DATA),
                        'label_file_list':[str(DATA/f'{split}.txt')],
                        'transforms':(transform if split=='train' else [{'DecodeImage':{'img_mode':'BGR','channel_first':False}}])+encode},
                        'loader':{'shuffle':split=='train','drop_last':split=='train',
                                  'batch_size_per_card':args.batch,'num_workers':8 if split=='train' else 4,
                                  'use_shared_memory':False}}
    yaml_path=run_dir/'configuration.yaml'
    yaml_path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    swanlab.init(project='arithmetic-lab3-lab4',experiment_name=args.run_id,logdir=str(run_dir/'swanlab'),mode='local',config=vars(args))
    sys.path.insert(0,str(REPO))
    os.chdir(REPO)
    sys.argv=['train.py','-c',str(yaml_path)]
    from tools import program
    from tools.train import main as train_main
    config,device,logger,writer=program.preprocess(is_train=True)
    handler=SwanMetrics(run_dir,max(1,13126//args.batch))
    logger.addHandler(handler)
    from ppocr.utils.utility import set_seed
    set_seed(20261017)
    train_main(config,device,logger,writer,20261017)
    if not (checkpoints/'best_accuracy.pdparams').is_file() or 'acc' not in handler.best:
        raise RuntimeError('Training did not produce a measured validation-selected OCR checkpoint.')
    result={'checkpoint':str(checkpoints/'best_accuracy.pdparams'),'validation':handler.best,
            'model':'PP-OCRv5_server_rec','variant':args.variant}
    (run_dir/'selected_model.json').write_text(json.dumps(result,indent=2))
    swanlab.finish()
    print('TRAINING_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
