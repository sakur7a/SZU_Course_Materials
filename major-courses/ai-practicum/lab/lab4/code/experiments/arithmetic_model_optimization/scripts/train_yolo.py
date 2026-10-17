"""Controlled YOLO26 fine-tuning with local SwanLab and native feature distillation."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import swanlab
import torch
from ultralytics import YOLO, settings

ROOT=Path(__file__).resolve().parents[3]
DATA = ROOT/'dataset/processed/arithmetic_worksheet/yolo'
PRETRAINED = ROOT/'models/pretrained'
WEIGHTS = ROOT/'checkpoints'
EXPERIMENTS = ROOT/'experiments'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--variant',choices=['raw','augmented'],required=True)
    parser.add_argument('--scale',default='m',choices=['m','n'])
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--teacher')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--imgsz',type=int,default=960)
    args=parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('Run with bash run.sh.')
    experiment='arithmetic_model_acceleration' if args.scale=='n' else 'arithmetic_model_optimization'
    run_dir=EXPERIMENTS/experiment/'runs'/args.run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,name,uuid,memory.used,utilization.gpu','--format=csv'],text=True))
    weight_dir=WEIGHTS/args.run_id/'weights'
    weight_dir.mkdir(parents=True,exist_ok=True)
    link=run_dir/'weights'
    if not link.exists(): link.symlink_to(weight_dir,target_is_directory=True)
    settings.update({k:False for k in ['wandb','mlflow','clearml','comet','dvc','neptune','raytune','tensorboard','sync','swanlab'] if k in settings})
    torch.set_num_threads(8)
    swanlab.init(project='arithmetic-lab3-lab4',experiment_name=args.run_id,logdir=str(run_dir/'swanlab'),mode='local',config=vars(args))
    model=YOLO(PRETRAINED/f'yolo26{args.scale}.pt')
    (run_dir/'configuration.json').write_text(json.dumps(vars(args),indent=2))

    def log_epoch(trainer):
        values={str(k):float(v) for k,v in trainer.metrics.items()}
        values.update({str(k):float(v) for k,v in trainer.label_loss_items(trainer.tloss,prefix='train').items()})
        values.update({str(k):float(v) for k,v in trainer.lr.items()})
        swanlab.log(values,step=trainer.epoch+1)
    model.add_callback('on_fit_epoch_end',log_epoch)
    aug=args.variant=='augmented'
    config=dict(data=str(DATA/f'{args.variant}.yaml'),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=0,
                workers=8,seed=20261017,deterministic=True,optimizer='AdamW',lr0=0.001,lrf=0.05,
                weight_decay=0.0005,warmup_epochs=3,patience=0 if args.scale=='n' else 15,amp=True,
                project=str(run_dir.parent),name=args.run_id,exist_ok=True,
                hsv_h=0.0,hsv_s=0.1 if aug else 0.0,hsv_v=0.2 if aug else 0.0,
                degrees=3.0 if aug else 0.0,translate=0.04 if aug else 0.0,scale=0.15 if aug else 0.0,
                shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.0,mosaic=0.0,mixup=0.0,copy_paste=0.0,
                close_mosaic=0,plots=True,save=True,save_period=-1,cache=False,augmentations=[])
    if args.teacher:
        config['distill_model']=args.teacher
        config['dis']=6.0
    model.train(**config)
    best=weight_dir/'best.pt'
    selected=YOLO(best)
    training_checkpoint=best
    if args.teacher and hasattr(selected.model,'student_model'):
        from ultralytics.nn.distill_model import DistillationModel
        for module in selected.model.student_model.modules():
            DistillationModel._clear_feature_hooks(module)
        selected.model=selected.model.student_model
        best=weight_dir/'deployed_student.pt'
        selected.save(best)
        selected=YOLO(best)
    validation=selected.val(data=str(DATA/'raw.yaml'),split='val',imgsz=args.imgsz,batch=args.batch,device=0,
                            workers=4,project=str(run_dir),name='selected_validation',plots=True)
    result={'best_checkpoint':str(best),'training_checkpoint':str(training_checkpoint),'validation':{k:float(v) for k,v in validation.results_dict.items()},'parameters':sum(p.numel() for p in selected.model.parameters())}
    (run_dir/'selected_model.json').write_text(json.dumps(result,indent=2))
    swanlab.finish()
    print('TRAINING_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
