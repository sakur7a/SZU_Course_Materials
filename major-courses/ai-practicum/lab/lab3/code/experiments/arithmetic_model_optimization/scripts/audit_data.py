"""Reproduce the data coverage, vocabulary and official YOLO deduplication audit."""
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'dataset/processed/arithmetic_worksheet'


def main():
    path=DATA/'data_audit.json'
    audit=json.loads(path.read_text())
    counts,duplicates,lookup={},{},{}
    vocab=set((ROOT/'official_repo/PaddleOCR_lab_source/ppocr/utils/dict/ppocrv5_dict.txt').read_text().splitlines())|{' '}
    for split in ['train','val','test']:
        total=0;details=[]
        for p in sorted((DATA/'yolo/labels'/split).glob('*.txt')):
            assert (DATA/'yolo/images'/split/(p.stem+'.jpg')).is_file()
            rows=np.array([[float(v) for v in line.split()] for line in p.read_text().splitlines() if line.strip()],dtype=np.float32)
            assert rows.shape[1]==5 and np.isfinite(rows).all()
            unique=len(np.unique(rows,axis=0));total+=unique
            if unique<len(rows): details.append({'image':p.stem+'.jpg','original_boxes':len(rows),'unique_boxes':unique})
        counts[split]=total;duplicates[split]=details
        pairs=[s.split('\t',1) for s in (DATA/'ocr'/f'{split}.txt').read_text().splitlines()]
        assert all((DATA/'ocr'/rel).is_file() for rel,_ in pairs)
        assert max(len(t) for _,t in pairs)<=25
        assert not {c for _,t in pairs for c in t if c not in vocab}
        for rel,text in pairs:
            stem=Path(rel).name.split('_')[0]
            lookup.setdefault(stem,(rel,text))
    audit.update(yolo_unique_boxes=counts,yolo_duplicate_box_audit=duplicates,
        yolo_duplicate_policy='Ultralytics official loader removes identical float32 class/xywh label rows within a page; original labels and manifest preserved.',
        source_pairing_report=json.loads((ROOT/'lab3/实验3/pairing_report.json').read_text()),
        validation_checks={'ocr_vocabulary_coverage':'all characters supported','max_label_length_within_25':True,'all_referenced_crops_exist':True,'all_referenced_pages_exist':True})
    path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    ids=['382','493','564','44','49','50']
    sheet=Image.new('RGB',(1000,100*len(ids)),'white');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype(str(ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'),24)
    for i,stem in enumerate(ids):
        rel,text=lookup[stem]
        with Image.open(DATA/'ocr'/rel) as im:
            im.thumbnail((480,90));sheet.paste(im,(0,i*100))
        draw.text((500,i*100+25),f'{stem}: {text}',font=font,fill='black')
    sheet.save(DATA/'preprocessing_examples.png')
    print(json.dumps({'yolo_unique_boxes':counts,'duplicate_pages':duplicates},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
