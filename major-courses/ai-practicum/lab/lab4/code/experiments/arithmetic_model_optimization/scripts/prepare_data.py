"""Freeze a page-level split and derive YOLO labels and rectified OCR crops."""
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[3]
SOURCE = ROOT/'lab3/实验3/train'
OUT = ROOT/'dataset/processed/arithmetic_worksheet'
MANIFESTS = ROOT/'dataset/manifests/arithmetic_worksheet'
SEED = 20261017


def normalize_text(text):
    table = str.maketrans({'×': '*', '∗': '*', '÷': '/', '−': '-', '（': '(', '）': ')', '。': '.'})
    return re.sub(r'\s+', '', text.translate(table))


def corners(box, width, height):
    x, y = box['x'] * width / 100, box['y'] * height / 100
    bw, bh = box['width'] * width / 100, box['height'] * height / 100
    theta = math.radians(box.get('rotation', 0))
    c, s = math.cos(theta), math.sin(theta)
    return [(x + u*c - v*s, y + u*s + v*c) for u, v in [(0,0),(bw,0),(bw,bh),(0,bh)]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS / 'split_manifest.json'
    frozen=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if frozen:
        assert hashlib.sha256((SOURCE/'annotations.json').read_bytes()).hexdigest()==frozen['source_annotations_sha256']
    samples = json.loads((SOURCE / 'annotations.json').read_text())
    rng = random.Random(SEED)
    rng.shuffle(samples)
    sizes = {'train': 407, 'val': 87, 'test': 87}
    splits = {}
    offset = 0
    for split, size in sizes.items():
        splits[split] = sorted(samples[offset:offset+size], key=lambda s: int(Path(s['image']).stem))
        offset += size
    if frozen:
        assert {s['image']:split for split,pages in splits.items() for s in pages}=={r['image']:r['split'] for r in frozen['records']}
    records, issues, counts, sha_groups = [], [], Counter(), {}
    labels_by_split, previews = {}, []
    for split, pages in splits.items():
        lines = []
        for folder in [OUT/'yolo/images'/split, OUT/'yolo/labels'/split, OUT/'ocr/crops'/split]:
            folder.mkdir(parents=True, exist_ok=True)
        if split == 'train':
            (OUT/'yolo/images/train_preprocessed').mkdir(parents=True, exist_ok=True)
            (OUT/'yolo/labels/train_preprocessed').mkdir(parents=True, exist_ok=True)
        for sample in pages:
            src = SOURCE / 'images' / sample['image']
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            sha_groups.setdefault(digest, []).append((split, sample['image']))
            with Image.open(src) as stored:
                im = ImageOps.exif_transpose(stored).convert('RGB')
                corrected = stored.size != im.size
            if im.size != (sample['original_width'], sample['original_height']):
                raise ValueError(f"Annotation/image dimensions disagree after EXIF: {sample['image']}")
            counts['exif_corrected_pages'] += corrected
            stem = Path(sample['image']).stem
            resized = im.copy()
            resized.thumbnail((1600,1600), Image.Resampling.LANCZOS)
            resized.save(OUT/'yolo/images'/split/(stem+'.jpg'), quality=95)
            yolo_lines, boxes_meta = [], []
            for idx, b in enumerate(sample['boxes']):
                pts = corners(b, *im.size)
                xs, ys = zip(*pts)
                x1,y1,x2,y2 = max(0,min(xs)),max(0,min(ys)),min(im.width,max(xs)),min(im.height,max(ys))
                if x2-x1 < 2 or y2-y1 < 2:
                    issues.append({'page':sample['image'],'box':idx,'reason':'degenerate_clipped_box'})
                    continue
                if any(x<0 or x>im.width or y<0 or y>im.height for x,y in pts):
                    counts['clipped_boxes'] += 1
                yolo_lines.append('0 %.8f %.8f %.8f %.8f' % ((x1+x2)/2/im.width,(y1+y2)/2/im.height,(x2-x1)/im.width,(y2-y1)/im.height))
                counts[f'{split}_boxes'] += 1
                counts['rotated_boxes'] += bool(b.get('rotation',0))
                text = normalize_text(b.get('text',''))
                boxes_meta.append({'box_id':idx,'xyxy':[x1,y1,x2,y2],'polygon':pts,'text':text})
                if not text:
                    counts[f'{split}_blank_text_excluded_from_ocr'] += 1
                    continue
                bw = max(2, round(b['width']*im.width/100))
                bh = max(2, round(b['height']*im.height/100))
                # Pillow QUAD order: upper-left, lower-left, lower-right, upper-right.
                quad = [*pts[0],*pts[3],*pts[2],*pts[1]]
                crop = im.transform((bw,bh), Image.Transform.QUAD, quad, Image.Resampling.BICUBIC)
                crop.thumbnail((1200,256),Image.Resampling.LANCZOS)
                rel = f'crops/{split}/{stem}_{idx:03d}.jpg'
                crop.save(OUT/'ocr'/rel,quality=95)
                lines.append(f'{rel}\t{text}')
                counts[f'{split}_ocr_crops'] += 1
                if len(previews)<48 and (len(previews)//8 == list(sizes).index(split) or int(stem)%43==0):
                    previews.append((crop.copy(),text,split))
            label_text='\n'.join(yolo_lines)+'\n'
            (OUT/'yolo/labels'/split/(stem+'.txt')).write_text(label_text)
            if split=='train':
                enhanced=ImageOps.autocontrast(resized,cutoff=0.3,preserve_tone=True)
                arr=np.array(enhanced).astype(np.float32)
                noise=np.random.default_rng(SEED+int(stem)).normal(0,1.5,arr.shape)
                Image.fromarray(np.uint8(np.clip(arr+noise,0,255))).save(OUT/'yolo/images/train_preprocessed'/(stem+'.jpg'),quality=95)
                (OUT/'yolo/labels/train_preprocessed'/(stem+'.txt')).write_text(label_text)
            records.append({'image':sample['image'],'split':split,'sha256':digest,'exif_corrected':corrected,'width':im.width,'height':im.height,'boxes':boxes_meta})
        labels_by_split[split]=len(lines)
        (OUT/'ocr'/f'{split}.txt').write_text('\n'.join(lines)+'\n')
    for digest, members in sha_groups.items():
        if len({m[0] for m in members})>1:
            raise ValueError(f'Duplicate images cross split: {members}')
    for mode,train_dir in [('raw','train'),('augmented','train_preprocessed')]:
        (OUT/'yolo'/f'{mode}.yaml').write_text(f'path: {OUT}/yolo\ntrain: images/{train_dir}\nval: images/val\ntest: images/test\nnames:\n  0: expression\n')
    manifest={'seed':SEED,'source':str(SOURCE),'source_annotations_sha256':hashlib.sha256((SOURCE/'annotations.json').read_bytes()).hexdigest(),'split_sizes':sizes,'normalization':'remove whitespace; ×∗ -> *, ÷ -> /, − -> -, fullwidth parentheses -> ASCII, 。 -> .; no arithmetic correction','records':records}
    if frozen:
        assert json.loads(json.dumps(records))==frozen['records']
    else:
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if not (OUT/'split_manifest.json').exists():
        (OUT/'split_manifest.json').symlink_to(manifest_path)
    summary={'pages':sizes,'counts':dict(counts),'invalid_boxes':issues,'exact_duplicate_groups':[v for v in sha_groups.values() if len(v)>1],'ocr_crops':labels_by_split}
    (OUT/'data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (MANIFESTS/'split_manifest.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['image','split','sha256','exif_corrected','boxes'])
        writer.writerows((r['image'],r['split'],r['sha256'],r['exif_corrected'],len(r['boxes'])) for r in records)
    if not (OUT/'split_manifest.csv').exists():
        (OUT/'split_manifest.csv').symlink_to(MANIFESTS/'split_manifest.csv')
    sheet=Image.new('RGB',(800,70*len(previews)), 'white')
    draw=ImageDraw.Draw(sheet)
    for i,(crop,text,split) in enumerate(previews):
        crop.thumbnail((450,48));sheet.paste(crop,(0,i*70))
        draw.text((460,i*70+10),f'{split}: {text}',fill='black')
    sheet.save(OUT/'ocr_crop_audit.jpg')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
