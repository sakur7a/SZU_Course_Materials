"""比较完整重生成后的三角形数据，提示局部候选是否引起远处网格重划分。"""
import argparse
from collections import Counter
import json
import struct
from pathlib import Path

from shapely.ops import substring
import generate_campus as campus

parser=argparse.ArgumentParser()
parser.add_argument('--baseline-assets',type=Path,required=True)
parser.add_argument('--candidate-assets',type=Path,required=True)
parser.add_argument('--padding-m',type=float,default=15)
parser.add_argument('--fail-on-nonlocal',action='store_true')
args=parser.parse_args()
entry=json.loads((campus.DATA/'road-surface-overrides.json').read_text(encoding='utf8'))['overrides'][0]
line=next(g for key,_,g in campus.ground_vehicle_roads if key==entry['source_way'])
segment=substring(line,*entry['range_m'])
x0,z0,x1,z1=segment.buffer(entry['mask_half_width_m']+args.padding_m).bounds


def triangles(path):
    data=path.read_bytes()
    if data[:4]!=b'SZM1':raise ValueError(f'Invalid mesh: {path}')
    count,=struct.unpack_from('<I',data,4)
    if count%3 or len(data)!=8+count*32:raise ValueError(f'Invalid mesh size: {path}')
    payload=data[8:]
    return Counter(payload[i:i+96] for i in range(0,len(payload),96))


nonlocal_count=0
for name in ('campus-road-surface.mesh','campus-road-curbs.mesh','campus-network-sidewalk.mesh'):
    old=triangles(args.baseline_assets/name);new=triangles(args.candidate_assets/name)
    changed=list((old-new).elements())+list((new-old).elements())
    outside=0
    for raw in changed:
        values=struct.unpack('<24f',raw)
        if any(not (x0<=values[i]<=x1 and z0<=values[i+2]<=z1) for i in (0,8,16)):
            outside+=1
    nonlocal_count+=outside
    print(f'{name}: changed triangle records={len(changed)}, outside padded area={outside}')
print('Note: triangle-record changes can reflect re-triangulation or resampling, not necessarily visible surface changes.')
if nonlocal_count and args.fail_on_nonlocal:
    raise SystemExit('Nonlocal mesh-record changes require investigation before adopting this candidate.')
