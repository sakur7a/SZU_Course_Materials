"""核验禁用保持基线、候选路段连续且风险检查有效。"""
import copy
import json
import math

from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

import campus_road_overrides as overrides
import generate_campus as campus

config=json.loads((campus.DATA/'road-surface-overrides.json').read_text(encoding='utf8'))
# 无论磁盘配置当前是否启用，都从独立的关闭配置验证原始基线。
disabled_config=copy.deepcopy(config)
for item in disabled_config['overrides']:item['enabled']=False
# 从道路中心线独立重建原始车行面，因此在候选开启与关闭的副本均可运行。
base=unary_union([g.buffer(campus.road_width(key,t)*.5,cap_style='round',join_style='round')
                  for key,t,g in campus.ground_vehicle_roads])
def apply(config,pedestrian=campus.pedestrian_conflicts):
    return overrides.apply_overrides(base,campus.ground_vehicle_roads,campus.road_junctions,
        campus.road_obstacles,campus.DATA,config,pedestrian=pedestrian,road_width=campus.road_width)
disabled,report=apply(disabled_config)
if disabled.wkb!=base.wkb or any(item['status']!='disabled' for item in report):
    raise ValueError('Disabled override changed the baseline')
enabled=copy.deepcopy(disabled_config);entry=enabled['overrides'][0];entry['enabled']=True
updated,report=apply(enabled)
line=next(g for key,_,g in campus.ground_vehicle_roads if key==entry['source_way'])
segment=substring(line,*entry['range_m'])
mask=segment.buffer(entry['mask_half_width_m'],cap_style='flat',join_style='round')
collar=unary_union([Point(segment.coords[0]).buffer(entry['transition_m']),
                    Point(segment.coords[-1]).buffer(entry['transition_m'])])
diff=base.symmetric_difference(updated)
if diff.is_empty or diff.difference(mask).area>1e-7 or diff.intersection(collar).area>1e-7:
    raise ValueError('Road change escaped its local mask or end collars')
for d in range(101):
    if not updated.buffer(.001).covers(segment.interpolate(segment.length*d/100)):
        raise ValueError('Road centerline is disconnected')
local=updated.intersection(segment.buffer(6,cap_style='flat'))
widths=[]
for i in range(1,math.floor(segment.length*2)):
    d=i*.5;p=segment.interpolate(d)
    a=segment.interpolate(max(0,d-.2));b=segment.interpolate(min(segment.length,d+.2))
    dx,dz=b.x-a.x,b.y-a.y;length=math.hypot(dx,dz);nx,nz=-dz/length,dx/length
    section=LineString(((p.x-nx*5,p.y-nz*5),(p.x+nx*5,p.y+nz*5)))
    widths.append(section.intersection(local).length)
if max(abs(a-b) for a,b in zip(widths,widths[1:]))>.20:
    raise ValueError('Road width changes too abruptly between neighboring sections')
bad=copy.deepcopy(enabled);bad['overrides'][0]['range_m']=[30,50]
try:
    apply(bad)
except ValueError as error:
    if 'junction' not in str(error):raise
else:raise ValueError('Junction-crossing override was not rejected')
try:
    apply(enabled,pedestrian=segment.interpolate(.5,normalized=True))
except ValueError as error:
    if 'pedestrian' not in str(error):raise
else:raise ValueError('Pedestrian-crossing override was not rejected')
bad=copy.deepcopy(enabled);bad['overrides'][0]['max_removed_fraction']=.01
try:
    apply(bad)
except ValueError as error:
    if 'excessive removed area' not in str(error):raise
else:raise ValueError('Excessive removal was not rejected')
bad=copy.deepcopy(enabled);bad['overrides'][0]['candidate_path']='../../outside.geojson'
try:
    apply(bad)
except ValueError as error:
    if 'escapes data directory' not in str(error):raise
else:raise ValueError('External candidate path was not rejected')
bad=copy.deepcopy(enabled);bad['overrides'].append(copy.deepcopy(bad['overrides'][0]));bad['overrides'][1]['id']='duplicate-area'
try:
    apply(bad)
except ValueError as error:
    if 'masks overlap' not in str(error):raise
else:raise ValueError('Overlapping road overrides were not rejected')
bad=copy.deepcopy(enabled);bad['overrides'][0]['confidence']='measured'
try:
    apply(bad)
except ValueError as error:
    if 'provenance' not in str(error):raise
else:raise ValueError('Mismatched candidate confidence was not rejected')
print(f"Road override verified: disabled baseline unchanged; local candidate changes {diff.area:.3f} m²; centerline and end collars intact; unsafe cases rejected.")
