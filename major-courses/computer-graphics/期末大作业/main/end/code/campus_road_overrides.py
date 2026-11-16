"""可审查、默认关闭的局部车行面候选；不覆盖 OSM 原始线与标注。"""
import json
import math
from pathlib import Path

from shapely.geometry import Point, Polygon, shape
from shapely.ops import substring, unary_union


def apply_overrides(base, roads, junctions, obstacles, data, config=None, pedestrian=None, road_width=None):
    root=Path(data).resolve()
    config=config if config is not None else json.loads((root/'road-surface-overrides.json').read_text(encoding='utf8'))
    result=base
    report=[]
    accepted_masks=[]
    for item in config.get('overrides',[]):
        if not item.get('enabled',False):
            report.append({'id':item.get('id'),'status':'disabled'})
            continue
        ident=item.get('id')
        if not ident or not item.get('source_photo') or item.get('confidence') not in ('illustrative','photo_estimate','measured'):
            raise ValueError('Road override requires an id, source and confidence')
        matches=[(t,g) for key,t,g in roads if key==item['source_way']]
        if len(matches)!=1 or matches[0][1].geom_type!='LineString' or road_width is None:
            raise ValueError(f'{ident}: source way missing or ambiguous')
        tags,line=matches[0]
        start,end=item['range_m']
        width=item['mask_half_width_m'];transition=item['transition_m'];inset=item['max_inset_m']
        if not (0<=start<end<=line.length and 0<width<=20 and transition>0 and 0<inset<=1.0 and
                0<=item['max_removed_fraction']<=.20 and 0<=item['max_added_area_m2']<=50):
            raise ValueError(f'{ident}: invalid range or safety margin')
        segment=substring(line,start,end)
        mask=segment.buffer(width,cap_style='flat',join_style='round')
        if any(mask.intersects(previous) for previous in accepted_masks):
            raise ValueError(f'{ident}: local masks overlap')
        if mask.intersects(junctions):raise ValueError(f'{ident}: mask touches a mapped road junction')
        if pedestrian is not None and mask.intersects(pedestrian):
            raise ValueError(f'{ident}: mask touches a mapped pedestrian path or crossing')
        if any(g.intersection(mask).length>1e-6 for key,_,g in roads if key!=item['source_way']):
            raise ValueError(f'{ident}: another road crosses the local mask')
        location=(root/item['candidate_path']).resolve()
        if not location.is_relative_to(root):raise ValueError(f'{ident}: candidate path escapes data directory')
        feature=json.loads(location.read_text(encoding='utf8'))
        props=feature.get('properties',{})
        if (feature.get('type')!='Feature' or props.get('source_way')!=item['source_way'] or
                props.get('confidence')!=item['confidence'] or not props.get('source') or not props.get('license')):
            raise ValueError(f'{ident}: candidate provenance does not match source way')
        candidate=shape(feature['geometry'])
        if candidate.geom_type not in ('Polygon','MultiPolygon') or not candidate.is_valid or candidate.is_empty:
            raise ValueError(f'{ident}: invalid candidate geometry')
        # 沿中心线渐变保护旧路核心：两端各保留过渡带，再平滑达到最大内缩。
        half_width=road_width(item['source_way'],tags)*.5
        if not inset<half_width<width:
            raise ValueError(f'{ident}: road width incompatible with safety mask')
        samples=max(2,math.ceil(segment.length*2));left=[];right=[]
        for i in range(samples+1):
            d=segment.length*i/samples
            p=line.interpolate(start+d)
            a=line.interpolate(max(0,start+d-.25));b=line.interpolate(min(line.length,start+d+.25))
            dx,dz=b.x-a.x,b.y-a.y;length=math.hypot(dx,dz)
            if length<1e-6:raise ValueError(f'{ident}: degenerate road tangent')
            u=max(0.0,min(1.0,(min(d,segment.length-d)-transition)/transition))
            blend=u*u*(3-2*u)
            radius=half_width-inset*blend;nx,nz=-dz/length,dx/length
            left.append((p.x+nx*radius,p.y+nz*radius))
            right.append((p.x-nx*radius,p.y-nz*radius))
        protected=Polygon(left+right[::-1])
        if not protected.is_valid:raise ValueError(f'{ident}: invalid transition geometry')
        protected=protected.intersection(result).intersection(mask)
        proposed=result.difference(mask).union(candidate.intersection(mask)).union(protected)
        removed=result.difference(proposed);added=proposed.difference(result)
        base_area=result.intersection(mask).area
        if base_area<=1e-6 or removed.area/base_area>item['max_removed_fraction']:
            raise ValueError(f'{ident}: excessive removed area')
        if added.area>item['max_added_area_m2'] or added.intersection(obstacles).area>1e-5:
            raise ValueError(f'{ident}: candidate adds excessive area or crosses an obstacle')
        before_coverage=result.buffer(.001);after_coverage=proposed.buffer(.001)
        for i in range(max(2,math.ceil(segment.length)+1)):
            p=segment.interpolate(segment.length*i/max(1,math.ceil(segment.length)))
            if before_coverage.covers(p) and not after_coverage.covers(p):
                raise ValueError(f'{ident}: candidate disconnects the road centerline')
        if not proposed.is_valid:raise ValueError(f'{ident}: resulting road geometry invalid')
        result=proposed
        accepted_masks.append(mask)
        report.append({'id':ident,'status':'enabled','removed_m2':round(removed.area,3),
                       'added_m2':round(added.area,3),'source_way':item['source_way'],
                       'confidence':item['confidence']})
    return result,report
