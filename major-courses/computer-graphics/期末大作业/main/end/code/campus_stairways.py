"""根据 OSM 台阶线和现有地形生成独立的地面台阶模型。

大部分线没有踏步数量或实测高程：只处理纵向基本单调、横坡较小且未被其他
专门建模场景占用的路段，宽度、阶数及具体构造仍是近似。桥梁和已建台阶另管。
"""
import json
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep


def lines(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type in ('LineString','LinearRing'):
        yield geometry
    elif hasattr(geometry,'geoms'):
        for part in geometry.geoms:
            yield from lines(part)


def make_stairways(*, campus, roadways, vehicle_surface, buildings, waters, pitches,
                   reserved, height_at, mesh_type, assets, data):
    """生成可步行的独立踏面、立面与侧边，返回线位及估计台阶数。"""
    # 以下地图线已有专门构造；不能叠加第二套台阶。
    dedicated={'way/232602597','way/232598299','way/237891957',
               'way/231885484','way/1282922807'}
    blocked=unary_union([g.buffer(.20) for _,_,g in buildings]
                        +[g.buffer(.35) for _,_,g in waters]
                        +[g.buffer(.50) for _,_,g in pitches]
                        +[reserved,vehicle_surface.buffer(.15)])
    allowed=campus.difference(blocked)
    prepared_allowed=prep(allowed)
    edge_allowed=prep(allowed.buffer(.02))
    mesh=mesh_type()
    profiles=[]

    def section(line,distance):
        p=line.interpolate(distance)
        a=line.interpolate(max(0,distance-.18))
        b=line.interpolate(min(line.length,distance+.18))
        dx,dz=b.x-a.x,b.y-a.y
        length=math.hypot(dx,dz)
        if length<.001:
            raise ValueError('Degenerate stair centerline')
        tangent=np.asarray((dx/length,dz/length))
        return np.asarray((p.x,p.y)),np.asarray((-tangent[1],tangent[0])),tangent

    for key,t,geometry in roadways:
        if (key in dedicated or t.get('highway')!='steps' or t.get('bridge') not in (None,'no')
                or t.get('tunnel') not in (None,'no','building_passage') or t.get('layer','0')!='0'
                or not geometry.intersects(campus)):
            continue
        for original in lines(geometry.intersection(campus).difference(blocked)):
            if original.length<4.0:
                continue
            samples=[]
            for dist in np.linspace(0,original.length,31):
                center,normal,_=section(original,float(dist))
                samples.append([height_at(*(center+normal*s)) for s in (-1,0,1)])
            difference=samples[-1][1]-samples[0][1]
            longitudinal=np.diff([q[1] for q in samples])
            max_grade=max(abs(longitudinal))/(original.length/30)
            variation=sum(abs(longitudinal))/max(abs(difference),.001)
            crossfall=max(max(q)-min(q) for q in samples)
            # 极陡、坡向反复或横坡过大通常意味着粗 DEM 不足以决定踏步，保留原步道。
            if abs(difference)<.5 or max_grade>.75 or variation>1.3 or crossfall>.35:
                continue
            line=original if difference>0 else LineString(list(original.coords)[::-1])
            start,end=section(line,0)[0],section(line,line.length)[0]
            rise=height_at(*end)-height_at(*start)
            # 先根据总高差估计级数，再限制局部较陡路段的单级高差。
            # 缓坡采用较深的踏面或平台，避免生成大量仅有几厘米高的假台阶。
            count=max(2,math.ceil(rise/.17),math.ceil(max_grade*line.length/.23),
                      math.ceil(line.length/1.6))
            if count>180:
                continue
            segments=[]
            for i in range(count):
                d0,d1=line.length*i/count,line.length*(i+1)/count
                p0,n0,_=section(line,d0)
                p1,n1,tangent=section(line,d1)
                corners=(p0-n0,p1-n1,p1+n1,p0+n0)
                tread=Polygon(corners)
                if not prepared_allowed.covers(tread):
                    tread=tread.intersection(allowed)
                if tread.is_empty or tread.area<.02:
                    continue
                # 踏面位于两侧和前后地形的最高点上方，避免坡面穿过踏步。
                surface=max(height_at(*q) for q in corners)+.065
                segments.append((p0,p1,n0,n1,tangent,tread,surface,corners))
            if len(segments)!=count:
                continue
            levels=[s[6] for s in segments]
            if any(b-a<-.08 or b-a>.28 for a,b in zip(levels,levels[1:])):
                continue
            previous=None
            for p0,p1,n0,n1,tangent,tread,y,corners in segments:
                mesh.horizontal(tread,y)
                # 沿台阶边缘盖住踏面与坡地之间的空隙；遇道路或障碍时不绘制侧壁。
                for side in (-1,1):
                    a,b=p0+n0*side,p1+n1*side
                    edge=LineString((a,b))
                    if not edge_allowed.covers(edge):
                        continue
                    low_a,low_b=height_at(*a)-.02,height_at(*b)-.02
                    outward=(n0[0]*side,0,n0[1]*side)
                    mesh.quad((a[0],low_a,a[1]),(b[0],low_b,b[1]),
                              (b[0],y,b[1]),(a[0],y,a[1]),outward,
                              ((0,0),(1,0),(1,1),(0,1)))
                left,right=p0-n0,p0+n0
                edge=LineString((left,right))
                if edge_allowed.covers(edge):
                    if previous is None:
                        # 起步处同时接到左右两侧地面，避免第一块踏面下露出空洞。
                        low_left,low_right=height_at(*left)-.02,height_at(*right)-.02
                    else:
                        low_left=low_right=previous
                    if y>max(low_left,low_right)+.005:
                        mesh.quad((left[0],low_left,left[1]),(right[0],low_right,right[1]),
                                  (right[0],y,right[1]),(left[0],y,left[1]),
                                  (-tangent[0],0,-tangent[1]),((0,0),(1,0),(1,1),(0,1)))
                previous=y
            profiles.append({'osm':key,'start_m':[round(float(v),3) for v in start],
                             'end_m':[round(float(v),3) for v in end],
                             'centerline_m':[[round(float(v),3) for v in p] for p in line.coords],
                             'length_m':round(line.length,2),'terrain_rise_m':round(rise,2),
                             'step_count_estimate':len(segments),
                             'max_rise_m':round(max([0]+[b-a for a,b in zip(levels,levels[1:])]),3)})
    if not profiles:
        raise ValueError('No compatible mapped stairway was generated')
    vertices=mesh.save(assets/'campus-network-stairs.mesh')
    (data/'campus-stair-profile.json').write_text(json.dumps({
        'note':'线位来自 OSM highway=steps；高差来自粗地形，宽度、阶数和侧壁均为估计。',
        'routes':profiles},ensure_ascii=False,indent=2),encoding='utf-8')
    return {'vertices':vertices,'estimated_stairways':len(profiles)}
