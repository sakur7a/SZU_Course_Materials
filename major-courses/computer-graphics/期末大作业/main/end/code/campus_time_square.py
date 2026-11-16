"""按地图和校园照片生成可独立维护的时光广场铺装与绿化。"""
import math

import numpy as np
from shapely.geometry import LineString, Point


def make_time_square(*, square, paving_area, lawn, entry_area, surface_height,
                     drape, mesh_type, beam, clean, lines, assets):
    paving,curb,hedge,leaves=(mesh_type() for _ in range(4))
    center=square.centroid
    y=surface_height(center.x,center.y)
    # 已有汇元楼门前铺装单独建模，广场砖面在入口及草坪处留空。
    surface=paving_area.difference(lawn.buffer(.22)).difference(entry_area)
    drape(paving,surface,.055,lambda x,z:(x/3.2,z/3.2))
    ring=clean(lawn.buffer(.22).difference(lawn.buffer(-.22)))
    curb.horizontal(ring,y+.08)
    boundary=list(lawn.exterior.coords)
    for a,b in zip(boundary[:-1],boundary[1:]):
        dx,dz=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dz)
        if length<.01:
            continue
        curb.quad((a[0],y-.06,a[1]),(b[0],y-.06,b[1]),
                  (b[0],y+.08,b[1]),(a[0],y+.08,a[1]),
                  (dz/length,0,-dx/length),((0,0),(1,0),(1,1),(0,1)))
    # 航拍可见放射状低绿篱；形状及数量近似，叶簇只用于增加近景层次。
    dial=Point(-243.2,-48.6).buffer(6.4)
    for angle in np.arange(0,math.tau,math.tau/10):
        end=(center.x+25*math.cos(angle),center.y+25*math.sin(angle))
        ray=LineString(((center.x,center.y),end)).intersection(lawn.buffer(-.5)).difference(dial)
        for part in lines(ray):
            if part.length<.4:
                continue
            a,b=part.coords[0],part.coords[-1]
            beam(hedge,(a[0],y+.18,a[1]),(b[0],y+.18,b[1]),.12)
            for d in np.arange(0,part.length,1.25):
                p,q=part.interpolate(float(d)),part.interpolate(min(float(d)+1.25,part.length))
                dx,dz=q.x-p.x,q.y-p.y
                length=math.hypot(dx,dz)
                if length<.05:
                    continue
                nx,nz=-dz/length,dx/length
                for side in (-1,1):
                    x0,z0=p.x+nx*.13*side,p.y+nz*.13*side
                    x1,z1=q.x+nx*.13*side,q.y+nz*.13*side
                    leaves.quad((x0,y+.08,z0),(x1,y+.08,z1),
                                (x1,y+.44,z1),(x0,y+.44,z0),
                                (nx*side,0,nz*side),((0,0),(1,0),(1,1),(0,1)))
    return {'paving':paving.save(assets/'campus-time-paving.mesh'),
            'curb':curb.save(assets/'campus-time-curb.mesh'),
            'hedges':hedge.save(assets/'campus-time-hedges.mesh'),
            'hedge-leaves':leaves.save(assets/'campus-time-hedge-leaves.mesh')}
