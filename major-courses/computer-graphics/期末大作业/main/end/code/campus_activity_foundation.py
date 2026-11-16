"""为坡地上的学生活动中心生成独立的近似基础墙。"""
import math

import numpy as np
from shapely.geometry.polygon import orient


def add_foundation(mesh, footprint, *, top, terrain_height, polygons):
    """从楼体墙脚分段接到可见地面；尺寸和实际建筑做法未经测量。"""
    uv=((0,0),(1,0),(1,1),(0,1))
    for poly in polygons(footprint):
        points=list(orient(poly,sign=1).exterior.coords)
        for a,b in zip(points[:-1],points[1:]):
            dx,dz=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dz)
            if length<.05:
                continue
            count=max(1,int(math.ceil(length/2.0)))
            normal=(dz/length,0,-dx/length)
            for t0,t1 in zip(np.linspace(0,1,count+1)[:-1],np.linspace(0,1,count+1)[1:]):
                x0,z0=a[0]+dx*t0,a[1]+dz*t0
                x1,z1=a[0]+dx*t1,a[1]+dz*t1
                low0=min(top,terrain_height(x0,z0)-.12)
                low1=min(top,terrain_height(x1,z1)-.12)
                if max(top-low0,top-low1)<.02:
                    continue
                mesh.quad((x0,low0,z0),(x1,low1,z1),(x1,top,z1),(x0,top,z0),normal,uv)
