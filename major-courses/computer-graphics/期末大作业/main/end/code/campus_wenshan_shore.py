"""上文山湖东岸的局部地形过渡。

低分辨率高程在平整湖面和东侧环路之间产生明显折坡；依据湖岸照片，
仅在水面与地图道路之间作平滑视觉估计，尽量保留水面、道路和建筑基座。
"""
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


def smooth_east_bank(heights, *, x_bounds, z_bounds, lake, lake_level, road, buildings):
    before=heights.copy()
    nz,nx=before.shape
    x0,x1=x_bounds
    z0,z1=z_bounds
    dx,dz=(x1-x0)/(nx-1),(z1-z0)/(nz-1)
    protected=prep(unary_union([geom.buffer(3) for _,_,geom in buildings]))

    def smooth(t):
        t=max(0.0,min(1.0,t))
        return t*t*(3-2*t)

    def sample(x,z):
        u=max(0,min(nx-1,(x-x0)/dx))
        v=max(0,min(nz-1,(z-z0)/dz))
        i,j=min(int(u),nx-2),min(int(v),nz-2)
        a,b=u-i,v-j
        p,q,r,s=before[j,i],before[j,i+1],before[j+1,i],before[j+1,i+1]
        return float((1-b)*p+(b-a)*r+a*s if b>=a else (1-a)*p+b*s+(a-b)*q)

    changed=[]
    # 投影距离限定在已单独平滑的文山湖东侧道路路段。
    for j in range(nz):
        z=z0+j*dz
        if not 130<z<280:
            continue
        for i in range(nx):
            x=x0+i*dx
            if not -245<x<-130:
                continue
            p=Point(x,z)
            distance_lake=lake.distance(p)
            distance_road=road.distance(p)
            along=road.project(p)
            if (distance_lake<8 or distance_lake>50 or distance_road<4 or
                    not 55<along<205 or protected.covers(p)):
                continue
            nearest=road.interpolate(along)
            gap=lake.distance(nearest)
            # 只调整位于湖与道路之间的地带，避免影响道路另一侧。
            if gap<10 or distance_lake+distance_road>gap+8:
                continue
            t=distance_lake/(distance_lake+distance_road)
            road_y=max(lake_level+.2,sample(nearest.x,nearest.y))
            target=(lake_level+.2)*(1-t)+road_y*t
            weight=smooth((along-55)/18)*smooth((205-along)/18)
            weight*=smooth((distance_road-4)/5)*smooth((50-distance_lake)/12)
            # 仅消除过高的异常折坡，原本低于插值面的地形不抬高。
            correction=min(0.0,target-float(before[j,i]))*weight
            if correction<-.001:
                heights[j,i]=before[j,i]+correction
                changed.append(abs(correction))
    return {'cells_adjusted':len(changed),
            'max_downward_adjustment_m':round(max(changed,default=0),2),
            'note':'湖岸至既有道路之间的视觉插值，不是现场测量'}
