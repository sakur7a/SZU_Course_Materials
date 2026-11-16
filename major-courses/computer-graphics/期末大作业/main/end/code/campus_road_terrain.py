"""对粗高程中道路与场地接缝造成的局部尖折进行可重复的视觉修正。

道路线位采用地图数据，目标坡度只用于抑制插值伪影，并非实测或道路设计值。
"""
import math

import numpy as np
from shapely.geometry import Point
from shapely.ops import substring
from shapely.prepared import prep


def smooth_segment(heights, *, line, center, half_length, target_grade,
                   x_bounds, z_bounds, protected):
    """在固定地物之外拟合道路中心和两侧，同时保留道路两端的高差。"""
    nz,nx=heights.shape
    x0,x1=x_bounds
    z0,z1=z_bounds
    dx,dz=(x1-x0)/(nx-1),(z1-z0)/(nz-1)

    def triangle_weights(x,z):
        # 与 OpenGL 地形网格使用的两片三角形相同，避免按双线性面拟合后仍有错层。
        u=max(0,min(nx-1,(x-x0)/dx));v=max(0,min(nz-1,(z-z0)/dz))
        i,j=min(int(u),nx-2),min(int(v),nz-2)
        a,b=u-i,v-j
        if b>=a:
            return [((j,i),1-b),((j+1,i),b-a),((j+1,i+1),a)]
        return [((j,i),1-a),((j,i+1),a-b),((j+1,i+1),b)]

    def height(x,z):
        return sum(float(heights[j,i])*weight for (j,i),weight in triangle_weights(x,z))

    midpoint=line.project(Point(*center))
    start,end=max(0,midpoint-half_length),min(line.length,midpoint+half_length)
    if end-start<20:
        raise ValueError('The selected road segment is too short to smooth')
    distances=np.linspace(start,end,max(3,int(round((end-start)/2.5))+1))
    original=np.asarray([height(p.x,p.y) for p in (line.interpolate(d) for d in distances)])
    # 平滑纵断面并约束过急的变化；首尾仍保持原来的相对高度。
    second=np.diff(np.eye(len(distances)),n=2,axis=0)
    weights=np.ones(len(distances));weights[:3]=weights[-3:]=15.0
    profile=np.linalg.solve(np.diag(weights)+55*(second.T@second),weights*original)
    for _ in range(1400):
        diff=np.diff(profile)
        excess=np.sign(diff)*np.maximum(0,np.abs(diff)-target_grade*np.diff(distances))*.48
        profile[:-1]+=excess
        profile[1:]-=excess
        profile[0],profile[-1]=original[0],original[-1]

    # 未被建筑和广场占用的地形网格作为变量，其余点固定，保持独立地标的基座。
    zone=substring(line,max(0,start-15),min(line.length,end+15)).buffer(23)
    locked=prep(protected.buffer(1.0))
    lookup={}
    for j in range(nz):
        for i in range(nx):
            p=Point(x0+i*dx,z0+j*dz)
            if zone.covers(p) and not locked.covers(p):
                lookup[j,i]=len(lookup)
    if not lookup:
        raise ValueError('No adjustable terrain grid points near the selected road')

    rows,values=[],[]
    for d,level in zip(distances,profile):
        p=line.interpolate(float(d))
        a=line.interpolate(max(0,d-.5));b=line.interpolate(min(line.length,d+.5))
        vx,vz=b.x-a.x,b.y-a.y
        length=math.hypot(vx,vz)
        if length<.05:
            continue
        tx,tz=-vz/length,vx/length
        fade=max(0,min(1,(d-start)/10,(end-d)/10))
        for offset in (-5.0,-3.0,0.0,3.0,5.0):
            x,z=p.x+tx*offset,p.y+tz*offset
            if locked.covers(Point(x,z)):
                continue
            row=np.zeros(len(lookup))
            for node,coefficient in triangle_weights(x,z):
                if node in lookup:
                    row[lookup[node]]=coefficient
            if np.linalg.norm(row)<.01:
                continue
            # 离路心更远的人行带权重稍低，优先保持道路横向平顺。
            weight=2.0 if abs(offset)>3 else 4.5
            rows.append(row*weight)
            values.append((level-height(x,z))*fade*weight)
    # 相邻网格的改变量也应连续，避免路侧再出现新的地形折线。
    for node,index in lookup.items():
        row=np.zeros(len(lookup));row[index]=.22
        rows.append(row);values.append(0.0)
        for adjacent in ((node[0]+1,node[1]),(node[0],node[1]+1)):
            row=np.zeros(len(lookup));row[index]=.9
            if adjacent in lookup:
                row[lookup[adjacent]]=-.9
            rows.append(row);values.append(0.0)
    delta=np.linalg.lstsq(np.asarray(rows),np.asarray(values),rcond=None)[0]
    for (j,i),change in zip(lookup,delta):
        heights[j,i]+=change
    final=np.asarray([height(p.x,p.y) for p in (line.interpolate(d) for d in distances)])
    return {
        'start_m':round(float(start),2), 'end_m':round(float(end),2),
        'before_max_sample_grade':round(float(np.max(np.abs(np.diff(original)/np.diff(distances)))),3),
        'after_max_sample_grade':round(float(np.max(np.abs(np.diff(final)/np.diff(distances)))),3),
        'end_height_difference_m':round(float(final[-1]-final[0]),2),
        'max_grid_adjustment_m':round(float(np.max(np.abs(delta))),2),
        'note':'坡度与局部地形均为视觉拟合，非实测数据'
    }
