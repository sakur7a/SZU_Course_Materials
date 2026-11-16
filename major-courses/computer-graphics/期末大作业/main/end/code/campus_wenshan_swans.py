"""上文山湖黑天鹅家族：自建网格和受湖面边界约束的演示轨迹。

体色、红喙和灰色幼鸟参考学校照片；数量、局部形态与运动仅作近似。
模型以水面为 y=0，鸟头沿局部 +X；成年、幼鸟、喙与水纹分层。
"""
import json
import math
import struct

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point


def make_swans(*, mesh_type, upper_lake, water_level, assets, data):
    cfg=json.loads((data/'wenshan-swans.json').read_text(encoding='utf-8'))
    cx,cz=cfg['center_xz_m']
    rx,rz=cfg['radii_m']
    speed=cfg['angular_speed_rad_s']
    phases=cfg['phase_offsets_rad']
    if len(phases)!=5 or not (0 < speed < .2):
        raise ValueError('Invalid swan animation parameters')
    # 留出身体及水纹的宽度，避免湖水边界较窄处穿岸。
    safe=upper_lake.buffer(-2.0)
    for i in range(720):
        a=2*math.pi*i/720
        if not safe.covers(Point(cx+rx*math.cos(a),cz+rz*math.sin(a))):
            raise ValueError('Swan route is too close to the mapped lakeshore')
    with (assets/'campus-wenshan-swan-route.bin').open('wb') as f:
        f.write(b'SWN1')
        f.write(struct.pack('<6f',cx,cz,rx,rz,water_level,speed))
        f.write(struct.pack('<5f',*phases))

    adult,feather,bill,tip,adult_eye,chick,chick_wing,chick_bill,chick_eye,wake=(mesh_type() for _ in range(10))

    def vertex(mesh,p,n,uv):
        mesh.v(tuple(p),tuple(n),uv)

    def smooth_triangle(mesh, a,b,c):
        for p,n,uv in (a,b,c):
            vertex(mesh,p,n,uv)

    def ellipsoid(mesh,center,radii,latitudes=12,longitudes=20):
        center=np.asarray(center,dtype=float)
        radii=np.asarray(radii,dtype=float)
        def sample(i,j):
            theta=math.pi*i/latitudes
            phi=2*math.pi*j/longitudes
            unit=np.array((math.cos(theta),math.sin(theta)*math.sin(phi),
                           math.sin(theta)*math.cos(phi)))
            n=unit/radii
            return center+unit*radii,n/np.linalg.norm(n),(j/longitudes,i/latitudes)
        for i in range(latitudes):
            for j in range(longitudes):
                a,b=sample(i,j),sample(i+1,j)
                c,d=sample(i+1,j+1),sample(i,j+1)
                smooth_triangle(mesh,a,b,c)
                smooth_triangle(mesh,a,c,d)

    def catmull(points, index, t):
        p0=points[max(0,index-1)]; p1=points[index]
        p2=points[min(len(points)-1,index+1)]; p3=points[min(len(points)-1,index+2)]
        return .5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+
                   (-p0+3*p1-3*p2+p3)*t*t*t)

    def neck(mesh,controls):
        # Catmull-Rom 曲线形成连续的颈部，截面法线随曲线方向变化。
        points=[np.array(p,dtype=float) for p in controls]
        rows=[]
        for k in range(len(points)-1):
            for j in range(5): rows.append(catmull(points,k,j/5))
        rows.append(points[-1])
        sides=16
        def ring(i,j):
            q=rows[i]
            tangent=rows[min(i+1,len(rows)-1)][:2]-rows[max(i-1,0)][:2]
            tangent/=np.linalg.norm(tangent)
            normal2=np.array((-tangent[1],tangent[0],0.0))
            angle=2*math.pi*j/sides
            n=normal2*math.cos(angle)+np.array((0,0,1))*math.sin(angle)
            return np.array((q[0],q[1],0))+n*q[2],n,(j/sides,i/(len(rows)-1))
        for i in range(len(rows)-1):
            for j in range(sides):
                a,b=ring(i,j),ring(i+1,j)
                c,d=ring(i+1,j+1),ring(i,j+1)
                smooth_triangle(mesh,a,b,c)
                smooth_triangle(mesh,a,c,d)

    def tapered_bill(mesh,sections):
        # 横截面为椭圆，分段衔接喙根、红色主体和浅色尖端。
        sides=12
        def point(section,j):
            x,y,ry,rz=section
            a=2*math.pi*j/sides
            p=np.array((x,y+ry*math.sin(a),rz*math.cos(a)))
            n=np.array((.18,math.sin(a),math.cos(a)))
            return p,n/np.linalg.norm(n),(j/sides,0)
        for i in range(len(sections)-1):
            for j in range(sides):
                a,b=point(sections[i],j),point(sections[i+1],j)
                c,d=point(sections[i+1],j+1),point(sections[i],j+1)
                smooth_triangle(mesh,a,b,c)
                smooth_triangle(mesh,a,c,d)

    # 成年天鹅的低矮船形躯干、折叠翅、长颈和红喙。
    ellipsoid(adult,(-.08,.17,0),(.62,.30,.34))
    for side in (-1,1):
        ellipsoid(feather,(-.13,.31,side*.225),(.46,.17,.12),10,18)
        # 小幅抬起的卷曲羽缘，保持整体剪影柔和。
        for k in range(5):
            x=-.49+k*.105
            z=side*(.15+.018*k)
            ellipsoid(feather,(x,.42+.009*(k%2),z),(.15,.026,.048),6,10)
    neck(adult,[(.36,.27,.14),(.43,.39,.12),(.42,.52,.09),(.37,.67,.075),
                (.39,.82,.070),(.46,.98,.078),(.54,1.04,.085)])
    ellipsoid(adult,(.56,1.04,0),(.13,.095,.09),10,16)
    for side in (-1,1):
        ellipsoid(adult_eye,(.585,1.065,side*.085),(.020,.018,.012),6,10)
    tapered_bill(bill,[(.65,1.04,.065,.065),(.73,1.01,.058,.055),(.79,.98,.038,.040)])
    tapered_bill(tip,[(.79,.98,.038,.040),(.82,.965,.025,.028),(.835,.96,.004,.005)])

    # 幼鸟为灰褐色、体形短圆且颈更短；不是缩小的成年鸟。
    ellipsoid(chick,(-.04,.13,0),(.36,.19,.24),10,18)
    for side in (-1,1):
        ellipsoid(chick_wing,(-.08,.21,side*.16),(.26,.10,.08),8,14)
    neck(chick,[(.18,.19,.10),(.24,.28,.08),(.25,.36,.065),(.28,.44,.065)])
    ellipsoid(chick,(.30,.44,0),(.095,.075,.075),8,14)
    for side in (-1,1):
        ellipsoid(chick_eye,(.325,.458,side*.068),(.013,.013,.009),6,10)
    tapered_bill(chick_bill,[(.37,.44,.045,.045),(.44,.415,.03,.032),(.47,.40,.004,.005)])

    # 开口的椭圆细水纹跟随身体运动；形状为场景效果，并非水动力模拟。
    for radius in (.62,.92):
        for i in range(40):
            a=math.pi*.60+math.pi*.80*i/40
            b=math.pi*.60+math.pi*.80*(i+1)/40
            def p(t,r): return (-.22+r*math.cos(t),.022,r*.58*math.sin(t))
            wake.quad(p(a,radius),p(b,radius),p(b,radius+.018),p(a,radius+.018),
                      (0,1,0),((0,0),(1,0),(1,1),(0,1)))
    # 程序绘制细羽纹理，低对比度叠加到黑色及灰色身体，不使用实景照片。
    rng=np.random.default_rng(5477990523)
    grain=np.clip(rng.normal(224,7,(512,512)),195,247).astype(np.uint8)
    texture=Image.fromarray(grain,'L').convert('RGB')
    draw=ImageDraw.Draw(texture)
    for row in range(-1,22):
        for col in range(-1,18):
            x=col*32+(row%2)*16
            y=row*25
            shade=int(rng.integers(188,211))
            draw.arc((x,y,x+33,y+37),25,150,fill=(shade,shade,shade),width=1)
            draw.arc((x+2,y+1,x+31,y+34),28,145,fill=(236,236,236),width=1)
    texture.save(assets/'wenshan-swan-feathers.ppm')
    meshes={'adult':adult,'feather':feather,'bill':bill,'bill-tip':tip,'adult-eye':adult_eye,
            'chick':chick,'chick-wing':chick_wing,'chick-bill':chick_bill,'chick-eye':chick_eye,'wake':wake}
    counts={name:mesh.save(assets/f'campus-wenshan-swan-{name}.mesh') for name,mesh in meshes.items()}
    return {'model_vertices':counts,'center_xz_m':[cx,cz],
            'water_level_m':round(water_level,3),'route_radii_m':[rx,rz],
            'individuals':len(phases)}
