"""荔园小巴的独立模型与道路中心线资源。

车型参照校方 2021 年照片，尺寸和内部结构为近似；路线用于交互演示，
并不声称与某个时段实际运营线路或停车站点完全一致。
"""
import math
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_shuttle(*, mesh_type, way_points, ways, assets):
    body,metal,interior,seats,glass,lamps,sign=(mesh_type() for _ in range(7))
    uv=((0,0),(1,0),(1,1),(0,1))

    def box(mesh,x0,x1,y0,y1,z0,z1):
        faces=(
            (((x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)),(-1,0,0)),
            (((x1,y0,z1),(x1,y0,z0),(x1,y1,z0),(x1,y1,z1)),(1,0,0)),
            (((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)),(0,-1,0)),
            (((x0,y1,z1),(x1,y1,z1),(x1,y1,z0),(x0,y1,z0)),(0,1,0)),
            (((x1,y0,z0),(x0,y0,z0),(x0,y1,z0),(x1,y1,z0)),(0,0,-1)),
            (((x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)),(0,0,1)))
        for points,normal in faces:
            mesh.quad(*points,normal,uv)

    def beam(mesh,a,b,width):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        axis=b-a
        axis/=np.linalg.norm(axis)
        reference=np.asarray((0,0,1) if abs(axis[2])<.9 else (0,1,0))
        u=np.cross(axis,reference)
        u=u/np.linalg.norm(u)*width/2
        v=np.cross(axis,u/np.linalg.norm(u))*width/2
        corners=(u+v,-u+v,-u-v,u-v)
        for i in range(4):
            j=(i+1)%4
            normal=(corners[i]+corners[j]); normal/=np.linalg.norm(normal)
            mesh.quad(a+corners[i],b+corners[i],b+corners[j],a+corners[j],normal,uv)

    # 低车身、悬挑的绿色顶篷和底盘。车头朝本地 +X。
    box(body,-2.35,2.40,.32,.72,-.82,.82)
    box(body,-2.45,1.28,1.94,2.04,-.91,.91)
    box(body,-2.47,-2.32,.55,1.00,-.82,.82)
    box(body,1.94,2.42,.65,.91,-.82,.82)
    box(interior,-2.26,1.78,.71,.76,-.78,.78)
    box(interior,-2.48,2.48,.25,.34,-.84,.84)
    box(interior,-2.48,-2.35,.36,.47,-.90,.90)
    box(interior,2.40,2.49,.34,.44,-.90,.90)

    # 乘客侧开放，沿两侧配置立柱和细栏杆；不再用整块玻璃封住座位。
    for z in (-.80,.80):
        for x in (-2.20,-1.28,-.36,.56,1.22):
            beam(metal,(x,.72,z),(x,1.96,z),.045)
        beam(metal,(-2.2,1.29,z),(1.20,1.29,z),.035)
        beam(metal,(-2.2,.77,z),(1.20,.77,z),.035)
    for x in (-1.91,-1.06,-.21,.64):
        for z in (-.40,.40):
            box(seats,x-.12,x+.45,.79,.91,z-.34,z+.34)
            box(seats,x-.16,x-.04,.88,1.35,z-.34,z+.34)
            box(interior,x+.02,x+.10,.72,.79,z-.23,z+.23)
    # 前排驾驶座及仪表台为简化几何。
    box(seats,1.10,1.53,.78,.91,-.68,-.06)
    box(seats,1.05,1.16,.88,1.32,-.68,-.06)
    box(interior,1.74,2.10,.82,.96,-.72,.72)
    beam(metal,(1.75,.90,-.37),(1.47,1.17,-.37),.04)

    # 沿车顶到车头构造多段圆弧前挡风玻璃和绿色两侧框。
    curve=[]
    for i in range(11):
        angle=math.pi*.5*i/10
        curve.append((1.27+1.10*math.sin(angle),.84+1.15*math.cos(angle)))
    for (x0,y0),(x1,y1) in zip(curve,curve[1:]):
        normal=np.cross((0,0,1),(x1-x0,y1-y0,0))
        normal/=np.linalg.norm(normal)
        glass.quad((x0,y0,-.76),(x0,y0,.76),(x1,y1,.76),(x1,y1,-.76),normal,uv)
        for z in (-.80,.80):
            beam(body,(x0,y0,z),(x1,y1,z),.09)
    beam(interior,(2.34,.95,-.60),(1.82,1.65,.25),.025)
    for z in (-.65,.65):
        box(lamps,2.42,2.45,.46,.59,z-.105,z+.105)

    # 正面字样为自行绘制，照片中的校徽细节未使用实拍纹理。
    sign.quad((2.452,.47,.50),(2.452,.47,-.50),(2.452,.72,-.50),(2.452,.72,.50),
              (1,0,0),uv)
    image=Image.new('RGB',(512,128),(92,214,25))
    draw=ImageDraw.Draw(image)
    face=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',58)
    caption=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw.text((256,42),'荔园小巴',font=face,fill=(24,61,22),anchor='mm')
    draw.text((256,101),'SHENZHEN UNIVERSITY BUS',font=caption,fill=(25,64,24),anchor='mm')
    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(assets/'shuttle-sign.ppm')

    vertices={}
    for label,mesh in (('body',body),('metal',metal),('interior',interior),('seats',seats),
                       ('glass',glass),('lamps',lamps),('sign',sign)):
        vertices[label]=mesh.save(assets/f'campus-shuttle-{label}.mesh')

    # 该道路在地图上为连续线，运行时按弧长插值，车体随道路方向及高差变化。
    points=way_points(ways['231530782'])
    if len(points)<2 or any(math.dist(a,b)<.01 for a,b in zip(points,points[1:])):
        raise ValueError('Campus shuttle route has too few or duplicate points')
    with (assets/'campus-shuttle-route.bin').open('wb') as output:
        output.write(b'SZR1'+struct.pack('<I',len(points)))
        for x,z in points:
            output.write(struct.pack('<2f',x,z))
    return {'osm_demo_road':'way/231530782','route_vertices':len(points),
            'route_length_m':round(sum(math.dist(a,b) for a,b in zip(points,points[1:])),1),
            'model_vertices':vertices}
