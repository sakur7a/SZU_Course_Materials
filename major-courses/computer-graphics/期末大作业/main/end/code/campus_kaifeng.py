"""凯风餐厅独立模型：地图轮廓、照片特征和可开合的入口门。

外观根据 2019/2020 年外景及 2023 年入口近照概括；入口朝向、层高、
楼梯位置和少量室内陈设均未实测，资料与估计边界见 data/kaifeng-sources.md。
生成的门扇位于局部坐标，运行时使用单独的铰链变换，不与建筑壳体合并。
"""
import json
import math
import random
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient


def front_geometry(way_points, config):
    a,b=(np.asarray(way_points[i],dtype=float) for i in config['front_edge_node_indexes'])
    tangent=(b-a)/np.linalg.norm(b-a)
    outward=np.asarray((-tangent[1],tangent[0]))
    return a,b,tangent,outward


def make_kaifeng(*, config, geom, points, altitude, mesh_type, extrude, extrude_openings,
                 beam, normal, height_at, assets):
    start,end,u,v=front_geometry(points,config)
    length=float(np.linalg.norm(end-start))
    door_center,door_width=config['entry_center_m'],config['entry_width_m']
    door_height=config['entry_height_m']
    top=altitude+config['height_m']
    floor_y=altitude+.04
    if not (door_width>2 and 6<door_center<length-6):
        raise ValueError('Invalid Kai Feng entrance configuration')
    meshes={name:mesh_type() for name in ('shell','roof','windows','yellow','red','frame','rail',
             'stairs','apron','interior','furniture','door-glass','door-frame')}
    def p(s,y,d=0):
        q=start+u*s+v*d
        return (float(q[0]),float(y),float(q[1]))
    def strip(s0,s1,y0,y1,d,mesh,uv_scale=4):
        mesh.quad(p(s0,y0,d),p(s1,y0,d),p(s1,y1,d),p(s0,y1,d),
                  (float(v[0]),0,float(v[1])),
                  ((0,0),((s1-s0)/uv_scale,0),((s1-s0)/uv_scale,(y1-y0)/3.2),(0,(y1-y0)/3.2)))
    def window(s0,s1,y0,y1):
        strip(s0,s1,y0,y1,.07,meshes['windows'])
        for s in np.linspace(s0,s1,max(2,int((s1-s0)/2.6)+1)):
            beam(meshes['frame'],p(s,y0,.12),p(s,y1,.12),.055)
        for y in (y0,(y0+y1)/2,y1):
            beam(meshes['frame'],p(s0,y,.12),p(s1,y,.12),.055)

    def lettering(text,center,bottom,width,height,d,font_path):
        # 自行排字后把字形的水平实心片段转为红色几何；复用现有材质与输出，
        # 不把现场照片、水印或商业广告作为纹理，也不需要改动渲染器。
        font=ImageFont.truetype(font_path,100)
        box=font.getbbox(text)
        mask=Image.new('L',(box[2]-box[0],box[3]-box[1]),0)
        ImageDraw.Draw(mask).text((-box[0],-box[1]),text,font=font,fill=255)
        pixels_high=48 if height>.2 else 16
        mask=mask.resize((max(1,round(width/height*pixels_high)),pixels_high),Image.Resampling.LANCZOS)
        for row in range(mask.height):
            values=[mask.getpixel((col,row))>=128 for col in range(mask.width)]+[False]
            first=None
            for col,value in enumerate(values):
                if value and first is None:
                    first=col
                elif not value and first is not None:
                    strip(center-width/2+width*first/mask.width,
                          center-width/2+width*col/mask.width,
                          bottom+height*(1-(row+1)/mask.height),
                          bottom+height*(1-row/mask.height),d,meshes['red'])
                    first=None

    def railing(s0,d0,s1,d1,y0,y1,floor_at=None):
        # 2020 年外景可辨细密竖杆；间距、杆径和高度均按比例估计。
        span=math.hypot(s1-s0,d1-d0)
        top=config['railing_height_m']
        def at(t,h=0):
            return p(s0+(s1-s0)*t,y0+(y1-y0)*t+h,d0+(d1-d0)*t)
        for spacing,radius in ((1.45,.035),(config['railing_picket_spacing_m'],.012)):
            count=max(1,int(math.ceil(span/spacing)))
            for t in np.linspace(0,1,count+1):
                t=float(t)
                lower=list(at(t,-.02 if radius>.02 else .12))
                if floor_at is not None and radius>.02:
                    lower[1]=floor_at(t)-.02
                beam(meshes['rail'],lower,at(t,top),radius)
        for h,radius in ((.12,.022),(top,.035)):
            beam(meshes['rail'],at(0,h),at(1,h),radius)

    # 主壳体按 OSM 外轮廓建模，并在门位置切出可通行的开口。
    entry=LineString((p(door_center,0,-3)[::2],p(door_center,0,4)[::2]))
    extrude_openings(meshes['shell'],meshes['roof'],geom,altitude-2,top,(),
                     leveled_paths=((entry,floor_y),),unit_width_override=4.0)

    # 非正面立面的窗格只提供照片风格下的近似，未取得实测立面图。
    for poly in ([geom] if geom.geom_type=='Polygon' else list(geom.geoms)):
        coords=list(orient(poly,sign=1).exterior.coords)
        for aa,bb in zip(coords,coords[1:]):
            aa,bb=np.asarray(aa),np.asarray(bb)
            delta=bb-aa
            L=float(np.linalg.norm(delta))
            if L<5 or LineString((aa,bb)).distance(entry)<.2:
                continue
            t=delta/L
            n=np.asarray((t[1],-t[0]))
            for floor in range(4):
                low=altitude+floor*3.5+.65
                high=min(top-.3,low+2.15)
                for k in range(max(1,int((L-2)/4))):
                    first=1+k*4; last=min(first+3.1,L-1)
                    if last-first<1:continue
                    q0=aa+t*first+n*.06; q1=aa+t*last+n*.06
                    meshes['windows'].quad((q0[0],low,q0[1]),(q1[0],low,q1[1]),
                                           (q1[0],high,q1[1]),(q0[0],high,q0[1]),
                                           (n[0],0,n[1]),((0,0),(1,0),(1,1),(0,1)))

    # 绿灰色玻璃、黄色墙框和红色楼梯背墙来自历史照片的可见特征。
    for s0,s1 in ((1.5,door_center-door_width/2-.15),(door_center+door_width/2+.15,28.0)):
        window(s0,s1,altitude+.25,altitude+3.15)
    for floor in (1,2,3):
        y0=altitude+floor*3.5+.45
        y1=min(top-.45,y0+2.25)
        for s0,s1 in ((2.0,14.8),(27.7,30.6)):
            window(s0,s1,y0,y1)
    # 黄色面中保留竖向窗洞，四个矩形分别构成窗洞两侧及上下边框。
    for s0,s1,y0,y1 in ((15.5,18.7,3.55,12.6),(23.1,27.0,3.55,12.6),
                         (18.7,23.1,3.55,8.0),(18.7,23.1,10.9,12.6)):
        strip(s0,s1,altitude+y0,altitude+y1,.10,meshes['yellow'])
    window(18.7,23.1,altitude+8.0,altitude+10.9)
    for s0,s1 in ((32.0,35.0),(36.0,39.1)):
        strip(s0,s1,altitude+.15,altitude+11.5,.10,meshes['red'])
    # 2023 年入口近照显示浅色横向名牌，已不同于 2019/2020 年的红色雨棚。
    sign=config['entry_sign']
    sign_width=sign['width_m'];sign_bottom=altitude+sign['bottom_m']
    sign_poly=Polygon((p(door_center-sign_width/2,0,.16)[::2],p(door_center+sign_width/2,0,.16)[::2],
                       p(door_center+sign_width/2,0,.24)[::2],p(door_center-sign_width/2,0,.24)[::2]))
    extrude(meshes['frame'],meshes['frame'],sign_poly,sign_bottom,sign_bottom+sign['height_m'])
    lettering('凯 风 餐 厅',door_center,sign_bottom+.40,3.30,.50,.252,'C:/Windows/Fonts/simsun.ttc')
    lettering('K A I  F E N G  C A N T E E N',door_center,sign_bottom+.20,3.00,.105,.252,
              'C:/Windows/Fonts/msyh.ttc')

    # 局部窗下可见小型室外机；数量与安装位置依照片概括。
    for s in (13.4,28.8):
        unit=Polygon((p(s,0,.12)[::2],p(s+1.05,0,.12)[::2],
                      p(s+1.05,0,.68)[::2],p(s,0,.68)[::2]))
        extrude(meshes['frame'],meshes['frame'],unit,altitude+4.2,altitude+5.0)
        for y in np.arange(altitude+4.30,altitude+4.95,.12):
            beam(meshes['rail'],p(s+.10,float(y),.70),p(s+.95,float(y),.70),.025)

    # 红墙前的外廊、白色大框、横梁与楼梯；构件的尺寸是视觉估计。
    for s in (29.9,46.0):
        beam(meshes['frame'],p(s,altitude-.15,3.1),p(s,altitude+12.2,3.1),.65)
    beam(meshes['frame'],p(29.9,altitude+12.2,3.1),p(46.0,altitude+12.2,3.1),.70)
    for s in (31.0,35.0,39.0,43.0):
        beam(meshes['frame'],p(s,altitude+12.2,.2),p(s,altitude+12.2,3.1),.28)
    stair=config['external_stair']
    step_count=stair['tread_count'];step_run=stair['tread_run_m']
    stair_bottom=altitude+stair['bottom_offset_m'];stair_top=altitude+stair['top_offset_m']
    step_rise=(stair_top-stair_bottom)/(step_count-1)
    top_tread_start=44.3-(step_count-1)*step_run
    landing_start=stair['landing_start_m']
    for level in (3.55,7.05,10.55):
        # 实景外廊有完整楼板与浅色边梁，不能只用一根横杆代替。
        if level>3.55:
            slab=Polygon((p(30.1,0,.05)[::2],p(44.2,0,.05)[::2],
                          p(44.2,0,2.58)[::2],p(30.1,0,2.58)[::2]))
            extrude(meshes['frame'],meshes['frame'],slab,altitude+level-.20,altitude+level)
        strip(30.2,44.3,altitude+level-.22,altitude+level,2.60,meshes['frame'])
        spans=((30.5,landing_start),(top_tread_start,44.0)) if level==3.55 else ((30.5,44.0),)
        for s0,s1 in spans:
            railing(s0,2.62,s1,2.62,altitude+level,altitude+level)
    for i in range(step_count):
        s1=44.3-i*step_run
        s0=s1-step_run
        y=stair_bottom+i*step_rise
        poly=Polygon((p(s0,0,3.15)[::2],p(s1,0,3.15)[::2],p(s1,0,5.20)[::2],p(s0,0,5.20)[::2]))
        extrude(meshes['stairs'],meshes['stairs'],poly,y-step_rise,y)
    # 顶踏步与外廊同高，补转向平台；栏杆在平台入口留口，不横挡上楼路线。
    landing=Polygon((p(landing_start,0,3.15)[::2],p(44.3-step_count*step_run,0,3.15)[::2],
                     p(44.3-step_count*step_run,0,5.20)[::2],p(landing_start,0,5.20)[::2]))
    extrude(meshes['stairs'],meshes['stairs'],landing,stair_top-.20,stair_top)
    railing(44.3,5.23,top_tread_start,5.23,stair_bottom,stair_top,
            floor_at=lambda t:stair_bottom+min(step_count-1,int(t*(step_count-1)+1e-7))*step_rise)
    railing(landing_start,5.23,top_tread_start,5.23,stair_top,stair_top)
    railing(landing_start,2.62,landing_start,5.23,stair_top,stair_top)
    for d in (3.2,5.08):
        beam(meshes['frame'],p(44.3,stair_bottom-.15,d),p(44.3-step_count*step_run,stair_top-.20,d),.18)
    # 顶端外廊与楼梯相接，二层内部未重建。
    balcony=Polygon((p(30.1,0,.05)[::2],p(44.2,0,.05)[::2],
                     p(44.2,0,3.3)[::2],p(30.1,0,3.3)[::2]))
    extrude(meshes['stairs'],meshes['stairs'],balcony,altitude+3.36,altitude+3.55)

    # 小入口前按地形渐接到道路侧的人行空间，地面高差与坡度未测绘。
    apron=meshes['apron']
    for i in range(8):
        for j in range(9):
            s0,s1=door_center-5+i*1.25,door_center-5+(i+1)*1.25
            d0,d1=j*1.0,(j+1)*1.0
            points3=[]
            for s,d in ((s0,d0),(s1,d0),(s1,d1),(s0,d1)):
                q=start+u*s+v*d
                terrain=height_at(float(q[0]),float(q[1]))
                near=height_at(*(start+u*s))
                t=max(0.0,1-d/9.0)
                t=t*t*(3-2*t)
                points3.append((q[0],terrain+(floor_y-near)*t+.025,q[1]))
            apron.quad(*points3,normal(*points3[:3]),((0,0),(1,0),(1,1),(0,1)))
    # 楼梯底端通过独立渐变铺装接回路侧步行空间。
    for j in range(9):
        d0,d1=3.2+j*(7.8/9),3.2+(j+1)*(7.8/9)
        points3=[]
        for s,d in ((44.3,d0),(45.7,d0),(45.7,d1),(44.3,d1)):
            q=start+u*s+v*d
            terrain=height_at(float(q[0]),float(q[1]))
            near=height_at(*(start+u*s+v*5.2))
            t=max(0.0,min(1.0,(11.0-d)/5.8))
            t=t*t*(3-2*t)
            y=altitude-.05 if d<=5.2 else terrain+(altitude-.05-near)*t
            points3.append((q[0],y+.025,q[1]))
        apron.quad(*points3,normal(*points3[:3]),((0,0),(1,0),(1,1),(0,1)))

    # 入口内仅有示意地面与少量桌椅，不宣称还原餐厅完整内部。
    meshes['interior'].horizontal(geom,floor_y-.01)
    # 入口旁的休憩长凳仅借鉴历史照片的室外座位氛围。
    for s in (5.0,9.5):
        seat=Polygon((p(s-1.5,0,2.2)[::2],p(s+1.5,0,2.2)[::2],
                      p(s+1.5,0,2.65)[::2],p(s-1.5,0,2.65)[::2]))
        y=height_at(*(start+u*s+v*2.4))
        extrude(meshes['furniture'],meshes['furniture'],seat,y+.45,y+.53)
        for ds in (-1.1,1.1):
            beam(meshes['furniture'],p(s+ds,y,2.42),p(s+ds,y+.45,2.42),.09)
    for s in (door_center-4,door_center+4):
        for d in (-3.3,-5.6):
            poly=Polygon((p(s-.65,0,d-.38)[::2],p(s+.65,0,d-.38)[::2],
                          p(s+.65,0,d+.38)[::2],p(s-.65,0,d+.38)[::2]))
            extrude(meshes['furniture'],meshes['furniture'],poly,floor_y+.72,floor_y+.79)
            for ds in (-.50,.50):
                beam(meshes['furniture'],p(s+ds,floor_y+.03,d),p(s+ds,floor_y+.72,d),.055)
            # 两只靠窗侧的简化座椅留出入口中轴通道。
            for ds in (-.95,.95):
                seat=Polygon((p(s+ds-.20,0,d-.18)[::2],p(s+ds+.20,0,d-.18)[::2],
                              p(s+ds+.20,0,d+.18)[::2],p(s+ds-.20,0,d+.18)[::2]))
                extrude(meshes['furniture'],meshes['furniture'],seat,floor_y+.43,floor_y+.48)
                beam(meshes['furniture'],p(s+ds,floor_y+.02,d),p(s+ds,floor_y+.43,d),.055)

    # 两个门扇共享局部网格：正方向 +X，铰链位于原点。
    leaf=door_width/2
    meshes['door-glass'].quad((0,0,0),(leaf,0,0),(leaf,door_height,0),(0,door_height,0),
                              (0,0,1),((0,0),(1,0),(1,1),(0,1)))
    for x in (0,leaf):
        beam(meshes['door-frame'],(x,0,.04),(x,door_height,.04),.085)
    for y in (0,1.12,door_height):
        beam(meshes['door-frame'],(0,y,.04),(leaf,y,.04),.075)
    beam(meshes['door-frame'],(leaf-.13,1.02,.12),(leaf-.13,1.65,.12),.055)
    left=p(door_center-door_width/2,floor_y,.12)
    right=p(door_center+door_width/2,floor_y,.12)
    with (assets/'campus-kaifeng-entry.bin').open('wb') as f:
        f.write(b'KFE1'+struct.pack('<7f',left[0],left[2],right[0],right[2],float(u[0]),float(u[1]),floor_y))

    # 自绘浅色小瓷砖，保持独立材质，不将照片作为贴图打包。
    rng=random.Random(230935723)
    texture=Image.new('RGB',(256,256),(226,225,219))
    draw=ImageDraw.Draw(texture)
    for y in range(0,256,16):
        for x in range(0,256,16):
            k=rng.randrange(-4,5)
            draw.rectangle((x+1,y+1,x+15,y+15),fill=(229+k,228+k,222+k))
    texture.save(assets/'kaifeng-tiles.ppm')
    counts={name:mesh.save(assets/f'campus-kaifeng-{name}.mesh') for name,mesh in meshes.items()}
    return {'model_vertices':counts,'height_m':config['height_m'],
            'entry_floor_estimate_m':round(floor_y,2),'front_length_m':round(length,1)}
