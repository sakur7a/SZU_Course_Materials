"""沧海校区五个独立校门模型；位置有地图依据，构件尺寸与设备为近似。"""
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon


def make_gates(*, mesh_type, height_at, beam, extrude, assets, data, vehicle_surface):
    """生成按校门和材料分离的资源；中央车行线保持开放以便检查。"""
    catalog=json.loads((data/'gates-south.json').read_text(encoding='utf8'))
    records=[]
    colors={'stone':[.81,.83,.80], 'metal':[.44,.49,.49],
            'glass':[.55,.70,.70], 'roof':[.90,.91,.87], 'sign':[1,1,1]}
    for config in catalog['gates']:
        gate_id=config['id'];cx,cz=config['center_xz_m']
        forward=np.asarray(config['inward_xz'],dtype=float)
        forward/=np.linalg.norm(forward)
        lateral=np.asarray((forward[1],-forward[0]))
        meshes={key:mesh_type() for key in colors}
        skipped=0

        def xz(u,v):
            p=np.asarray((cx,cz))+lateral*u+forward*v
            return float(p[0]),float(p[1])

        def point(u,v,y,base=None):
            x,z=xz(u,v)
            return (x,(height_at(x,z) if base is None else base)+y,z)

        def footprint(u,v,w,d):
            return Polygon([xz(u+du,v+dv) for du,dv in
                            ((-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2))])

        def prism(material,u,v,w,d,bottom,top,base=None):
            geom=footprint(u,v,w,d)
            if base is None:
                x,z=xz(u,v);base=height_at(x,z)
            extrude(meshes[material],meshes[material],geom,base+bottom,base+top,
                    unit_width=2,unit_height=2)

        def post(material,u,v,bottom,top,radius=.04,base=None):
            beam(meshes[material],point(u,v,bottom,base),point(u,v,top,base),radius)

        def horizontal_beam(material,a,b,y,radius=.035,base=None):
            beam(meshes[material],point(*a,y,base),point(*b,y,base),radius)

        # 主门牌和小型设备色块放入同一张自绘纹理，避免引用实拍或重复材质。
        texture=Image.new('RGB',(1024,256),(227,230,225));draw=ImageDraw.Draw(texture)
        draw.rectangle((4,4,889,251),outline=(118,127,125),width=3)
        small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',46)
        large=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',92)
        draw.text((446,13),'深圳大学',font=small,anchor='mt',fill=(61,69,68))
        draw.text((446,91),config['name'],font=large,anchor='mt',fill=(40,47,46))
        palette={'white':(229,230,222),'red':(184,36,34),'blue':(37,80,117),'green':(30,127,89)}
        swatches={}
        for index,(name,rgb) in enumerate(palette.items()):
            draw.rectangle((896,index*64,1023,index*64+63),fill=rgb)
            uv=(.94,1-(index+.5)*.25);swatches[name]=(uv,uv,uv,uv)
        texture_name=f'gate-{gate_id}-sign.ppm'
        texture.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(assets/texture_name)

        def color_panel(name,corners,normal):
            meshes['sign'].quad(*corners,normal,swatches[name])

        def plate(u,v,w,h,y,base=None):
            # 识别牌面朝校外，字体与安置位置均为示意。
            corners=(point(u+w/2,v,y,base),point(u-w/2,v,y,base),
                     point(u-w/2,v,y+h,base),point(u+w/2,v,y+h,base))
            normal=(-forward[0],0,-forward[1])
            meshes['sign'].quad(*corners,normal,((0,0),(.875,0),(.875,1),(0,1)))

        def rail(a,b):
            # 路边护栏在车道外截断，不横穿 OSM 车行面。
            geom=LineString([xz(*a),xz(*b)]).difference(vehicle_surface.buffer(.25))
            parts=[geom] if geom.geom_type=='LineString' else list(getattr(geom,'geoms',()))
            for line in parts:
                if line.geom_type!='LineString' or line.length<.3:continue
                count=max(1,math.ceil(line.length/2.2))
                points=[line.interpolate(i*line.length/count) for i in range(count+1)]
                for p in points:
                    y=height_at(p.x,p.y)
                    beam(meshes['metal'],(p.x,y,p.y),(p.x,y+1.1,p.y),.035)
                for y in (.55,1.08):
                    for p,q in zip(points,points[1:]):
                        beam(meshes['metal'],(p.x,height_at(p.x,p.y)+y,p.y),
                             (q.x,height_at(q.x,q.y)+y,q.y),.027)

        booth_u,booth_v=config['booth_uv_m'];booth_w,booth_d=config['booth_size_m']
        # 附属岗亭具体点位无实测；仅在近邻无车道覆盖的位置放置。
        initial=booth_u
        direction=1 if booth_u>=0 else -1
        for _ in range(18):
            if footprint(booth_u,booth_v,booth_w+.1,booth_d+.1).intersection(vehicle_surface).area<.02:break
            booth_u+=direction*.5
        else:
            booth_u=initial
            raise ValueError(f'No roadside space for the estimated {gate_id} booth')
        ground=max(height_at(*xz(booth_u+u,booth_v+v)) for u in (-booth_w/2,booth_w/2)
                   for v in (-booth_d/2,booth_d/2))
        low=min(height_at(*xz(booth_u+u,booth_v+v)) for u in (-booth_w/2,booth_w/2)
                for v in (-booth_d/2,booth_d/2))
        prism('stone',booth_u,booth_v,booth_w+.18,booth_d+.18,low-ground-.08,.14,ground)
        # 岗亭四面低墙、外露玻璃分格及一扇带把手的门分别建模。
        for sign in (-1,1):
            prism('stone',booth_u+sign*(booth_w/2-.08),booth_v,.16,booth_d,.14,.93,ground)
        prism('stone',booth_u,booth_v+booth_d/2-.08,booth_w,.16,.14,.93,ground)
        door_w=min(.9,booth_w*.42)
        for sign in (-1,1):
            width=(booth_w-door_w)/2
            prism('stone',booth_u+sign*(door_w/2+width/2),booth_v-booth_d/2+.08,
                  width,.16,.14,.93,ground)
        corners=[(booth_u-booth_w/2,booth_v-booth_d/2),
                 (booth_u+booth_w/2,booth_v-booth_d/2),
                 (booth_u+booth_w/2,booth_v+booth_d/2),
                 (booth_u-booth_w/2,booth_v+booth_d/2)]
        for i,a in enumerate(corners):
            b=corners[(i+1)%4]
            du,dv=b[0]-a[0],b[1]-a[1];length=math.hypot(du,dv)
            normal=np.asarray(point(*b,1,ground))-point(*a,1,ground)
            normal=np.cross((0,1,0),normal);normal/=np.linalg.norm(normal)
            meshes['glass'].quad(point(*a,.94,ground),point(*b,.94,ground),
                                 point(*b,2.64,ground),point(*a,2.64,ground),normal,
                                 ((0,0),(1,0),(1,1),(0,1)))
            divisions=max(1,round(length/1.2))
            for j in range(divisions+1):
                post('metal',a[0]+du*j/divisions,a[1]+dv*j/divisions,.91,2.71,.03,ground)
            for y in (.93,1.78,2.67):horizontal_beam('metal',a,b,y,.028,ground)
        front=booth_v-booth_d/2-.025
        for u in (-door_w/2,door_w/2):post('metal',booth_u+u,front,.17,2.66,.033,ground)
        horizontal_beam('metal',(booth_u-door_w/2,front),(booth_u+door_w/2,front),2.65,.033,ground)
        post('metal',booth_u+door_w*.32,front-.04,1.05,1.36,.025,ground)
        rw,rd=config['roof_size_m']
        prism('metal',booth_u,booth_v,rw-.1,rd-.1,2.76,2.85,ground)
        prism('roof',booth_u,booth_v,rw,rd,2.85,3.08,ground)
        # 檐底窄灯带不发射额外点光，只表示照片常见的灯具构造。
        prism('roof',booth_u,booth_v-rd/2+.25,max(.8,rw-.5),.08,2.73,2.76,ground)
        board_w=min(3.1,rw-.12)
        prism('metal',booth_u,booth_v-rd/2-.04,board_w,.08,3.10,3.91,ground)
        for sign in (-1,1):
            post('metal',booth_u+sign*board_w*.35,booth_v-rd/2,3.03,3.18,.025,ground)
        plate(booth_u,booth_v-rd/2-.085,board_w,.78,3.12,ground)

        for u,v in config['vehicle_controls_uv_m']:
            x,z=xz(u,v);level=height_at(x,z)
            geom=footprint(u,v,.55,.7)
            if geom.intersection(vehicle_surface.buffer(.15)).area>.02:
                skipped+=1;continue
            prism('stone',u,v,.78,2.9,.015,.16,level)
            prism('metal',u,v,.38,.52,.16,1.22,level)
            # 蓝色识别面板与抬起的红白道闸，杆在竖直位置保留车道净空。
            color_panel('blue',(point(u+.22,v-.275,.30,level),point(u-.22,v-.275,.30,level),
                               point(u-.22,v-.275,1.15,level),point(u+.22,v-.275,1.15,level)),
                        (-forward[0],0,-forward[1]))
            post('roof',u,v+.43,1.0,3.85,.045,level)
            for k in range(3):
                y=1.1+k*.9
                color_panel('red',(point(u+.052,v+.377,y,level),point(u-.052,v+.377,y,level),
                                   point(u-.052,v+.377,y+.34,level),point(u+.052,v+.377,y+.34,level)),
                            (-forward[0],0,-forward[1]))
            post('metal',u+.45,v-.6,.12,2.72,.026,level)
            prism('roof',u+.45,v-.60,.22,.24,2.67,2.90,level)

        pu,pv=config['pedestrian_uv_m']
        # 人行设备的通道保持空隙；其数量与精确位置均未逐项核实。
        for u in (pu-.68,pu+.68):
            if footprint(u,pv,.27,1.5).intersects(vehicle_surface.buffer(.15)):
                skipped+=1;continue
            prism('metal',u,pv,.27,1.5,.05,1.03)
            prism('roof',u,pv,.28,.42,1.03,1.10)
            color_panel('green',(point(u+.12,pv-.755,.65),point(u-.12,pv-.755,.65),
                                 point(u-.12,pv-.755,.98),point(u+.12,pv-.755,.98)),
                        (-forward[0],0,-forward[1]))
        outer=config['fence_outer_u_m']
        rail((-outer-5,0),(-outer,0));rail((outer,0),(outer+5,0))
        for u in (-outer,outer):
            if footprint(u,0,.55,.55).intersects(vehicle_surface.buffer(.1)):continue
            prism('stone',u,0,.55,.55,-.06,1.35)
            prism('roof',u,0,.64,.64,1.35,1.44)

        if config['style']=='bridge_forecourt':
            # 致远门保留旧模型前场范围，石板分缝与地形相贴，车辆仍可通行。
            width,depth=config['forecourt_size_m'];step=1.0
            for u in np.arange(-width/2,width/2-.01,step):
                for v in np.arange(-depth/2,depth/2-.01,step):
                    u1,v1=min(u+step,width/2),min(v+step,depth/2)
                    corners2=[point(u+.012,v+.012,.045),point(u1-.012,v+.012,.045),
                              point(u1-.012,v1-.012,.045),point(u+.012,v1-.012,.045)]
                    normal=np.cross(np.asarray(corners2[1])-corners2[0],np.asarray(corners2[2])-corners2[0])
                    normal/=max(np.linalg.norm(normal),.001)
                    if normal[1]<0:normal=-normal
                    meshes['stone'].quad(*corners2,normal,((0,0),(1,0),(1,1),(0,1)))
            # 降下的防撞柱表示2026年安装通知中的设施类别，数量仅为近似。
            for u in np.linspace(-3.45,3.45,7):
                prism('metal',float(u),5,.16,.16,.04,.13)
        elif config['style']=='split_east_entry':
            # 致诚门两条车行线之间的浅色基座，不制作横跨车道的无证据门楼。
            prism('stone',0,.25,3.5,6.5,-.05,.12)
        elif config['style']=='south_road_entry':
            # 南向入口的短翼墙顺边界延伸，中央服务道路留空。
            for sign in (-1,1):
                prism('stone',sign*10.5,.4,3.5,.65,-.08,.65)

        layers=[]
        for material,mesh in meshes.items():
            if len(mesh.vertices)<24:continue
            filename=f'campus-gate-{gate_id}-{material}.mesh'
            count=mesh.save(assets/filename)
            layers.append({'material':material,'mesh':filename,'color':colors[material],
                           'vertices':count})
        eye_x,eye_z=xz(-4.0,-19.0);target_x,target_z=xz(0,2.0)
        eye_y=max(height_at(eye_x,eye_z)+2.0,height_at(cx,cz)+2.5)
        notes=config['notes']
        if abs(booth_u-initial)>.01:notes+=f' 为避开已建车道，估计岗亭沿侧向调整{abs(booth_u-initial):.1f}米。'
        records.append({'id':gate_id,'name':config['name'],'center_xz_m':[cx,cz],
                        'view_eye':[eye_x,eye_y,eye_z],
                        'view_target':[target_x,height_at(target_x,target_z)+1.6,target_z],
                        'layers':layers,'sign_texture':texture_name,'notes':notes,
                        'booth_position_xz_m':list(xz(booth_u,booth_v)),
                        'skipped_road_conflicts':skipped})
    return {'gates':records}
