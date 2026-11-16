"""粤海校区西侧入口的独立模型。

立德门依据校方照片表现双高墙、水平细缝和悬挑岗亭；立业门使用地图岗亭
轮廓，门禁设备尺寸为估计。立志门资料不足，明确只建立位置参考牌。
照片仅用于观察，不作为发布纹理；中文字形为系统字体重绘。
"""
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _atlas(assets, ident, name, *, address=False, location_only=False):
    image=Image.new('RGB',(512,768),(232,233,223))
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/STXINGKA.TTF',148)
    for index,char in enumerate('深圳大学'):
        draw.text((100,94+index*176),char,font=font,fill=(33,38,36),anchor='mm')
    draw.rectangle((220,0,511,160),fill=(28,89,137))
    draw.rectangle((227,7,504,153),outline=(238,244,239),width=4)
    draw.text((366,80),'3688' if address else name,
              font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',75 if address else 67),
              fill=(242,247,242),anchor='mm')
    draw.text((366,255),name,font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',66),
              fill=(41,52,49),anchor='mm')
    draw.text((366,316),'位置参考' if location_only else '深圳大学',
              font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',35),
              fill=(87,103,94),anchor='mm')
    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(assets/f'gate-{ident}-sign.ppm')
    # 返回原始画布区域对应的 OpenGL 纹理范围。
    def uv(rect):
        x0,y0,x1,y1=rect
        return ((x0/512,1-y1/768),(x1/512,1-y1/768),
                (x1/512,1-y0/768),(x0/512,1-y0/768))
    return {'school':uv((0,0,200,740)), 'address':uv((220,0,512,160)),
            'name':uv((220,184,512,350))}


class _Gate:
    def __init__(self,config,mesh_type,height_at,beam):
        self.config=config
        self.cx,self.cz=config['center_xz_m']
        angle=math.radians(config['inward_direction_degrees'])
        self.u=np.asarray((math.cos(angle),math.sin(angle)))
        self.v=np.asarray((-math.sin(angle),math.cos(angle)))
        self.height_at=height_at;self.beam=beam
        self.meshes={name:mesh_type() for name in
                     ('stone','metal','metal-red','metal-yellow','glass','roof','sign')}
        self.footprints=[]

    def xz(self,u,v):
        point=np.asarray((self.cx,self.cz))+u*self.u+v*self.v
        return tuple(map(float,point))

    def point(self,u,v,y):
        x,z=self.xz(u,v)
        return (x,float(y),z)

    def ground(self,u=0,v=0):
        return float(self.height_at(*self.xz(u,v)))

    def quad(self,material,points,uv=None):
        normal=np.cross(np.asarray(points[1])-points[0],np.asarray(points[2])-points[0])
        normal/=max(float(np.linalg.norm(normal)),.000001)
        self.meshes[material].quad(*points,normal,
             uv or ((0,0),(1,0),(1,1),(0,1)))

    def box(self,u,v,low,high,width,depth,material='stone',base=None,keep_clear=False):
        base=self.ground(u,v) if base is None else base
        a,b=u-width*.5,u+width*.5;c,d=v-depth*.5,v+depth*.5
        p=lambda x,z,y:self.point(x,z,base+y)
        # 六面封闭，包括悬挑檐口可见的底面。
        for points in ((p(a,c,low),p(a,d,low),p(a,d,high),p(a,c,high)),
                       (p(b,d,low),p(b,c,low),p(b,c,high),p(b,d,high)),
                       (p(b,c,low),p(a,c,low),p(a,c,high),p(b,c,high)),
                       (p(a,d,low),p(b,d,low),p(b,d,high),p(a,d,high)),
                       (p(a,c,high),p(a,d,high),p(b,d,high),p(b,c,high)),
                       (p(a,d,low),p(a,c,low),p(b,c,low),p(b,d,low))):
            self.quad(material,points)
        if keep_clear:
            self.footprints.append([self.xz(a,c),self.xz(b,c),self.xz(b,d),self.xz(a,d)])

    def banded_box(self,u,v,height,width,depth,base):
        low=min(self.ground(u+du*width*.5,v+dv*depth*.5)
                for du in (-1,1) for dv in (-1,1))-base-.12
        self.box(u,v,low,height,width,depth,base=base,keep_clear=True)
        # 实景门柱上的细密水平缝由几何线带表现，避免近景仅有整块灰墙。
        for y in np.arange(.22,height,.24):
            for sign in (-1,1):
                self.box(u+sign*(width*.5+.009),v,y,y+.025,.015,depth+.016,'metal',base)
                self.box(u,v+sign*(depth*.5+.009),y,y+.025,width+.016,.015,'metal',base)

    def sign(self,u,v,bottom,height,width,uv,base,*,reverse=False):
        points=(self.point(u,v-width*.5,base+bottom),self.point(u,v+width*.5,base+bottom),
                self.point(u,v+width*.5,base+bottom+height),self.point(u,v-width*.5,base+bottom+height))
        if reverse:
            # 背面另建朝外的四边形；保持 UV 从观察者左下角开始，避免文字镜像。
            points=(points[1],points[0],points[3],points[2])
        self.quad('sign',points,uv)

    def pole(self,u,v,bottom,top,radius=.035,base=None):
        base=self.ground(u,v) if base is None else base
        self.beam(self.meshes['metal'],self.point(u,v,base+bottom),self.point(u,v,base+top),radius)

    def cylinder(self,u,v,bottom,top,radius,material='metal',segments=16):
        base=self.ground(u,v)
        for index in range(segments):
            a,b=index*math.tau/segments,(index+1)*math.tau/segments
            ua,va=u+math.cos(a)*radius,v+math.sin(a)*radius
            ub,vb=u+math.cos(b)*radius,v+math.sin(b)*radius
            low_a,low_b=self.point(ua,va,base+bottom),self.point(ub,vb,base+bottom)
            high_a,high_b=self.point(ua,va,base+top),self.point(ub,vb,base+top)
            self.quad(material,(low_a,high_a,high_b,low_b))
            self.meshes[material].triangle(self.point(u,v,base+top),high_b,high_a,(0,1,0),
                                         ((.5,.5),(1,0),(0,0)))

    def booth(self,u,v,width,depth,*,base=None):
        base=self.ground(u,v) if base is None else base
        low=min(self.ground(u+du*width*.5,v+dv*depth*.5)
                for du in (-1,1) for dv in (-1,1))-base-.12
        self.box(u,v,low,.87,width,depth,'stone',base,True)
        self.box(u,v,.90,2.72,width-.10,depth-.10,'glass',base)
        self.box(u,v,2.77,2.94,width,depth,'stone',base)
        self.box(u,v,2.94,3.17,width+.55,depth+.55,'roof',base)
        self.box(u,v,2.89,2.96,width+.40,depth+.40,'metal',base)
        # 真实照片可辨的玻璃分格和推拉窗；分格数量、窗台高度为比例估计。
        for x in (-.5,.0,.5):
            self.box(u+x*(width-.12),v-depth*.5-.02,.88,2.78,.055,.055,'metal',base)
            self.box(u+x*(width-.12),v+depth*.5+.02,.88,2.78,.055,.055,'metal',base)
        for z in (-.5,.0,.5):
            self.box(u-width*.5-.02,v+z*(depth-.12),.88,2.78,.055,.055,'metal',base)
            self.box(u+width*.5+.02,v+z*(depth-.12),.88,2.78,.055,.055,'metal',base)
        for y in (.92,1.78,2.74):
            self.box(u,v-depth*.5-.025,y,y+.045,width+.04,.055,'metal',base)
            self.box(u,v+depth*.5+.025,y,y+.045,width+.04,.055,'metal',base)
            self.box(u-width*.5-.025,v,y,y+.045,.055,depth+.04,'metal',base)
            self.box(u+width*.5+.025,v,y,y+.045,.055,depth+.04,'metal',base)
        # 门把手与窗台下的内侧工作台只作几何尺度示意。
        self.box(u-width*.5-.07,v+depth*.24,1.14,1.19,.08,.30,'metal',base)
        self.box(u,v,1.03,1.13,width*.72,depth*.55,'roof',base)

    def raised_barrier(self,u,v,base=None,body_material='metal'):
        base=self.ground(u,v) if base is None else base
        self.box(u,v,.05,.98,.43,.42,body_material,base)
        self.box(u,v,.98,1.04,.50,.47,'roof',base)
        self.box(u,v,1.0,3.8,.095,.095,'roof',base)
        for y in (1.35,2.20,3.05):
            self.box(u-.004,v,y,y+.34,.108,.108,'metal',base)
        self.pole(u+.65,v,.05,2.55,.035,base)
        self.box(u+.65,v-.06,2.45,2.65,.24,.18,'metal',base)

    def rail(self,u0,v0,u1,v1):
        distance=math.hypot(u1-u0,v1-v0)
        steps=max(1,math.ceil(distance/1.8))
        for i in range(steps+1):
            t=i/steps;u=u0+(u1-u0)*t;v=v0+(v1-v0)*t
            self.pole(u,v,0,1.08)
        for y in (.55,1.04):
            for i in range(steps):
                t0,t1=i/steps,(i+1)/steps
                ua,va=u0+(u1-u0)*t0,v0+(v1-v0)*t0
                ub,vb=u0+(u1-u0)*t1,v0+(v1-v0)*t1
                self.beam(self.meshes['metal'],self.point(ua,va,self.ground(ua,va)+y),
                          self.point(ub,vb,self.ground(ub,vb)+y),.025)


def _lide(gate,uv):
    # 与原 C++ 立德门坐标一致，保持两条入口车道及长草坪的关系。
    for side in (-1,1):
        v=side*13.0;base=gate.ground(0,v)
        if side<0:
            gate.banded_box(0,v-1.8,12.4,4.5,2.3,base)
            gate.banded_box(1.2,v+1.8,11.4,4.5,2.3,base)
            gate.box(.6,v,5.80,7.00,5.7,5.8,base=base)
        else:
            gate.banded_box(0,v-.9,13.0,4.5,4.4,base)
            gate.banded_box(2.1,v+2.0,12.0,4.5,3.0,base)
            gate.sign(-2.285,v-.25,.28,3.05,1.12,uv['school'],base)
            gate.sign(-2.30,v+2.10,2.30,.49,.96,uv['address'],base)
        booth_v=side*17.3
        gate.booth(3.3,booth_v,7.6,7.0,base=base)
        # 既有照片显示门柱与岗亭上方有向车道挑出的薄檐。
        gate.box(2.2,side*16.5,3.16,3.60,10.0,10.5,'roof',base)
        gate.box(2.2,side*16.5,3.05,3.17,9.80,10.30,'metal',base)
        for u in (-1.3,1.6,4.5):
            gate.box(u,side*12.0,3.03,3.08,.42,.16,'roof',base)
        # 人行闸机放在车行边线外，默认开放；尺寸和现行设备待核实。
        for v_lane in (side*8.30,side*9.75):
            ground=gate.ground(-.9,v_lane)
            gate.box(-.9,v_lane,0,1.00,1.45,.22,'metal',ground)
            gate.box(-1.20,v_lane,1.00,1.08,.52,.27,'roof',ground)
            gate.box(-1.38,v_lane,.55,1.13,.10,.13,'glass',ground)
    island=gate.ground(5.5,0)
    gate.box(5.5,0,0,.28,6.8,1.85,'stone',island,True)
    gate.box(6.9,0,.28,1.03,1.65,1.30,'stone',island)
    gate.box(6.9,0,1.03,1.93,1.40,1.18,'glass',island)
    gate.box(6.9,0,1.93,2.08,1.80,1.50,'roof',island)
    for side in (-1,1):gate.box(6.9,side*.59,1.03,1.93,.055,.055,'metal',island)
    for side in (-1,1):gate.raised_barrier(4.0,side*.77,island)


def _liye(gate,uv):
    # 岗亭位置与轮廓来自 OSM way/241588751；不扩建成没有证据的大门楼。
    u,v=gate.config['booth_center_uv_m']
    width,depth=gate.config['booth_size_uv_m'];base=gate.ground(u,v)
    gate.booth(u,v,width,depth,base=base)
    gate.sign(u-width*.5-.052,v,.96,1.22,1.70,uv['name'],base)
    gate.raised_barrier(-.5,-2.0,body_material='metal-red')
    # 2026 年校方门口监控中可辨的酒红色门禁立柱及白色识别面。
    base=gate.ground(1.0,-2.10)
    gate.box(1.0,-2.10,0,1.72,.32,.38,'metal-red',base)
    gate.box(.826,-2.10,1.20,1.54,.025,.25,'roof',base)
    gate.box(.806,-2.10,1.30,1.46,.022,.18,'glass',base)
    # 黄黑短柱与收拢的伸缩门均保持在路边，通行车道不设关闭的门扇。
    for u_post in (-3.2,-1.9,-.6,.7):
        gate.cylinder(u_post,-2.80,0,.76,.10,'metal-yellow')
        for y in (.21,.48):gate.cylinder(u_post,-2.80,y,y+.13,.103,'metal')
    for u_face in (-1.15,-.72):
        for index in range(4):
            v0,v1=5.25+index*.62,5.25+(index+1)*.62
            for va,vb in ((v0,v1),(v1,v0)):
                gate.beam(gate.meshes['metal'],
                    gate.point(u_face,va,gate.ground(u_face,va)+.22),
                    gate.point(u_face,vb,gate.ground(u_face,vb)+1.32),.024)
            gate.pole(u_face,v0,.12,1.43,.025)
        gate.rail(u_face,5.25,u_face,7.73)
    # 监控可见黄黑减速带；宽高及沿线位置仍按道路尺度近似。
    profile=((-0.28,.018),(-.20,.068),(-.10,.10),(.10,.10),(.20,.068),(.28,.018))
    for stripe in range(8):
        va,vb=-2.6+stripe*.65,-2.6+(stripe+1)*.65
        material='metal-yellow' if stripe%2==0 else 'metal'
        for (ua,ha),(ub,hb) in zip(profile,profile[1:]):
            ua-=4.7;ub-=4.7
            gate.quad(material,(gate.point(ua,va,gate.ground(ua,va)+ha),
                                gate.point(ua,vb,gate.ground(ua,vb)+ha),
                                gate.point(ub,vb,gate.ground(ub,vb)+hb),
                                gate.point(ub,va,gate.ground(ub,va)+hb)))
    # 较窄街口的低栏杆及独立人行通道；不跨越现有机动车路心。
    gate.rail(-1.0,-12,-1.0,-8.5)
    gate.rail(-1.0,5.0,-1.0,10.0)
    for lane_v in (-7.7,-6.0):
        ground=gate.ground(-.4,lane_v)
        gate.box(-.4,lane_v,0,1.0,1.35,.24,'metal',ground)
        gate.box(-.9,lane_v,1.00,1.09,.46,.30,'roof',ground)
        gate.box(-1.02,lane_v,.50,1.12,.10,.16,'glass',ground)
    # 保留立业门的通透入口特征；小型门名牌是定位示意，不声称复刻现状。
    base=gate.ground(-.3,10.6)
    gate.box(-.3,10.6,-.08,2.05,.45,1.60,'stone',base,True)
    gate.sign(-.54,10.6,.65,1.12,1.45,uv['name'],base)


def _lizhi(gate,uv):
    # 该入口未取得足够地面照片，明确使用位置参考牌，避免捏造永久门楼。
    base=gate.ground(0,0)
    gate.pole(0,-1.10,0,2.4,.045,base)
    gate.pole(0,1.10,0,2.4,.045,base)
    gate.box(0,0,1.05,2.35,.10,2.65,'stone',base)
    gate.sign(-.056,0,1.12,1.16,2.50,uv['name'],base)
    gate.sign(.056,0,1.12,1.16,2.50,uv['name'],base,reverse=True)


def make_gates(*,mesh_type,height_at,beam,extrude,assets,data,vehicle_surface):
    """生成各门独立资源，由主生成器统一接入；不改动道路和全校地形。"""
    configs=json.loads((data/'gates-west.json').read_text(encoding='utf8'))['gates']
    result=[]
    colors={'stone':[.69,.71,.68],'metal':[.30,.35,.35],
            'metal-red':[.48,.15,.16],'metal-yellow':[.95,.76,.09],
            'glass':[.61,.75,.75],'roof':[.85,.87,.83],'sign':[1,1,1]}
    for config in configs:
        gate=_Gate(config,mesh_type,height_at,beam)
        uv=_atlas(assets,config['id'],config['name'],address=config['id']=='lide',
                  location_only=config.get('model_status')=='location_marker')
        {'lide':_lide,'liye':_liye,'lizhi':_lizhi}[config['id']](gate,uv)
        layers=[]
        for material,mesh in gate.meshes.items():
            if not len(mesh.vertices):continue
            filename=f"campus-gate-{config['id']}-{material}.mesh"
            count=mesh.save(assets/filename)
            layers.append({'material':'metal' if material.startswith('metal-') else material,
                           'mesh':filename,'color':colors[material],
                           'vertices':count})
        cx,cz=config['center_xz_m'];ground=height_at(cx,cz)
        eye=config['view_eye_offset'];target=config['view_target_offset']
        eye_y=max(ground+eye[1],height_at(cx+eye[0],cz+eye[2])+1.8)
        result.append({'id':config['id'],'name':config['name'],'center_xz_m':[cx,cz],
            'view_eye':[cx+eye[0],eye_y,cz+eye[2]],
            'view_target':[cx+target[0],ground+target[1],cz+target[2]],
            'layers':layers,'sign_texture':f"gate-{config['id']}-sign.ppm",
            'model_status':config['model_status'],'notes':config['notes'],
            'vegetation_keep_clear':gate.footprints,
            'exclude_osm_buildings':config.get('exclude_osm_buildings',[])})
    return {'gates':result}
