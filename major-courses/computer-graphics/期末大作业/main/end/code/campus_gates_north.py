"""立功、立心、立言门的独立网格。

校门位置使用现有地图坐标，外观以逐门照片为依据；照片不能给出的尺寸
仍在 data/gates-north.json 与来源文件中明确记录为估计。网格均使用世界坐标。
"""
import json
import math
import random

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon


def _sign_atlas(path, name):
    """文字与色块由本机字体重绘，不使用实景照片或真实二维码。"""
    image=Image.new('RGB',(512,768),(235,236,227))
    draw=ImageDraw.Draw(image)
    bold=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',58)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27)
    english=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',23)
    draw.rectangle((0,0,511,189),fill=(234,235,224))
    draw.text((256,56),'深圳大学',font=bold,anchor='mm',fill=(42,47,47))
    draw.text((256,129),name,font=small,anchor='mm',fill=(68,73,71))
    draw.rectangle((0,204,247,387),fill=(30,85,147))
    draw.text((124,252),'通行设备',font=small,anchor='mm',fill='white')
    draw.text((124,302),'ACCESS',font=english,anchor='mm',fill='white')
    draw.line((36,362,210,362),fill=(228,234,232),width=4)
    draw.rectangle((264,204,511,387),fill=(235,237,229))
    draw.text((388,247),'南海大道',font=small,anchor='mm',fill=(36,42,43))
    numbers=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',66)
    draw.text((388,324),'3688',font=numbers,anchor='mm',fill=(30,35,36))
    draw.rectangle((0,404,247,587),fill=(27,102,65))
    draw.text((124,465),'出口',font=bold,anchor='mm',fill=(238,244,232))
    draw.text((124,535),'EXIT',font=english,anchor='mm',fill=(238,244,232))
    draw.rectangle((0,596,247,651),fill=(213,180,75))
    # 圆形徽记只作易识别的字体示意，不冒充校徽的精确复刻。
    draw.rectangle((264,404,511,651),fill=(237,238,230))
    draw.ellipse((280,420,495,635),outline=(159,64,69),width=5)
    draw.ellipse((289,429,486,626),outline=(159,64,69),width=2)
    letters=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',58)
    draw.text((388,518),'SZU',font=letters,anchor='mm',fill=(159,64,69))
    draw.text((388,568),'深圳大学',font=small,anchor='mm',fill=(159,64,69))
    draw.rectangle((264,652,511,667),fill=(178,36,38))
    draw.rectangle((0,668,247,767),fill=(22,69,125))
    draw.rectangle((264,668,511,767),fill=(48,90,40))
    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path)


class _Gate:
    def __init__(self, config, mesh_type, height_at, beam, extrude):
        self.config=config
        self.layers={key:mesh_type() for key in ('stone','metal','glass','roof','sign')}
        self.height_at=height_at
        self.beam_fn=beam
        self.extrude_fn=extrude
        self.cx,self.cz=config['center_xz_m']
        self.ux,self.uz=config['across_xz']
        self.vx,self.vz=config['outward_xz']
        self.level=height_at(self.cx,self.cz)

    def point(self,x,y,z):
        return (self.cx+x*self.ux+z*self.vx,self.level+y,self.cz+x*self.uz+z*self.vz)

    def ground(self,x,z):
        wx,_,wz=self.point(x,0,z)
        return self.height_at(wx,wz)-self.level

    def block(self,material,x,z,width,depth,bottom,top):
        p=[self.point(x+a,bottom,z+b) for a,b in
           ((-width/2,-depth/2),(width/2,-depth/2),(width/2,depth/2),(-width/2,depth/2))]
        geom=Polygon((q[0],q[2]) for q in p)
        target=self.layers[material]
        self.extrude_fn(target,target,geom,self.level+bottom,self.level+top)

    def line(self,material,a,b,radius):
        self.beam_fn(self.layers[material],self.point(*a),self.point(*b),radius)

    def cylinder(self,material,x,z,radius,bottom,top,segments=48):
        pts=[self.point(x+radius*math.cos(i*2*math.pi/segments),bottom,
                        z+radius*math.sin(i*2*math.pi/segments)) for i in range(segments)]
        geom=Polygon((p[0],p[2]) for p in pts)
        self.extrude_fn(self.layers[material],self.layers[material],geom,
                        self.level+bottom,self.level+top)

    def panel(self,x,z,width,bottom,top,box=(0,0,512,190),facing=1):
        u0,u1=box[0]/512,box[2]/512
        v0,v1=1-box[3]/768,1-box[1]/768
        pts=[self.point(x-width/2,bottom,z),self.point(x+width/2,bottom,z),
             self.point(x+width/2,top,z),self.point(x-width/2,top,z)]
        normal=(self.vx*facing,0,self.vz*facing)
        uv=[(u0,v0),(u1,v0),(u1,v1),(u0,v1)]
        if (-self.uz*self.vx+self.ux*self.vz)*facing<0:
            # 左手局部基或背面牌的横轴与观看者右向相反，文字也须翻转映射。
            uv=[(u1,v0),(u0,v0),(u0,v1),(u1,v1)]
            pts=pts[::-1];uv=uv[::-1]
        self.layers['sign'].quad(*pts,normal,uv)

    def blue_block(self,x,z,width,depth,bottom,top):
        # 设备蓝色不与金属构件混为同一材质，用程序色块包覆四侧。
        points=[self.point(x+a,bottom,z+b) for a,b in
                ((-width/2,-depth/2),(width/2,-depth/2),(width/2,depth/2),(-width/2,depth/2))]
        for a,b in zip(points,points[1:]+points[:1]):
            dx,dz=b[0]-a[0],b[2]-a[2];length=math.hypot(dx,dz)
            self.layers['sign'].quad(a,b,(b[0],self.level+top,b[2]),(a[0],self.level+top,a[2]),
                (dz/length,0,-dx/length),((.05,.03),(.45,.03),(.45,.10),(.05,.10)))
        self.block('metal',x,z,width,depth,top,top+.035)

    def railing(self,x,z0,z1,bottom,height=1.05,mesh=False):
        count=max(1,math.ceil(abs(z1-z0)/1.45))
        for i in range(count+1):
            z=z0+(z1-z0)*i/count
            self.line('metal',(x,bottom,z),(x,bottom+height,z),.027)
        for y in (bottom+.12,bottom+height):
            self.line('metal',(x,y,z0),(x,y,z1),.029)
        if mesh:
            for y in (.30,.48,.66,.84):
                self.line('metal',(x,bottom+y,z0),(x,bottom+y,z1),.008)
            count=max(1,math.ceil(abs(z1-z0)/.22))
            for i in range(count+1):
                z=z0+(z1-z0)*i/count
                self.line('metal',(x,bottom+.12,z),(x,bottom+height,z),.007)


def _ligong(g):
    # 岗亭留在机动车道东侧，分段圆弧玻璃及框架来自2020年入口近景。
    wx,wz=g.config['booth_center_xz_m']
    bx,bz=wx-g.cx,g.cz-wz
    ground=g.height_at(wx,wz)-g.level
    for material,radius,low,high in [('stone',3.2,0,.18),('stone',2.94,.18,1.37),
                                      ('glass',2.94,1.37,3.06),('roof',3.24,3.08,3.27)]:
        g.cylinder(material,bx,bz,radius,ground+low,ground+high)
    for y in (1.37,2.10,3.06):g.cylinder('metal',bx,bz,2.97,ground+y-.025,ground+y+.025)
    for i in range(20):
        angle=i*2*math.pi/20
        x,z=bx+2.97*math.cos(angle),bz+2.97*math.sin(angle)
        g.line('metal',(x,ground+1.38,z),(x,ground+3.05,z),.032)
    # 门框、门把手和浅台阶不靠近机动车道；真实开门方向未确认。
    g.block('stone',bx,bz-3.05,1.4,.50,ground,ground+.13)
    for x in (bx-.58,bx+.58):g.line('metal',(x,ground+.15,bz-2.96),(x,ground+2.91,bz-2.96),.034)
    g.line('metal',(bx+.37,ground+1.1,bz-3.01),(bx+.37,ground+1.47,bz-3.01),.025)
    # 三座闸机塔体和两条开敞通路；小臂收向侧边以保留自由探索的净空。
    for x in (7.9,9.5,11.1):
        z=-8.9;low=g.ground(x,z)
        g.block('metal',x,z,.28,1.28,low+.06,low+.96)
        g.block('roof',x,z,.34,.55,low+.98,low+1.06)
        g.blue_block(x,z+.65,.19,.035,low+.40,low+.87)
        g.line('metal',(x,low+.75,z+.28),(x+.21,low+.75,z+.77),.022)
    for x in (7.4,11.55):
        g.railing(x,-10.8,-7.0,g.ground(x,-8.9),1.06)
    # 右侧低矮标识只概括校门名称，字体和现时牌面不是实测复刻。
    g.block('metal',12.05,-8.6,.12,.12,ground,ground+1.92)
    g.panel(12.05,-8.53,1.02,ground+1.10,ground+1.70)


def _liyan(g):
    # 校方交通指南可辨中央岗亭、两侧黄黑分流岛及蓝色设备。
    g.block('stone',0,-1.0,1.8,7.0,0,.23)
    g.block('stone',0,-1.2,1.60,1.86,.23,1.00)
    g.block('glass',0,-1.2,1.57,1.83,1.00,2.02)
    for x in (-.79,.79):
        for z in (-2.12,-.28):g.line('metal',(x,.98,z),(x,2.04,z),.023)
    g.line('metal',(-.8,1.55,-.25),(.8,1.55,-.25),.023)
    g.block('roof',0,-1.2,1.98,2.24,2.04,2.23)
    g.block('metal',0,-1.2,1.94,2.20,2.015,2.055)
    g.panel(0,-.26,1.15,.37,.83)
    for side in (-1,1):
        x=side*5.1
        g.block('stone',x,0,1.03,5.8,0,.22)
        points=[g.point(xx,.223,z) for xx,z in
                ((x-.52,-2.9),(x+.52,-2.9),(x+.52,2.9),(x-.52,2.9))]
        g.layers['sign'].quad(*points,(0,1,0),((.02,.16),(.45,.16),(.45,.22),(.02,.22)))
        # 交替短条沿照片中交通岛的两侧表现黄黑漆边，不跨越车道。
        for i in range(7):
            z=-2.5+i*.8
            if i%2==0:g.block('metal',x,z,1.05,.35,.22,.245)
        g.blue_block(x,-.55,.54,.66,.24,1.42)
        g.panel(x,-.20,.45,.70,1.28,(0,204,248,388))
        g.line('metal',(x,.23,-1.25),(x,3.00,-1.25),.038)
        g.block('roof',x,-1.25,.27,.23,2.83,3.17)
        g.line('roof',(x,.24,.85),(x,3.08,.85),.043)
        # 道闸以竖起状态表达，不声称实时开放状态。
        for y in (1.0,1.83,2.66):g.panel(x,.90,.105,y,y+.29,(264,652,512,668))
    # 两侧人行道保留路宽之外的通过空间，高杆按照片近似。
    for side in (-1,1):
        x=side*8.4
        low=g.ground(x,0)
        g.line('metal',(x,low,0),(x,low+3.1,0),.036)
        g.panel(x,.05,.75,low+1.45,low+2.70,(0,0,512,190))


def _lixin(g, mesh_type):
    # 门区白灰塔体、低檐和内部台阶取自2023年小东门重新开放的同组照片。
    # 粗DEM无法确定真实上下平台高程，因此这是保留照片形态的独立近似。
    floor=g.ground(0,1.6)+.04
    entry=floor+.45
    walk=mesh_type()
    def walking_quad(x0,x1,z0,z1,y0,y1):
        points=[g.point(x0,y0,z0),g.point(x1,y0,z0),g.point(x1,y1,z1),g.point(x0,y1,z1)]
        a,b,c=points[:3]
        u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
        n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
        length=math.sqrt(sum(q*q for q in n));n=[q/length for q in n]
        if n[1]<0:n=[-q for q in n]
        walk.quad(*points,n,((0,0),(1,0),(1,1),(0,1)))
    def paved_transition(half_width,za,zb,inner_y,outer_y=None):
        # 两端补偿粗地形的横坡；长度与铺装范围为估计，不代表实测无障碍坡道。
        nx=max(1,math.ceil(2*half_width/.4));nz=max(1,math.ceil(abs(zb-za)/.3))
        grid=[]
        for j in range(nz+1):
            t=j/nz;z=za+(zb-za)*t;row=[]
            for i in range(nx+1):
                x=-half_width+2*half_width*i/nx
                end=g.ground(x,zb)+.02 if outer_y is None else outer_y
                y=max(g.ground(x,z)+.02,inner_y*(1-t)+end*t)
                row.append((x,y,z))
            grid.append(row)
        for j in range(nz):
            for i in range(nx):
                corners=(grid[j][i],grid[j][i+1],grid[j+1][i+1],grid[j+1][i])
                for indices in ((0,1,2),(0,2,3)):
                    local=[corners[k] for k in indices];points=[g.point(*p) for p in local]
                    a,b,c=points;u=[b[k]-a[k] for k in range(3)];v=[c[k]-a[k] for k in range(3)]
                    n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                    if n[1]<0:points[1],points[2]=points[2],points[1];local[1],local[2]=local[2],local[1];n=[-q for q in n]
                    length=math.sqrt(sum(q*q for q in n));n=[q/length for q in n]
                    uv=tuple((p[0]/2,p[2]/2) for p in local)
                    g.layers['stone'].triangle(*points,n,uv)
                    walk.triangle(*points,n,uv)
        # 接地裙边随地面变化，封住坡板及其与台阶相接处的悬空底边。
        edges=(grid[0],grid[-1],[row[0] for row in grid],[row[-1] for row in grid])
        for edge in edges:
            for a,b in zip(edge,edge[1:]):
                pa,pb=g.point(*a),g.point(*b)
                ba=g.point(a[0],g.ground(a[0],a[2])-.08,a[2])
                bb=g.point(b[0],g.ground(b[0],b[2])-.08,b[2])
                dx,dz=pb[0]-pa[0],pb[2]-pa[2];length=math.hypot(dx,dz)
                normal=(-dz/length,0,dx/length);points=[ba,bb,pb,pa]
                mx=(a[0]+b[0])/2;mz=(a[2]+b[2]-za-zb)/2
                if normal[0]*(mx*g.ux+mz*g.vx)+normal[2]*(mx*g.uz+mz*g.vz)<0:
                    points=points[::-1];normal=tuple(-q for q in normal)
                g.layers['stone'].quad(*points,normal,((0,0),(1,0),(1,1),(0,1)))
    for i in range(3):
        z=1.18-i*.55
        bottom=min(g.ground(x,zz) for x in (-4.7,-2.35,0,2.35,4.7) for zz in (z-.28,z,z+.28))-.10
        top=floor+.15*(i+1)
        if i==0:
            # 第一级仅将高侧埋入地面的角点贴地抬起，主体平面和台阶占地保持不变。
            paved_transition(4.7,z-.28,z+.28,top,outer_y=top)
        else:
            g.block('stone',0,z,9.4,.56,bottom,top)
            walking_quad(-4.7,4.7,z-.28,z+.28,top,top)
    paved_transition(4.7,1.46,3.6,floor+.15)
    bottom=min(g.ground(x,z) for x in (-5.1,-2.55,0,2.55,5.1) for z in (-4,-2,0))-.10
    g.block('stone',0,-2.0,10.2,4.0,bottom,entry)
    walking_quad(-5.1,5.1,-4,0,entry,entry)
    # 两边的实体体量不封住中央入口；照片所见的校徽以圆形字体示意。
    g.block('stone',4.85,-3.6,3.10,5.5,g.ground(4.85,-3.6)-.10,entry+7.10)
    g.block('stone',-5.0,-4.6,1.05,4.5,g.ground(-5.0,-4.6)-.10,entry+8.25)
    g.block('stone',-5.0,-.6,.40,2.0,entry,entry+3.36)
    g.block('roof',0,-.95,13.0,4.1,entry+3.34,entry+3.62)
    g.block('metal',0,-.95,12.95,4.03,entry+3.31,entry+3.35)
    g.panel(4.85,-.79,1.45,entry+5.05,entry+6.50,(264,404,512,652))
    # 黑边长窗和白色楼上体量从照片概括，不把此门当成宏大跨路门楼。
    g.block('glass',0,-4.1,5.7,.10,entry+4.25,entry+6.82)
    g.block('roof',0,-4.1,6.1,.38,entry+4.02,entry+4.25)
    g.block('roof',0,-4.1,5.9,.34,entry+6.82,entry+7.02)
    for x in (-2.95,2.95):g.block('roof',x,-4.1,.18,.38,entry+4.02,entry+7.02)
    for x in (-2.9,-1.45,0,1.45,2.9):g.line('metal',(x,entry+4.18,-4.03),(x,entry+6.84,-4.03),.035)
    # 首层侧窗、门楣和设备位置以照片为据，数量与比例为估计。
    g.block('glass',-3.25,-4.03,2.7,.10,entry+.40,entry+2.75)
    for x in (-4.65,-3.3,-1.85):g.line('metal',(x,entry+.38,-3.96),(x,entry+2.78,-3.96),.030)
    g.line('metal',(-4.65,entry+1.65,-3.95),(-1.85,entry+1.65,-3.95),.030)
    g.panel(1.82,-.10,1.10,entry+2.45,entry+3.02,(264,204,512,388))
    g.panel(-3.4,-.05,.70,entry+2.70,entry+3.03,(0,404,248,588))
    g.blue_block(2.4,-1.0,.34,.42,entry,entry+1.07)
    g.block('metal',2.4,-.98,.23,.26,entry+1.07,entry+1.23)
    # 电梯只表现门框与关闭门扇；尚未实现电梯运动或真实室内平面。
    g.block('metal',3.65,-3.81,1.45,.075,entry+.01,entry+2.45)
    g.line('roof',(3.65,entry+.02,-3.75),(3.65,entry+2.43,-3.75),.013)
    g.panel(2.70,-3.72,.17,entry+1.05,entry+1.46,(0,204,248,388))
    # 地图没有上下层高程，踏步按当前地形接回校园；方向与坡高仍待现场校准。
    z0,z1=-12.8,-20.0
    def deck_y(z):
        return max(g.ground(x,z) for x in (-1.56,0,1.56))+.06
    landing=deck_y(z0)
    count=max(20,int(math.ceil(abs(entry-landing)/.18)))
    run=8.8/count
    for i in range(count):
        z=-4.0-(i+.5)*run;top=entry+(landing-entry)*(i+1)/count
        g.block('stone',0,z,2.8,run+.01,min(entry,landing)-.12,top)
        walking_quad(-1.4,1.4,z-run*.5,z+run*.5,top,top)
        for side in (-1,1):
            g.block('roof',side*1.58,z,.34,run+.01,min(entry,landing)-.12,top+.70)
    for j in range(20):
        za=z0+(z1-z0)*j/20;zb=z0+(z1-z0)*(j+1)/20
        ya,yb=deck_y(za),deck_y(zb)
        points=[g.point(-1.56,ya,za),g.point(1.56,ya,za),g.point(1.56,yb,zb),g.point(-1.56,yb,zb)]
        dy,dz=yb-ya,zb-za;nz=-dy*math.copysign(1,dz)
        normal=(g.vx*nz,abs(dz),g.vz*nz)
        length=math.sqrt(sum(q*q for q in normal));normal=tuple(q/length for q in normal)
        g.layers['stone'].quad(*points,normal,((0,0),(1,0),(1,1),(0,1)))
        walking_quad(-1.56,1.56,za,zb,ya,yb)
        for side in (-1,1):
            x=side*1.56;pa,pb=g.point(x,ya,za),g.point(x,yb,zb)
            ba=g.point(x,min(ya-.18,g.ground(x,za)-.08),za)
            bb=g.point(x,min(yb-.18,g.ground(x,zb)-.08),zb)
            g.layers['stone'].quad(ba,bb,pb,pa,
                (side*g.ux,0,side*g.uz),((0,0),(1,0),(1,1),(0,1)))
    paved_transition(1.56,z1,z1-2.4,deck_y(z1))
    # 藤棚采用细钢管拱肋与网栏；叶片只概括常绿遮蔽，具体树种与密度未调查。
    for side in (-1,1):
        x=side*1.52
        for j in range(33):
            z=z0+(z1-z0)*j/32;base=deck_y(z)
            g.line('metal',(x,base+.12,z),(x,base+1.06,z),.007 if j%4 else .027)
        for k in range(6):
            za=z0+(z1-z0)*k/6;zb=z0+(z1-z0)*(k+1)/6
            for y in (.12,.30,.48,.66,.84,1.06):
                g.line('metal',(x,deck_y(za)+y,za),(x,deck_y(zb)+y,zb),.008 if .12<y<1.06 else .029)
    for k in range(7):
        z=z0+(z1-z0)*k/6
        for i in range(12):
            a,b=i*math.pi/12,(i+1)*math.pi/12
            p=(-1.6*math.cos(a),deck_y(z)+1.0+1.45*math.sin(a),z)
            q=(-1.6*math.cos(b),deck_y(z)+1.0+1.45*math.sin(b),z)
            g.line('metal',p,q,.023)
    for angle in (math.pi*.18,math.pi*.35,math.pi*.50,math.pi*.65,math.pi*.82):
        x=-1.6*math.cos(angle)
        for j in range(6):
            za=z0+(z1-z0)*j/6;zb=z0+(z1-z0)*(j+1)/6
            g.line('metal',(x,deck_y(za)+1.0+1.45*math.sin(angle),za),
                   (x,deck_y(zb)+1.0+1.45*math.sin(angle),zb),.017)
    for z in (z0,z1):
        for x in (-1.52,1.52):
            low=g.ground(x,z)
            if low<deck_y(z)-.20:g.line('metal',(x,low,z),(x,deck_y(z)-.18,z),.085)
    rng=random.Random(20230612)
    for i in range(260):
        angle=rng.uniform(.14,math.pi-.14);z=rng.uniform(z1,z0)
        x,y=-1.62*math.cos(angle),deck_y(z)+1.02+1.47*math.sin(angle)
        radius=rng.uniform(.13,.27)
        points=[g.point(x-radius,y,z),g.point(x,y+.03,z+radius*.7),
                g.point(x+radius,y,z),g.point(x,y-.03,z-radius*.7)]
        g.layers['sign'].quad(*points,(0,1,0),((.57,.03),(.94,.03),(.94,.10),(.57,.10)))
    return walk


def make_gates(*,mesh_type,height_at,beam,extrude,assets,data,vehicle_surface):
    """入口由根生成器传入共享地形与网格工具，分门、分材质保存。"""
    configs=json.loads((data/'gates-north.json').read_text(encoding='utf8'))['gates']
    result=[]
    colors={'stone':[.79,.80,.76],'metal':[.39,.45,.44],
            'glass':[.58,.72,.71],'roof':[.91,.92,.87],'sign':[1.,1.,1.]}
    for config in configs:
        gate=_Gate(config,mesh_type,height_at,beam,extrude)
        walking=None
        if config['id']=='ligong':_ligong(gate)
        elif config['id']=='liyan':_liyan(gate)
        else:walking=_lixin(gate,mesh_type)
        texture='gate-'+config['id']+'-sign.ppm'
        _sign_atlas(assets/texture,config['name'])
        layers=[]
        for material,mesh in gate.layers.items():
            if not mesh.vertices:continue
            filename=f"campus-gate-{config['id']}-{material}.mesh"
            count=mesh.save(assets/filename)
            layers.append({'material':material,'mesh':filename,'color':colors[material],
                           'vertices':count})
        item={'id':config['id'],'name':config['name'],'center_xz_m':config['center_xz_m'],
              'view_eye':list(gate.point(*config['view_local'])),
              'view_target':list(gate.point(*config['target_local'])),
              'layers':layers,'sign_texture':texture,'notes':config['notes'],
              'confidence':config['confidence']}
        if walking is not None:
            filename=f"campus-gate-{config['id']}-walk.mesh"
            walking.save(assets/filename);item['walking_mesh']=filename
        result.append(item)
    return {'gates':result}
