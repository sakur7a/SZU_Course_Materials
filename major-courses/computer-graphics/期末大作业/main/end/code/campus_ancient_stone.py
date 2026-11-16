"""文山湖畔“古石今人”独立景观模型。

石体使用程序纹理和独立三角网格；题字使用通用字体近似，未包含实景照片。
位置、高度和尺寸均待现场核实，生成参数保存在 data/wenshan-ancient-stone.json。
"""
import json
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


def make_ancient_stone(*, mesh_type, height_at, assets, data):
    config=json.loads((data/'wenshan-ancient-stone.json').read_text(encoding='utf-8'))
    cx,cz=config['position_xz_m']
    width,height,depth=config['width_m'],config['height_m'],config['depth_m']
    angle=math.radians(config['front_direction_degrees'])
    u=np.asarray((math.cos(angle),-math.sin(angle)))
    v=np.asarray((math.sin(angle),math.cos(angle)))
    ground=min(height_at(*(np.asarray((cx,cz))+a*u+b*v))
               for a in (-width/2,0,width/2) for b in (-depth/2,depth/2))-.05

    def granite(seed):
        rng=np.random.default_rng(seed)
        grain=rng.normal(0,10,(512,1024))
        coarse=Image.fromarray(rng.integers(0,256,(64,128),dtype=np.uint8))
        coarse=np.asarray(coarse.resize((1024,512),Image.Resampling.BICUBIC),dtype=float)-128
        values=np.clip(grain+coarse*.16,-35,35)
        color=np.asarray((126,104,91),dtype=float)[None,None,:]+values[:,:,None]
        return Image.fromarray(np.clip(color,0,255).astype(np.uint8),'RGB')

    def glyph(mask,character,box):
        # 四个字从右向左排列；专有书法以系统行楷代替，避免把实拍图片当纹理。
        canvas=Image.new('L',(400,400),0)
        face=ImageFont.truetype('C:/Windows/Fonts/STXINGKA.TTF',320)
        ImageDraw.Draw(canvas).text((30,0),character,font=face,fill=255)
        bounds=canvas.getbbox()
        if not bounds:
            raise ValueError('Cannot render the stone inscription')
        canvas=canvas.crop(bounds).resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
        mask.paste(ImageChops.lighter(mask.crop(box),canvas),box)

    def paint(image,mask):
        # 稍暗的外沿和红色内层仅表示刻字的颜料及凹陷，不模拟精确书法笔触。
        shadow=mask.filter(ImageFilter.MaxFilter(7)).transform(mask.size,Image.Transform.AFFINE,
                  (1,0,-2,0,1,-3),resample=Image.Resampling.BICUBIC)
        image.paste((70,25,20),(0,0,*image.size),shadow)
        image.paste((183,31,28),(0,0,*image.size),mask)
        highlight=ImageChops.subtract(mask,mask.filter(ImageFilter.MinFilter(5)))
        image.paste((209,55,43),(0,0,*image.size),highlight.point(lambda p:int(p*.45)))

    front,back=granite(1986),granite(1987)
    front_mask=Image.new('L',(1024,512),0)
    for char,box in (('人',(105,158,355,380)),('今',(345,130,535,401)),
                     ('石',(515,150,730,405)),('古',(705,157,903,390))):
        glyph(front_mask,char,box)
    paint(front,front_mask)
    back_mask=Image.new('L',(1024,512),0)
    glyph(back_mask,'无',(465,95,755,290))
    glyph(back_mask,'声',(470,245,750,430))
    paint(back,back_mask)
    atlas=Image.new('RGB',(2048,512))
    atlas.paste(front,(0,0));atlas.paste(back,(1024,0))
    atlas.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(assets/'wenshan-ancient-stone.ppm')

    mesh=mesh_type()
    columns,rows=32,18
    def local(i,j,front_side):
        t=j/rows
        half=width*.5*math.sqrt(max(.13,1-(.96*t)**2))
        left=-half*(.95+.045*math.sin(8*t)+.018*math.sin(19*t))
        right=half*(1.02+.045*math.sin(6*t+1)+.015*math.sin(17*t))
        x=left+(right-left)*i/columns
        # 浅鼓起的正反表面让题字随石面起伏，边缘保留不规则轮廓。
        wave=.028*math.sin(x*4+t*13)+.018*math.sin(x*9-t*17)
        bulge=.10*(1-(x/max(half,.01))**2)*math.sin(math.pi*t)
        z=(depth*.5+bulge+wave)*(1 if front_side else -1)
        y=t*height+.045*(t**6)*math.sin(i*math.pi/columns*5+.4)
        return x,y,z
    def world(p):
        x,y,z=p
        q=np.asarray((cx,cz))+x*u+z*v
        return (float(q[0]),float(ground+y),float(q[1]))
    def coords(p,front_side):
        x,y,_=p
        horizontal=(x/width+.5) if front_side else (.5-x/width)
        return ((.005+.49*horizontal) if front_side else (.505+.49*horizontal),
                .025+.95*y/height)
    def face(points,front_side):
        vertices=[world(p) for p in points]
        normal=np.cross(np.asarray(vertices[1])-vertices[0],np.asarray(vertices[2])-vertices[0])
        length=np.linalg.norm(normal)
        if length>.00001:
            mesh.quad(*vertices,normal/length,tuple(coords(p,front_side) for p in points))
    for j in range(rows):
        for i in range(columns):
            a,b=local(i,j,True),local(i+1,j,True)
            c,d=local(i+1,j+1,True),local(i,j+1,True)
            face((a,b,c,d),True)
            a,b=local(i+1,j,False),local(i,j,False)
            c,d=local(i,j+1,False),local(i+1,j+1,False)
            face((a,b,c,d),False)
    # 顶面和两侧连接正反面，以无字岩石纹理覆盖断面。
    def edge_quad(points):
        vertices=[world(p) for p in points]
        normal=np.cross(np.asarray(vertices[1])-vertices[0],np.asarray(vertices[2])-vertices[0])
        length=np.linalg.norm(normal)
        if length>.00001:
            mesh.quad(*vertices,normal/length,((.008,.08),(.040,.08),(.040,.90),(.008,.90)))
    for j in range(rows):
        edge_quad((local(0,j,False),local(0,j,True),local(0,j+1,True),local(0,j+1,False)))
        edge_quad((local(columns,j,True),local(columns,j,False),
                   local(columns,j+1,False),local(columns,j+1,True)))
    for i in range(columns):
        edge_quad((local(i,rows,True),local(i+1,rows,True),
                   local(i+1,rows,False),local(i,rows,False)))

    # 参考照片中的石脚绿植，以独立层表现；具体株位与品种均为近似。
    planting=mesh_type()
    for lx,lz,radius,tall in ((-2.35,.85,.62,.72),(-2.8,.40,.65,.80),
                              (-2.6,-.28,.55,.65),(2.35,.72,.54,.70),
                              (2.75,.14,.68,.85),(2.55,-.42,.56,.68)):
        center=np.asarray((cx,cz))+lx*u+lz*v
        x,z=map(float,center)
        y=height_at(x,z)-.03
        for a in (0,math.pi/3,2*math.pi/3):
            dx,dz=math.cos(a)*radius,math.sin(a)*radius
            planting.quad((x-dx,y,z-dz),(x+dx,y,z+dz),
                          (x+dx,y+tall,z+dz),(x-dx,y+tall,z-dz),
                          (-math.sin(a),0,math.cos(a)),((0,0),(1,0),(1,1),(0,1)))
    return {'vertices':mesh.save(assets/'campus-wenshan-ancient-stone.mesh'),
            'planting_vertices':planting.save(assets/'campus-wenshan-ancient-stone-planting.mesh'),
            'location':config['position_xz_m'],'ground_estimate_m':round(ground,2)}
