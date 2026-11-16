"""从独立地点目录生成中文标注图集；标签高度用于识读，不代表建筑测绘。"""
import json
import math
import struct

from PIL import Image, ImageDraw, ImageFont


def make_places(*, buildings, waters, nodes, height_at, height_for, gates, assets, data):
    config=json.loads((data/'campus-places.json').read_text(encoding='utf8'))
    building_index={key:(tags,geom) for key,tags,geom in buildings}
    areas={key:geom for key,_,geom in waters}
    records=[]
    for item in config['places']:
        bounds=[0,0,0,0];local_y=0
        kind=item.get('kind','building' if item.get('osm') else 'landmark')
        if item.get('osm'):
            tags,geom=building_index[item['osm']]
            p=geom.representative_point();x,z=p.x,p.y
            offset=height_for(item['osm'],tags,geom)[0]+5
            bounds=list(geom.bounds);local_y=height_at(x,z)+min(10,(offset-5)*.35)
        elif item.get('area'):
            p=areas[item['area']].representative_point();x,z=p.x,p.y
            offset=item.get('height_above_ground_m',10)
        else:
            x,z=nodes[item['node']] if item.get('node') else item['position_xz_m']
            offset=item.get('height_above_ground_m',8)
        records.append(dict(id=item['id'],name=item['name'],position=[x,height_at(x,z)+offset,z],
            kind=kind,priority=item.get('priority',1),max_distance_m=item.get('max_distance_m',3600),
            body_bounds_xz=bounds,local_label_y=local_y))
    for gate in gates:
        x,z=gate['center_xz_m']
        records.append(dict(id='gate-'+gate['id'],name=gate['name'],
            position=[x,gate.get('top_y_m',height_at(x,z)+10)+3,z],kind='gate',priority=1,max_distance_m=3800,
            body_bounds_xz=[0,0,0,0],local_label_y=0))
    if len({p['id'] for p in records})!=len(records):
        raise ValueError('Place identifiers must be unique')

    cell_w,cell_h,columns=512,64,2
    height=2**math.ceil(math.log2(max(64,math.ceil(len(records)/columns)*cell_h)))
    atlas=Image.new('RGBA',(cell_w*columns,height),(0,0,0,0))
    draw=ImageDraw.Draw(atlas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    colors={'building':(137,199,223),'district':(236,193,125),'gate':(135,221,183),
            'shop':(242,183,152),'water':(124,201,225),'sports':(174,210,160),
            'landmark':(206,207,223)}
    lines=['# id x y z u0 v0 u1 v1 width_px height_px max_distance priority min_x min_z max_x max_z local_y name']
    for index,record in enumerate(records):
        x0=(index%columns)*cell_w;y0=(index//columns)*cell_h
        width=min(cell_w-4,max(120,int(draw.textlength(record['name'],font=font))+54))
        draw.rounded_rectangle((x0+2,y0+2,x0+width-2,y0+49),radius=10,
                               fill=(26,43,49,228),outline=(214,226,224,195),width=1)
        color=colors[record['kind']]
        draw.ellipse((x0+13,y0+20,x0+23,y0+30),fill=(*color,255))
        draw.text((x0+34,y0+25),record['name'],font=font,anchor='lm',fill=(247,249,246,255))
        draw.line((x0+width/2,y0+49,x0+width/2,y0+59),fill=(*color,235),width=2)
        draw.ellipse((x0+width/2-3,y0+57,x0+width/2+3,y0+63),fill=(*color,255))
        # 图集整体翻转后，UV下边对应原图底边，保持中文字朝上。
        uv=[x0/atlas.width,1-(y0+cell_h)/atlas.height,(x0+width)/atlas.width,1-y0/atlas.height]
        record['uv']=uv;record['size_px']=[width*.5,cell_h*.5]
        values=[record['id'],*record['position'],*uv,*record['size_px'],
                record['max_distance_m'],record['priority'],*record['body_bounds_xz'],record['local_label_y']]
        lines.append('\t'.join(str(v) if isinstance(v,str) else f'{v:.6f}' for v in values)+'\t'+record['name'])
    (assets/'campus-places.tsv').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (assets/'campus-places.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    atlas=atlas.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    with (assets/'campus-place-labels.rgba').open('wb') as file:
        file.write(struct.pack('<II',*atlas.size));file.write(atlas.tobytes())
    return {'places':len(records),'atlas_size':list(atlas.size)}
