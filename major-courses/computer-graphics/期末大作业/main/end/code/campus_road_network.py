"""校园车道两侧和地图步行线的独立景观层。

OSM 快照没有记录绝大多数道路的人行道侧别；路侧宽度、铺装与植株均为近似，
不能当作对现状的逐段测绘。独立生成资源，避免混入各个标志建筑的模型。
"""
import math
import random
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

# 湖岸台阶和石头坞广场已有独立的三维模型，不再把其旧轮廓生成额外的铺装环。
MODELED_PEDESTRIAN_AREAS={'way/231664128','way/231983458','way/231665580'}


def adjust_road_alignment(line, building, config):
    """对与楼体几乎相切的地图道路做局部视觉修正，端点仍保持地图连接。"""
    def smooth(value):
        value=max(0.0,min(1.0,value))
        return value*value*(3.0-2.0*value)

    start,end,blend=config['start_m'],config['end_m'],config['blend_m']
    if not 0<start<end<line.length or blend<=0:
        raise ValueError('Invalid road-alignment interval')
    points=[]
    for distance in np.linspace(0,line.length,max(2,int(math.ceil(line.length))+1)):
        point=line.interpolate(distance)
        a=line.interpolate(max(0,distance-.5))
        b=line.interpolate(min(line.length,distance+.5))
        dx,dz=b.x-a.x,b.y-a.y
        length=math.hypot(dx,dz)
        if length<.001:
            raise ValueError('Road alignment has a degenerate tangent')
        weight=min(smooth((distance-start)/blend),smooth((end-distance)/blend))
        offset=config['offset_m']*weight
        points.append((point.x-offset*dz/length,point.y+offset*dx/length))
    adjusted=LineString(points)
    if adjusted.distance(building)<config['min_center_clearance_m']:
        raise ValueError('Adjusted road still touches the nearby building')
    return adjusted


def line_parts(geom):
    if geom.is_empty:
        return
    if geom.geom_type in ('LineString', 'LinearRing'):
        yield geom
    elif hasattr(geom, 'geoms'):
        for child in geom.geoms:
            yield from line_parts(child)


def sidewalk_texture(assets):
    """使用自绘浅灰砖纹区分步行带与沥青车道。"""
    image=Image.new('RGB',(256,256),(190,190,181))
    draw=ImageDraw.Draw(image)
    rng=random.Random(20261103)
    for row in range(8):
        y=row*32
        offset=(row%2)*32
        for x in range(-offset,256,64):
            k=rng.randrange(-5,6)
            draw.rectangle((x+1,y+1,x+62,y+30),fill=(198+k,198+k,188+k))
    image.save(assets/'campus-sidewalk.ppm')


def build_road_environment(*, campus, vehicle_roads, roadways, vehicle_surface,
                           buildings, waters, pitches, reserved, bridge_clearance,
                           junctions, road_width, pedestrian_types, drape, mesh_type,
                           height_at, beam, ring_ids, assets, data):
    """生成分离的路侧人行带、已映射步道和低矮绿化，并返回可供树木使用的候选点。"""
    roads=[(key,t,g.intersection(campus)) for key,t,g in vehicle_roads if g.intersects(campus)]
    paths=[]
    for key,t,g in roadways:
        if (t.get('highway') not in pedestrian_types or t.get('bridge') not in (None,'no')
                or t.get('tunnel') not in (None,'no','building_passage') or t.get('layer','0')!='0'
                or key in MODELED_PEDESTRIAN_AREAS):
            continue
        # area=yes 的闭合线是步行广场的边界，而不是一条环形道路。
        if t.get('area')=='yes' and g.geom_type=='LineString' and g.is_ring:
            g=Polygon(g.coords).buffer(0)
        if g.intersects(campus):
            paths.append((key,t,g.intersection(campus)))
    # 桥面和专门建模的广场由各自模型负责；楼体、水域、球场不能被人行道覆盖。
    obstacles=unary_union([g.buffer(.20) for _,_,g in buildings]
                          +[g.buffer(.35) for _,_,g in waters]
                          +[g.buffer(.50) for _,_,g in pitches]+[reserved])
    # 在较宽的主路预留步行带及可选择的草带；小支路仍保留原来的紧凑宽度。
    def landscaped_main_road(key,t,geom):
        return key in ring_ids or (t.get('highway')=='unclassified' and geom.length>=60)

    roads_outer=unary_union([g.buffer(road_width(key,t)/2+
                                     (3.15 if landscaped_main_road(key,t,g) else 2.0),
                                     cap_style='round',join_style='round') for key,t,g in roads])
    # 原有道路共用联合面，路口边缘不会被另一条车道的人行带横切。
    sidewalks=(roads_outer.intersection(campus).difference(vehicle_surface.buffer(.10))
               .difference(obstacles))
    mapped=unary_union([g if g.geom_type in ('Polygon','MultiPolygon') else
                        g.buffer(road_width(key,t)/2,cap_style='round',join_style='round')
                        for key,t,g in paths])
    mapped=mapped.difference(vehicle_surface.buffer(.08)).difference(obstacles)
    # 地图已有步行线优先绘制；与推测的人行带在同一地形面上衔接。
    sidewalks=sidewalks.difference(mapped)
    # 实景中的部分环路在车道和人行道之间有草带。只在空间足够的路段近似放置，
    # 路口、地图步行线和专用景点均留出开口；草带具体位置不是逐段实测。
    def offset_point(line,dist,offset,sign):
        p=line.interpolate(dist)
        a=line.interpolate(max(0,dist-.8)); b=line.interpolate(min(line.length,dist+.8))
        dx,dz=b.x-a.x,b.y-a.y
        length=math.hypot(dx,dz)
        return None if length<.05 else (p.x-sign*dz/length*offset,p.y+sign*dx/length*offset)

    available=prep(sidewalks.buffer(.025))
    no_verge=prep(unary_union((mapped.buffer(.55),junctions.buffer(6),bridge_clearance,
                              obstacles.buffer(.3))))
    strips=[]
    walking_rooms=[]
    for key,t,geom in sorted(roads,key=lambda item:item[0]):
        if not landscaped_main_road(key,t,geom):
            continue
        half=road_width(key,t)*.5
        for line in line_parts(geom):
            for start in np.arange(6,max(6,line.length-10),5.0):
                end=min(line.length-4,float(start)+5.0)
                for side in (-1,1):
                    inner=[offset_point(line,d,half+v,side) for d,v in
                           ((start,.24),(end,.24),(end,1.05),(start,1.05))]
                    outer=[offset_point(line,d,half+v,side) for d,v in
                           ((start,1.18),(end,1.18),(end,2.95),(start,2.95))]
                    if any(p is None for p in inner+outer):
                        continue
                    strip=Polygon(inner).buffer(0); walking_room=Polygon(outer).buffer(0)
                    if (not strip.is_empty and strip.area>1 and not walking_room.is_empty
                            and available.covers(strip) and available.covers(walking_room)
                            and not no_verge.intersects(strip.union(walking_room))):
                        strips.append(strip)
                        walking_rooms.append(walking_room)
    # 紧邻的另一条道路也可能生成草带；保留每条草带前先检查其外侧连续步行空间。
    proposed_verge=unary_union(strips) if strips else Polygon()
    strips=[strip for strip,room in zip(strips,walking_rooms)
            if room.intersection(proposed_verge).area<room.area*.01]
    verge=unary_union(strips).buffer(.025).buffer(-.025) if strips else Polygon()
    sidewalks=sidewalks.difference(verge)
    walkable=unary_union((sidewalks,mapped))
    if sidewalks.intersection(vehicle_surface.buffer(-.03)).area>.01:
        raise ValueError('Inferred sidewalks overlap the vehicle surface')
    sidewalk_mesh,path_mesh,shrubs,verge_mesh,grass_mesh= (mesh_type() for _ in range(5))
    drape(sidewalk_mesh,sidewalks,.055,lambda x,z:(x/2.4,z/2.4))
    drape(path_mesh,mapped,.055,lambda x,z:(x/3.0,z/3.0))
    drape(verge_mesh,verge,.050,lambda x,z:(x/2.5,z/2.5))

    # 在有余地的路侧安排少量植株；避让车道、人行道、路口、水面和已有景点。
    exclusion=unary_union((obstacles.buffer(2.0),vehicle_surface.buffer(1.0),
                           walkable.buffer(1.10),junctions.buffer(5),bridge_clearance))
    greenable=prep(campus.buffer(-3).difference(exclusion))
    rng=random.Random(2024270223)
    shrub_points=[]
    tree_points=[]
    grid=defaultdict(list)

    def isolated(x,z,spacing):
        gx,gz=int(math.floor(x/spacing)),int(math.floor(z/spacing))
        for i in range(gx-1,gx+2):
            for j in range(gz-1,gz+2):
                if any(math.hypot(x-a,z-b)<spacing for a,b in grid[(i,j)]):
                    return False
        grid[(gx,gz)].append((x,z))
        return True

    # 灯杆布置在步行带外的空地上；样式参考校方环路照片，位置和数量为近似。
    light_mesh,light_head=mesh_type(),mesh_type()
    light_points=[]
    for key,t,geom in sorted(roads,key=lambda item:item[0]):
        if not landscaped_main_road(key,t,geom):
            continue
        for line in line_parts(geom):
            for dist in np.arange(20,line.length-12,48):
                for side in (-1,1):
                    p=offset_point(line,float(dist),road_width(key,t)/2+5.2,side)
                    if p is None or not greenable.covers(Point(*p).buffer(.48)):
                        continue
                    if any(math.dist(p,q)<27 for q in light_points):
                        continue
                    light_points.append(p)
                    x,z=p; y=height_at(x,z)
                    beam(light_mesh,(x,y,z),(x,y+3.25,z),.055)
                    # 照片里的顶灯接近上宽下窄的浅罩；此处用低面数几何体示意。
                    for i in range(8):
                        a,b=2*math.pi*i/8,2*math.pi*(i+1)/8
                        low_a=(x+.085*math.cos(a),y+3.23,z+.085*math.sin(a))
                        low_b=(x+.085*math.cos(b),y+3.23,z+.085*math.sin(b))
                        high_a=(x+.30*math.cos(a),y+3.46,z+.30*math.sin(a))
                        high_b=(x+.30*math.cos(b),y+3.46,z+.30*math.sin(b))
                        light_head.quad(low_a,high_a,high_b,low_b,
                                        (math.cos((a+b)/2),.35,math.sin((a+b)/2)),
                                        ((0,0),(0,1),(1,1),(1,0)))

    def near_light(x,z,distance=2.0):
        return any(math.hypot(x-a,z-b)<distance for a,b in light_points)

    for key,t,geom in sorted(roads,key=lambda item:item[0]):
        width=road_width(key,t)
        for line in line_parts(geom):
            if line.length<24:
                continue
            for dist in np.arange(10,line.length-8,19):
                for side in (-1,1):
                    point=offset_point(line,float(dist),width/2+4.2,side)
                    if point is None:
                        continue
                    x,z=point
                    if (not greenable.covers(Point(x,z).buffer(.65)) or near_light(x,z) or
                            rng.random()<.22 or not isolated(x,z,6.0)):
                        continue
                    shrub_points.append((x,z))
            for dist in np.arange(17,line.length-12,32):
                for side in (-1,1):
                    point=offset_point(line,float(dist),width/2+4.8,side)
                    if (point is not None and not near_light(*point,4.0)
                            and greenable.covers(Point(*point).buffer(1.0))):
                        tree_points.append(point)
    # 稀疏草簇限制在草带内部，底座贴合三角地面，不侵占相邻步行带。
    grass_points=[]
    if not verge.is_empty:
        inside_verge=prep(verge.buffer(-.10))
        xmin,zmin,xmax,zmax=verge.bounds
        for x0 in np.arange(xmin,xmax,2.8):
            for z0 in np.arange(zmin,zmax,2.8):
                x,z=x0+rng.uniform(-.35,.35),z0+rng.uniform(-.35,.35)
                if not inside_verge.covers(Point(x,z)):
                    continue
                grass_points.append((x,z))
                y=height_at(x,z)+.055
                for angle in (0,math.pi/3,2*math.pi/3):
                    dx,dz=.055*math.cos(angle),.055*math.sin(angle)
                    grass_mesh.triangle((x-dx,y,z-dz),(x+dx,y,z+dz),
                                        (x+dx*.25,y+rng.uniform(.14,.25),z+dz*.25),
                                        (-math.sin(angle),.15,math.cos(angle)),
                                        ((0,0),(1,0),(.5,1)))
    # 三片交叉透明叶片表达低灌木，使用原有枝叶纹理，与校园树木共享风格。
    for x,z in shrub_points:
        y=height_at(x,z)+.03
        radius=rng.uniform(.48,.76)
        tall=rng.uniform(.65,1.05)
        for angle in (0,math.pi/3,2*math.pi/3):
            dx,dz=math.cos(angle)*radius,math.sin(angle)*radius
            normal=(-math.sin(angle),0,math.cos(angle))
            shrubs.quad((x-dx,y,z-dz),(x+dx,y,z+dz),
                        (x+dx,y+tall,z+dz),(x-dx,y+tall,z-dz),normal,
                        ((0,0),(1,0),(1,1),(0,1)))
    if not shrub_points:
        raise ValueError('No roadside planting could be generated')
    counts={'inferred_sidewalk':sidewalk_mesh.save(assets/'campus-network-sidewalk.mesh'),
            'mapped_paths':path_mesh.save(assets/'campus-network-paths.mesh'),
            'shrubs':shrubs.save(assets/'campus-network-shrubs.mesh'),
            'verges':verge_mesh.save(assets/'campus-network-verges.mesh'),
            'verge_grass':grass_mesh.save(assets/'campus-network-verge-grass.mesh'),
            'road_lamps':light_mesh.save(assets/'campus-network-lamps.mesh'),
            'road_lamp_heads':light_head.save(assets/'campus-network-lamp-heads.mesh'),
            'vehicle_way_count':len(roads), 'vehicle_way_length_m':round(sum(g.length for _,_,g in roads)),
            'mapped_path_count':len(paths),
            'mapped_area_count':sum(g.geom_type in ('Polygon','MultiPolygon') for _,_,g in paths),
            'mapped_path_length_m':round(sum(g.length for _,_,g in paths if g.geom_type not in ('Polygon','MultiPolygon'))),
            'shrub_count_approximate':len(shrub_points),
            'verge_area_estimate_m2':round(verge.area), 'verge_grass_clumps':len(grass_points),
            'lamp_count_approximate':len(light_points)}
    (data/'roadside-environment.json').write_text(json.dumps({
        'lamp_points_estimate_m':[[round(x,3),round(z,3)] for x,z in light_points],
        'verge_area_estimate_m2':round(verge.area,2),
        'note':'草带、灯具和植物为照片启发下的近似布置，不是实地调查'},
        ensure_ascii=False,indent=2),encoding='utf-8')
    return counts,walkable,tree_points,light_points
