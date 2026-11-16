#!/usr/bin/env python3
"""将离线 OSM / 高程快照转换为本作业使用的校园网格与纹理。

仅重新生成素材时需要 Pillow、numpy、shapely 和 mapbox-earcut；运行程序不需要 Python。
地理数据保留各自的来源，缺失高度的楼宇使用教学预览估算，不能视为实测。
"""
import csv
import collections
import json
import math
import random
import struct
from array import array
from pathlib import Path
from xml.etree import ElementTree as ET

import mapbox_earcut
import numpy as np
import shapely
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import campus_road_network
import campus_road_overrides
import campus_planting
import campus_lide_corridor
import campus_south_spine
import campus_road_terrain
import campus_terrain_patch
import campus_time_square
import campus_activity_foundation
import campus_ancient_stone
import campus_shuttle
import campus_stairways
import campus_wenshan_swans
import campus_wenshan_shore
import campus_kaifeng
import campus_gates
import campus_places
import campus_zhounan_portal
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, polygonize, substring, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
DATA, ASSETS = HERE / 'data', HERE / 'assets'
GATE_EXCLUDED_BUILDINGS=campus_gates.excluded_buildings(DATA)
INDEPENDENT_DORMS={'way/1258216924','way/1258216925','way/629940893','way/629940894','way/629940895',
                   'way/292598302','way/292598304','relation/13233284','way/220880938',
                   'way/292598303','way/242066977'}
LON0, LAT0 = 113.9318, 22.5355
METERS_LON, METERS_LAT = 102800.0, 111320.0
X0, X1, Z0, Z1 = -800.0, 720.0, -830.0, 860.0
# 立德门右侧步道的估计线和景观避让、独立网格共用配置。
LIDE_WALK = json.loads((DATA/'lide-walk.json').read_text(encoding='utf8'))
LIDE_WALK_POINTS = LIDE_WALK['centerline_m']
SOUTH_SPINE = json.loads((DATA/'south-spine.json').read_text(encoding='utf8'))

# 地面道路先以两倍分辨率绘制，再缩回 GPU 纹理尺寸，改善近景锯齿。
RES, GROUND_TEXTURE_SIZE = 8192, 4096

# 主要建筑用实拍单独制作；其余建筑直接按 OSM 多边形建体量。
SPECIAL = {'way/230847103', 'way/230847105', 'way/237484981', 'relation/3110985'}
HEIGHTS = {
    'way/230847103': (95.0, '深圳大学建筑设计研究院公开资料：科技楼建筑高度95米'),
    'way/230847105': (25.0, '学校2025年北馆结构检测采购：建筑高度25米'),
    'way/237484981': (20.3, '南馆5层；总高暂按造型估计'),
    'way/938357240': (44.01, '官方2026规划核查：汇萃楼8层、44.01米'),
    'way/1271852286': (46.8, '官方总平面图科研楼13层；总高按3.6米层高估计'),
    'way/1271852287': (57.7, '官方2026规划核查：汇语楼14层、57.7米'),
    'way/1081134194': (19.0, '汇意楼原院馆高度按实拍比例估计，非扩建楼核查高度'),
    'way/1207358023': (20.45, '官方2026规划核查：汇意楼扩建5层、20.45米'),
    'way/1455957244': (36.75, '官方2026核查：建筑与城市规划学院教学实验楼扩建9层、36.75米'),
    'way/1085792762': (82.8, '学校资料确认汇研楼23层；总高按层高估计'),
    'way/1558576826': (63.0, '资料记聚翰斋18层；层高按3.5米估计'),
    'way/1558576825': (35.0, '资料记紫薇斋10层；层高按3.5米估计'),
    'way/1558576824': (52.5, '资料记红豆斋15层；层高按3.5米估计'),
    'way/231572980': (49.5, '学校资料确认风槐斋15层；层高估计'),
    'way/629940893': (78.25, '西南二期乔苑20层宿舍；建筑高度78.25米'),
    'way/629940894': (78.25, '西南二期乔苑20层宿舍；建筑高度78.25米'),
    'way/629940895': (78.25, '西南二期乔苑20层宿舍；建筑高度78.25米'),
    'way/1258216924': (39.6, '学校资料确认乔木阁所在西南一期主体12层；层高估计'),
    'way/1258216925': (39.6, '学校资料确认乔林阁所在西南一期主体12层；层高估计'),
    'way/319101983': (33.0, 'OSM 致腾楼主体分部为10层；总高按层高估计'),
    'way/319101984': (59.0, 'OSM 致原楼分部标注14层、59米；其余分部按层数估计'),
    'way/629940889': (33.0, '设计团队资料：十层塔楼；总高按层高估计'),
    'way/629940890': (33.0, '设计团队资料：十层塔楼；总高按层高估计'),
    'way/776358621': (9.9, '设计团队资料：三层裙房；总高按层高估计'),
    'way/317029242': (20.0, '国家档案局资料为5层；总高按层高估计'),
    'way/1455962776': (9.2, '深圳市历史建筑资料为单层；屋顶高度按照片估计'),
    'way/230957349': (10.5, '学生指南称网球场在三楼；校方有四楼记录，结构待核实'),
    'way/231572974': (7.5, '石头坞石砌楼为两层；高度按翻新照片估计'),
    'way/231665582': (10.5, '南秋馆配套楼三层；高度按翻新照片估计'),
    'way/231795526': (10.2, '学生活动中心三层；高度按层高估计'),
}
NAMES = {'way/230847103':'汇星楼（科技楼）', 'way/230847104':'汇元楼（办公楼）',
         'way/230847105':'汇典楼（图书馆北馆）', 'way/237484981':'汇智楼（图书馆南馆）',
         'way/230847098':'汇紫楼C座', 'way/230847099':'汇紫楼D座',
         'way/230847100':'汇紫楼B座', 'way/230847102':'汇紫楼A座',
         'way/319101983':'致腾楼（计算机与软件学院等）', 'way/319101984':'致原楼（物理与光电工程学院）',
         'way/231086366':'致工楼（土木与交通工程学院）', 'way/984606893':'致知楼（高等研究院）',
         'way/629940889':'致艺楼（设计教学楼）', 'way/629940890':'致艺楼（设计教学楼）',
         'way/776358621':'致艺楼三层裙房',
         'way/938357240':'汇萃楼（艺术综合楼）',
         'way/1207358023':'汇意楼扩建（建规学院）',
         'way/629940893':'乔森阁', 'way/231572974':'石头坞石砌楼',
         'way/231067701':'汇德楼B座（原师范学院）',
         'way/231665582':'南秋馆（泳池配套楼）', 'way/231795526':'汇昀楼（学生活动中心）',
         # 用户确认该处已改作其他用途；新用途未确认，保留历史位置而不虚构现有名称。
         'way/230935221':'老西南餐厅旧址（现作其他用途）'}


def tags(item):
    return {t.get('k'): t.get('v') for t in item.findall('tag')}


def clean(poly):
    if poly.is_empty:
        return poly
    return poly if poly.is_valid else poly.buffer(0)


def polygons(geom):
    if geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        yield geom
    elif hasattr(geom, 'geoms'):
        for part in geom.geoms:
            yield from polygons(part)


def lines(geom):
    if geom.is_empty:
        return
    if geom.geom_type in ('LineString', 'LinearRing'):
        yield geom
    elif hasattr(geom, 'geoms'):
        for part in geom.geoms:
            yield from lines(part)


def geographic_point(lon, lat):
    return (lon - LON0) * METERS_LON, (LAT0 - lat) * METERS_LAT


root = ET.parse(DATA / 'yuehai.osm').getroot()
nodes = {n.get('id'): geographic_point(float(n.get('lon')), float(n.get('lat')))
         for n in root.findall('node')}
ways = {w.get('id'): w for w in root.findall('way')}
relations = {r.get('id'): r for r in root.findall('relation')}


def way_points(item):
    return [nodes[n.get('ref')] for n in item.findall('nd') if n.get('ref') in nodes]


def way_polygon(item):
    p = way_points(item)
    if len(p) < 4 or p[0] != p[-1]:
        return Polygon()
    return clean(Polygon(p))


def relation_polygon(item):
    # OSM 多边形关系可以包含多个外环与内环；保留湖中岛屿和建筑内院。
    rings = {'outer': [], 'inner': []}
    for member in item.findall('member'):
        if member.get('type') != 'way' or member.get('ref') not in ways:
            continue
        role = member.get('role') or 'outer'
        if role in rings:
            p = way_points(ways[member.get('ref')])
            if len(p) >= 2:
                rings[role].append(LineString(p))
    outer = unary_union(list(polygonize(unary_union(rings['outer'])))) if rings['outer'] else Polygon()
    inner = unary_union(list(polygonize(unary_union(rings['inner'])))) if rings['inner'] else Polygon()
    return clean(outer.difference(inner))


north = way_polygon(ways['97548953'])
south = way_polygon(ways['1037951940'])
campus = clean(north.union(south))
near_campus = campus.buffer(30)
# 北门外的深南大道主车道和辅路距校界略超 30 米，但从校园可直接看到。
shennan_ways = {'633620701','633620767','183402036','413035251'}
shennan_primary = {'way/633620701','way/633620767'}
terrain_extent = box(X0,Z0,X1,Z1)
shennan_corridor = LineString(((280,-790),(720,-813))).buffer(31,cap_style=2)
shennan_inner,shennan_outer = prep(shennan_corridor),prep(shennan_corridor.buffer(20))
campus_interior = prep(campus)

# 北门路及环校路交叉处：30 米级 DEM 混入树冠后将路面拉成陡峭横坡。
# 道路中心线取 OSM，纵向标高仅为与周围地形连续的视觉估计，不是测绘数据。
legong_road = LineString(((285.6,-760),(285.6,-708),(286.1,-688.5),
                          (286.9,-657.4),(288.3,-600.6),(289.3,-527),(289.5,-500)))
legong_crossroad = LineString(((255,-688),(286.1,-688.5),(320,-692)))
legong_road_outer = prep(legong_road.buffer(34).union(legong_crossroad.buffer(20)))
# 入口与高尔夫场之间的低谷同样是粗高程产生的孤立伪影，不能只抬高路心。
legong_entry_area = box(250,-735,340,-655)
legong_entry_outer = prep(legong_entry_area.buffer(25))
# 北门内两排棕榈仅是按资料复原的行道树结构，逐棵位置和数量没有调查数据。
legong_palms = [(286+(z+688)*.019+side*12.5,z)
                 for z in (-668,-648,-628,-608,-588,-568,-548,-488,-468)
                 for side in (-1,1)]

# 立言门与致远门入口沿 OSM 路线定位，铺装范围根据卫星影像估计。
liyan_entry_axis = LineString(((126.0,325.0),(149.0,349.0)))
liyan_entry = liyan_entry_axis.buffer(9.0,cap_style=2)
zhiyuan_entry_axis = LineString(((186.0,386.5),(200.0,402.5)))
zhiyuan_entry = zhiyuan_entry_axis.buffer(17.0,cap_style=2)
zhiyuan_entry_inner,zhiyuan_entry_outer = prep(zhiyuan_entry.buffer(6)),prep(zhiyuan_entry.buffer(18))
# 致远门旁两块相邻草坪在卫星影像和校方活动照片中是开敞大草坪。
south_gate_lawn = unary_union([way_polygon(ways[k]) for k in ('1272413426','1272413427')])
# 致知楼与 L4 之间的开敞通道按航拍概括，用于避免随机树穿过庭院。
zhizhi_courtyard = box(181,583.5,251,600.5).difference(way_polygon(ways['984606893']))
# 致理楼 A 座入口前留出航拍可见的硬质通行区域，边界按地图和照片估计。
_zhili_a=way_points(ways['776358622'])
_za,_zb=np.asarray(_zhili_a[6]),np.asarray(_zhili_a[0])
_zt=(_zb-_za)/np.linalg.norm(_zb-_za)
_zn=np.asarray((_zt[1],-_zt[0]))
_zleft,_zright=_za+(_zb-_za)*.43,_za+(_zb-_za)*.76
zhili_a_entry=Polygon((_zleft,_zright,_zright+_zn*18,_zleft+_zn*18))
# OSM 用不同 layer 区分三条 L1–L3 高架线；楼层与相应高程仅作为建模估计。
zhili_high_walkways=(('1272661380',2),('1272661379',4),('1272661378',6))
# 致工楼西侧停车场有明确的 OSM 范围；入口前场和台阶位置参考校方照片估计。
zhigong_parking=way_polygon(ways['976046850'])
zhigong_forecourt=unary_union((box(168,651,205,659),box(168,659,184.5,679),
                               box(184.5,659,205.5,679)))
zhigong_stair_site=box(170,673,181,688)

# 深南大道以北仅作城市远景，不扩展可探索的校园边界。
CITY_NORTH = -1160.0
CITY_EAST, CITY_SOUTH = 1050.0, -500.0
city_extent = unary_union((box(X0,CITY_NORTH,CITY_EAST,Z0),box(X1,Z0,CITY_EAST,CITY_SOUTH)))
city_root = ET.parse(DATA / 'north-city.osm').getroot()
city_nodes = {n.get('id'): geographic_point(float(n.get('lon')),float(n.get('lat')))
              for n in city_root.findall('node')}
city_ways = city_root.findall('way')


def city_points(item):
    return [city_nodes[n.get('ref')] for n in item.findall('nd') if n.get('ref') in city_nodes]


def in_campus(geom):
    return not geom.is_empty and campus.covers(geom.representative_point())


buildings, surrounding, waters, parks, pitches, roadways = [], [], [], [], [], []
usable_terrain = box(X0 + 10,Z0 + 10,X1 - 10,Z1 - 10)
for kind, collection in (('way', ways), ('relation', relations)):
    for osm_id, item in collection.items():
        t = tags(item)
        if kind == 'relation' and t.get('type') != 'multipolygon':
            continue
        selected = ('building' in t or t.get('natural') in ('water', 'wood')
                    or t.get('landuse') in ('grass', 'forest', 'meadow')
                    or t.get('leisure') in ('park', 'garden', 'pitch', 'stadium'))
        if not selected:
            continue
        geom = way_polygon(item) if kind == 'way' else relation_polygon(item)
        if not in_campus(geom):
            # 用已有高度信息的少量真实周边楼宇作为背景，避免把城区山包误认为校园山体。
            if ('building' in t and not geom.is_empty and geom.intersects(campus.buffer(180))
                    and usable_terrain.covers(geom) and (t.get('height') or t.get('building:levels'))):
                try:
                    height = float(t.get('height', '0').replace('m',''))
                    levels = float(t.get('building:levels', '0'))
                    if height >= 12 or levels >= 4:
                        surrounding.append((f'{kind}/{osm_id}', t, geom))
                except ValueError:
                    pass
            continue
        key = f'{kind}/{osm_id}'
        if 'building' in t and t.get('building') not in ('no', 'roof'):
            buildings.append((key, t, geom))
        if t.get('natural') == 'water':
            waters.append((key, t, geom))
        if t.get('natural') == 'wood' or t.get('landuse') in ('grass', 'forest', 'meadow') or t.get('leisure') in ('park', 'garden'):
            parks.append((key, t, geom))
        if t.get('leisure') in ('pitch', 'stadium'):
            pitches.append((key, t, geom))

# 2026 总平面将建规学院教学实验楼和五层扩建楼分列；OSM 施工轮廓包含前者，需去重。
buildings = [(key,t,clean(geom.difference(way_polygon(ways['1455957244']))))
             if key == 'way/1207358023' else (key,t,geom) for key,t,geom in buildings]

# 两处天人广场小水景的实拍湖岸为圆弧，按 OSM 轮廓作小尺度圆角化。
waters = [(key,t,clean(geom.buffer(-1.3).buffer(1.3)))
          if key in ('way/231311761','way/231311762') else (key,t,geom)
          for key,t,geom in waters]

for osm_id, item in ways.items():
    t = tags(item)
    if 'highway' not in t:
        continue
    p = way_points(item)
    if len(p) < 2:
        continue
    line = LineString(p)
    if osm_id in shennan_ways and line.intersects(terrain_extent):
        roadways.append((f'way/{osm_id}',t,line.intersection(terrain_extent)))
    elif line.intersects(near_campus):
        roadways.append((f'way/{osm_id}', t, line.intersection(near_campus)))

# 个别地图线位与楼体重叠时，对道路局部做明确记录的近似偏移，保持原有端点。
road_alignment_config=json.loads((DATA/'road-alignment-adjustments.json').read_text(encoding='utf-8'))
road_alignment_profiles=[]
for config in road_alignment_config['adjustments']:
    road_key='way/'+config['osm_way']
    building=next(g for key,_,g in buildings if key==config['avoid_building'])
    for index,(key,t,geom) in enumerate(roadways):
        if key!=road_key:
            continue
        adjusted=campus_road_network.adjust_road_alignment(geom,building,config)
        roadways[index]=(key,t,adjusted)
        road_alignment_profiles.append({'osm_way':config['osm_way'],
            'centerline_m':[[round(x,4),round(z,4)] for x,z in adjusted.coords],
            'note':config['note']})
        break
    else:
        raise ValueError(f'Missing road for alignment: {road_key}')

# 同一组道路宽度同时用于铺装纹理和立体路缘，地图缺失的宽度均为近似。
ROAD_WIDTHS = {'path':1.8,'footway':2.3,'pedestrian':4.0,'steps':2.2,'cycleway':2.7,'track':2.5,
               'service':5.0,'residential':6.0,'unclassified':6.5,'tertiary':8.0,'secondary':10.0,
               'primary':12.0,'primary_link':7.0,'secondary_link':7.0,'tertiary_link':7.0}
PEDESTRIAN_ROADS = {'path','footway','pedestrian','steps','cycleway','track'}
VEHICLE_ROADS = set(ROAD_WIDTHS)-PEDESTRIAN_ROADS
def road_width(key,t):
    return 14.5 if key in shennan_primary else ROAD_WIDTHS[t['highway']]

ground_vehicle_roads=[(key,t,g) for key,t,g in roadways
                      if t.get('highway') in VEHICLE_ROADS and t.get('bridge') in (None,'no')
                      and t.get('tunnel') in (None,'no','building_passage')
                      and t.get('layer','0')=='0' and key!='way/220071397']
# 先求整体车行铺装，交汇处不应留下另一条路的横向路缘线。
vehicle_surface=unary_union([g.buffer(road_width(key,t)*.5,cap_style='round',join_style='round')
                             for key,t,g in ground_vehicle_roads])
vehicle_node_ways=collections.defaultdict(set)
for key,_,_ in ground_vehicle_roads:
    for nd in ways[key[4:]].findall('nd'):
        vehicle_node_ways[nd.get('ref')].add(key)
road_junctions=unary_union([Point(nodes[ref]).buffer(4.5)
                            for ref,keys in vehicle_node_ways.items() if len(keys)>1 and ref in nodes])
mapped_crossings=[(ref,Point(nodes[ref]),tags(node)) for ref,node in
                  ((n.get('id'),n) for n in root.findall('node')) if ref in nodes
                  and tags(node).get('highway')=='crossing'
                  and (tags(node).get('crossing:markings') in ('zebra','yes')
                       or tags(node).get('crossing') in ('marked','zebra'))]
pedestrian_conflicts=unary_union([g for _,t,g in roadways if t.get('highway') in PEDESTRIAN_ROADS
                                  and t.get('bridge') in (None,'no') and t.get('layer','0')=='0']
                                 +[p for _,p,_ in mapped_crossings])
# 候选按独立配置显式启用；默认保留原道路，全链路共享同一车行面。
road_obstacles=unary_union([g for _,_,g in buildings+waters+pitches])
vehicle_surface,road_override_report=campus_road_overrides.apply_overrides(
    vehicle_surface,ground_vehicle_roads,road_junctions,road_obstacles,DATA,
    pedestrian=pedestrian_conflicts,road_width=road_width)
CAMPUS_RING_IDS={'way/231407864','way/1434756580','way/488453361','way/231664133',
                 'way/231530782','way/222088986'}
def road_marking_polygons():
    """标线只用于已绘制车行道；横道仅取明确标记的地图节点或线段。"""
    markings=[g.difference(road_junctions).buffer(.09,cap_style=2)
              for key,_,g in ground_vehicle_roads if key in CAMPUS_RING_IDS]
    # 学校环路照片中可见靠路缘的细白线；具体路段和漆线宽度按照片近似。
    for key,t,geom in ground_vehicle_roads:
        if key not in CAMPUS_RING_IDS:
            continue
        for line in lines(geom):
            for side in ('left','right'):
                edge=line.parallel_offset(max(.1,road_width(key,t)*.5-.25),side,join_style=2)
                edge=edge.difference(road_junctions.buffer(2))
                markings.append(edge.buffer(.045,cap_style=2).intersection(vehicle_surface.buffer(-.09)))
    for _,point,_ in mapped_crossings:
        candidates=[(g.distance(point),g,road_width(key,t)) for key,t,g in ground_vehicle_roads
                    if g.distance(point)<2.0]
        if not candidates:
            continue
        _,road,width=min(candidates,key=lambda q:q[0])
        pos=road.project(point)
        a,b=road.interpolate(max(0,pos-1.0)),road.interpolate(min(road.length,pos+1.0))
        dx,dz=b.x-a.x,b.y-a.y
        length=math.hypot(dx,dz)
        if length<.1:
            continue
        tx,tz=dx/length,dz/length
        nx,nz=-tz,tx
        for offset in np.arange(-width*.5+.6,width*.5-.3,1.0):
            x,z=point.x+nx*offset,point.y+nz*offset
            bar=LineString(((x-tx*1.3,z-tz*1.3),(x+tx*1.3,z+tz*1.3))).buffer(.23,cap_style=2)
            markings.append(bar.intersection(vehicle_surface))
    # OSM 有些过街点标在 footway=crossing 线段上，避免漏掉；和节点重复时只画一次。
    for _,t,geom in roadways:
        if (t.get('footway')!='crossing' or t.get('bridge') not in (None,'no')
                or t.get('tunnel') not in (None,'no')
                or not (t.get('crossing:markings') in ('zebra','yes')
                        or t.get('crossing') in ('marked','zebra'))):
            continue
        for line in lines(geom):
            if any(line.distance(p)<4 for _,p,_ in mapped_crossings):
                continue
            for distance in np.arange(.5,line.length,1.0):
                p=line.interpolate(float(distance))
                a=line.interpolate(max(0,distance-.5)); b=line.interpolate(min(line.length,distance+.5))
                dx,dz=b.x-a.x,b.y-a.y
                length=math.hypot(dx,dz)
                if length<.05:
                    continue
                tx,tz=-dz/length,dx/length
                bar=LineString(((p.x-tx*1.3,p.y-tz*1.3),(p.x+tx*1.3,p.y+tz*1.3))).buffer(.23,cap_style=2)
                markings.append(bar.intersection(vehicle_surface))
    return unary_union(markings)

road_markings=road_marking_polygons()

# 乔相阁前的森林校巴花园：学校资料确定其在小巴站旁，轮廓和路径按照片估计。
# 坐标由环校路及宿舍轮廓定位；并非现场测绘边界。
qiaoyuan_origin=np.asarray((-280.0,512.82))
qiaoyuan_u=np.asarray((.98058,-.19612))
qiaoyuan_v=np.asarray((.19612,.98058))
def qiaoyuan_xy(u,v):
    return qiaoyuan_origin+qiaoyuan_u*u+qiaoyuan_v*v
def qiaoyuan_rect(u0,u1,v0,v1):
    return Polygon((qiaoyuan_xy(u0,v0),qiaoyuan_xy(u1,v0),
                    qiaoyuan_xy(u1,v1),qiaoyuan_xy(u0,v1)))
qiaoyuan_garden_site=qiaoyuan_rect(-18,18,5.0,25.0)
qiaoxiang_tower=way_polygon(ways['629940894'])
qiaoyuan_path_a=LineString([qiaoyuan_xy(*p) for p in
                            ((-14,6.3),(-11,9.6),(-7.5,12),(-3,13.8),(2,14.5),(8,16.5),(12,20))])
qiaoyuan_path_b=LineString([qiaoyuan_xy(*p) for p in
                            ((-3,13.8),(.5,11.2),(4,8.9),(8,6.0))])
qiaoyuan_door=nearest_points(qiaoxiang_tower.boundary,Point(qiaoyuan_xy(2,28)))[0]
qiaoyuan_interior=np.asarray(qiaoxiang_tower.representative_point().coords[0])-np.asarray(qiaoyuan_door.coords[0])
qiaoyuan_interior/=np.linalg.norm(qiaoyuan_interior)
qiaoyuan_tower_path=LineString([qiaoyuan_xy(1.5,14.4),qiaoyuan_xy(2.0,22),
                                 qiaoyuan_door.coords[0],np.asarray(qiaoyuan_door.coords[0])+qiaoyuan_interior*2.0])
qiaoyuan_walkway=unary_union([p.buffer(1.05,cap_style='round',join_style='round') for p in
                              (qiaoyuan_path_a,qiaoyuan_path_b,qiaoyuan_tower_path)]).difference(vehicle_surface)
qiaoyuan_mapped_sidewalk=LineString(way_points(ways['1272661384']))
qiaoyuan_brick_axis=qiaoyuan_mapped_sidewalk.intersection(qiaoyuan_rect(-26,26,3.5,10.5))
qiaoyuan_brick=qiaoyuan_brick_axis.buffer(1.15,cap_style='round',join_style='round').difference(vehicle_surface)

# 北馆前的较高一组台阶与两侧连廊均有独立 OSM 线位。
north_library_steps = LineString(way_points(ways['232602597']))
north_library_step_site = north_library_steps.buffer(10,cap_style=2)
north_library_step_inner = prep(north_library_step_site)
north_library_step_outer = prep(north_library_step_site.buffer(7))
north_library_links = [LineString(way_points(ways[i])) for i in ('1274485928','231782436')]

# 演会中心：屋顶与入口的方向分别来自 OSM 外轮廓和对应台阶线。
yanhui_site = way_polygon(ways['1455962776'])
yanhui_center = np.asarray((yanhui_site.centroid.x,yanhui_site.centroid.y))
yanhui_edge = np.asarray(way_points(ways['1455962776'])[1])-np.asarray(way_points(ways['1455962776'])[0])
yanhui_u = yanhui_edge/np.linalg.norm(yanhui_edge)
yanhui_v = np.asarray((-yanhui_u[1],yanhui_u[0]))
def yanhui_point(u,v):
    return yanhui_center+yanhui_u*u+yanhui_v*v
def yanhui_rect(u0,v0,u1,v1):
    return Polygon((yanhui_point(u0,v0),yanhui_point(u1,v0),yanhui_point(u1,v1),yanhui_point(u0,v1)))
yanhui_mapped_steps = LineString(way_points(ways['231885484']))
yanhui_step_end = nearest_points(yanhui_site.boundary,Point(yanhui_mapped_steps.coords[-1]))[0]
yanhui_step_line = LineString((yanhui_mapped_steps.coords[0],(yanhui_step_end.x,yanhui_step_end.y)))
yanhui_site_inner = prep(yanhui_site.buffer(4))
yanhui_site_outer = prep(yanhui_site.buffer(16))
yanhui_step_area = yanhui_step_line.buffer(8,cap_style=2)
yanhui_step_inner = prep(yanhui_step_area)
yanhui_step_outer = prep(yanhui_step_area.buffer(7))

# 汇紫楼四段地图轮廓合并后自然形成三个院落，保留真实的围合关系。
huizi_ids = ('230847102','230847100','230847098','230847099')
huizi_parts = {key:way_polygon(ways[key]) for key in huizi_ids}
huizi_site = unary_union(list(huizi_parts.values()))
huizi_courtyards = [Polygon(ring) for ring in huizi_site.interiors]
huizi_north_courtyard = min(huizi_courtyards,key=lambda g:g.centroid.y)
huizi_a = huizi_parts['230847102']
huizi_ab = unary_union([huizi_parts[k] for k in huizi_ids[:2]])
huizi_cd = unary_union([huizi_parts[k] for k in huizi_ids[2:]])
huizi_inner = prep(huizi_site.buffer(5))
huizi_outer = prep(huizi_site.buffer(20))
huizi_side_road = LineString(way_points(ways['232096325'])[:4])
huizi_side_road_outer = prep(huizi_side_road.buffer(22))
huizi_site_guard = prep(huizi_site.buffer(2))
huizi_roof_levels = None
# A 座朝中心广场的长边：旧照可见宽铺装与几级台阶；尺寸为照片近似。
huizi_a_points = list(huizi_a.exterior.coords)
huizi_entry_a = np.asarray(huizi_a_points[0],dtype=float)
huizi_entry_b = np.asarray(huizi_a_points[2],dtype=float)
huizi_entry_axis = huizi_entry_b-huizi_entry_a
huizi_entry_out = np.asarray((-huizi_entry_axis[1],huizi_entry_axis[0]))/np.linalg.norm(huizi_entry_axis)
def huizi_entry_strip(start,end):
    return Polygon((huizi_entry_a+huizi_entry_out*start,huizi_entry_b+huizi_entry_out*start,
                    huizi_entry_b+huizi_entry_out*end,huizi_entry_a+huizi_entry_out*end))
huizi_approach = huizi_entry_strip(0,8.2)
huizi_approach_inner = prep(huizi_approach.buffer(2))
huizi_approach_outer = prep(huizi_approach.buffer(8))
# 汇元楼门廊位置由穿过外墙的人行路线确定，朝向时光广场。
huiyuan_site = way_polygon(ways['230847104'])
huiyuan_entry_path = LineString(way_points(ways['237893526']))
huiyuan_entry_cross = huiyuan_entry_path.intersection(huiyuan_site.boundary)
huiyuan_path_coords = list(huiyuan_entry_path.coords)
huiyuan_out = np.asarray(huiyuan_path_coords[-1])-np.asarray(huiyuan_path_coords[0])
huiyuan_out /= np.linalg.norm(huiyuan_out)
huiyuan_right = np.asarray((huiyuan_out[1],-huiyuan_out[0]))
huiyuan_origin = np.asarray((huiyuan_entry_cross.x,huiyuan_entry_cross.y))
def huiyuan_point(side,depth):
    return huiyuan_origin+huiyuan_right*side+huiyuan_out*depth
def huiyuan_rect(left,right,near,far):
    return Polygon((huiyuan_point(left,near),huiyuan_point(right,near),
                    huiyuan_point(right,far),huiyuan_point(left,far)))
huiyuan_entry_area = huiyuan_rect(-4.5,11.5,-2,8.5).buffer(7)
huiyuan_entry_inner = prep(huiyuan_entry_area)
huiyuan_entry_outer = prep(huiyuan_entry_area.buffer(12))
# 时光广场的地图环线围合日晷绿地；照片显示环线及外侧为连续浅色铺装。
time_square = way_polygon(ways['1274486582'])
time_square_paving = time_square.buffer(8)
time_square_lawn = time_square.buffer(-7)
time_square_inner = prep(time_square_paving.buffer(4))
time_square_outer = prep(time_square_paving.buffer(18))

# 新斋三栋之间的硬质庭院按航拍明部和楼体轮廓估计；树荫处边界未测量。
zhai_buildings = unary_union([way_polygon(ways[i]) for i in ('1558576826','1558576825','1558576824')])
zhai_gaps = list(polygons(zhai_buildings.convex_hull.difference(zhai_buildings)))
zhai_south_apron = Polygon(((291,-131),(322,-146),(324,-139),(312,-132),(301,-118),(291,-113)))
zhai_courtyard = max(zhai_gaps,key=lambda g:g.area).buffer(-1).union(zhai_south_apron)
zhai_courtyard = zhai_courtyard.difference(zhai_buildings.buffer(.2)).difference(vehicle_surface.buffer(.15))
# 汇星楼北侧两块规则绿地使用 OSM 人行路径围合的范围，内部灌木形式按照片近似。
huixing_gardens = (box(-40.0,-361.8,-16.0,-346.8),box(-13.8,-361.8,12.0,-347.2))

# 设计院剖面显示致艺楼两座塔楼与三层裙房处在同一连续平台上。
# 该区低分辨率 DEM 被建筑/周边地物抬高，4 米仅为邻近填海低地的视觉估计。
zhiyi_site = unary_union([way_polygon(ways[i]) for i in ('629940889','629940890','776358621')])
zhiyi_level = 4.0
zhiyi_flat_inner = prep(zhiyi_site.buffer(12))
zhiyi_flat_outer = prep(zhiyi_site.buffer(30))
# 南区教学楼组团中庭的范围和内部草坪由地图分别记录，铺装外观参考航拍。
south_courtyard = way_polygon(ways['724264876'])

# 汇萃楼 OSM 外轮廓的四角按西北、东北、东南、西南顺序；
# 规划局屋顶总平面显示高层沿南、东两边成 L 形，而非位于正中央。
huicui_site = way_polygon(ways['938357240'])
huicui_corners = [np.asarray(p,dtype=float) for p in way_points(ways['938357240'])[:4]]
def huicui_point(u,v):
    a,b,c,d=huicui_corners
    return (1-v)*((1-u)*a+u*b)+v*((1-u)*d+u*c)
def huicui_rect(u0,v0,u1,v1):
    return Polygon((huicui_point(u0,v0),huicui_point(u1,v0),huicui_point(u1,v1),huicui_point(u0,v1)))
huicui_forecourt = huicui_rect(0,-.17,1,0)
huicui_greenbeds = (huicui_rect(.03,-.25,.19,-.17),huicui_rect(.24,-.25,.36,-.17),
                    huicui_rect(.84,-.25,.97,-.17))
huicui_flat_area = unary_union((huicui_site,huicui_forecourt)).buffer(5)
huicui_flat_inner = prep(huicui_flat_area)
huicui_flat_outer = prep(huicui_flat_area.buffer(10))
# 三层连桥位于两座塔楼之间；按航拍与地图墙线对齐，具体端点仍为估计。
zhiyi_bridge_ends = (nearest_points(way_polygon(ways['629940889']).boundary,Point(303.0,347.5))[0],
                     nearest_points(way_polygon(ways['629940890']).boundary,Point(306.8,340.3))[0])

# 汇文楼入口前的铺装及台阶参照照片近似，沿 H3 的地图轮廓确定方向。
huiwen_h3 = way_polygon(ways['231406259'])
_,huiwen_b,huiwen_c,huiwen_d,huiwen_e = [np.asarray(p,dtype=float) for p in list(huiwen_h3.exterior.coords)[:5]]
huiwen_axis = huiwen_d-huiwen_c
huiwen_outward = np.asarray((huiwen_axis[1],-huiwen_axis[0]))/np.linalg.norm(huiwen_axis)
huiwen_approach = Polygon((huiwen_c,huiwen_d,huiwen_d+huiwen_outward*20,huiwen_c+huiwen_outward*20))
# 网格采样间距约十米；向外留缓冲避免双线性插值穿过平台边缘。
huiwen_flat_area = huiwen_approach.buffer(8)
huiwen_approach_inner = prep(huiwen_flat_area)
huiwen_approach_outer = prep(huiwen_flat_area.buffer(12))


class Elevation:
    def __init__(self):
        self.images = {}
        for image in (DATA / 'elevation').glob('*.png'):
            _, x, y = image.stem.split('-')
            self.images[int(x), int(y)] = np.asarray(Image.open(image).convert('RGB'), dtype=np.float32)

    def value(self, x, z):
        lon, lat = LON0 + x / METERS_LON, LAT0 - z / METERS_LAT
        n = 2 ** 15
        tx = (lon + 180.0) / 360.0 * n
        ty = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
        ix, iy = math.floor(tx), math.floor(ty)
        image = self.images.get((ix, iy))
        if image is None:
            raise ValueError(f'Missing DEM tile for {lon}, {lat}: {(ix, iy)}')
        px = min(255, max(0, int((tx - ix) * 256)))
        py = min(255, max(0, int((ty - iy) * 256)))
        r, g, b = image[py, px]
        return float(r * 256 + g + b / 256 - 32768)


elevation = Elevation()
# 2026 年实景显示广场与泳池相邻但有台阶高差，粗 DEM 无法还原这一局部结构。
# 保留地图平面位置；两个平台的绝对高程和 2.4 米高差均为视觉估计，不是测量值。
stone_plaza = way_polygon(ways['231665580'])
stone_house = way_polygon(ways['231572974'])
south_pool = way_polygon(ways['230849087'])
south_pool_house = way_polygon(ways['231665582'])
# 北夏馆由两个露天池和面向泳池的看台组成；地图中两个泳池不是普通湖泊。
beixia_stand = way_polygon(ways['230956723'])
beixia_pools = (way_polygon(ways['230849085']), way_polygon(ways['230849086']))
beixia_deck = unary_union((beixia_stand, *beixia_pools)).convex_hull.buffer(3).difference(beixia_stand)
beixia_site = unary_union((beixia_stand, beixia_deck)).buffer(8)
beixia_inner, beixia_outer = prep(beixia_site), prep(beixia_site.buffer(14))
beixia_level = None

# 高尔夫球场是 OSM golf_course，而非普通 pitch；旧筛选漏掉了其开阔草坪。
golf_course = way_polygon(ways['231522459'])
golf_tee = way_polygon(ways['230956727'])
golf_wooded_corner = affinity.scale(Point(354,-558).buffer(1),xfact=32,yfact=35)
golf_clear = clean(golf_course.buffer(-5).difference(golf_wooded_corner))
golf_flat_area = golf_course.buffer(7)
golf_inner, golf_outer = prep(golf_flat_area), prep(golf_flat_area.buffer(16))
golf_level = None

# 朝向星空广场的东南立面沿 OSM 轮廓确定；入口和前场细节参考学校照片近似。
yuanping_building = way_polygon(ways['230956722'])
yuanping_a = np.asarray(list(yuanping_building.exterior.coords)[3], dtype=float)
yuanping_b = np.asarray(list(yuanping_building.exterior.coords)[4], dtype=float)
yuanping_mid = (yuanping_a + yuanping_b) / 2
yuanping_tangent = (yuanping_b - yuanping_a) / np.linalg.norm(yuanping_b - yuanping_a)
yuanping_outward = np.asarray((-yuanping_tangent[1], yuanping_tangent[0]))
yuanping_approach = Polygon((yuanping_a, yuanping_b, yuanping_b + yuanping_outward*17,
                             yuanping_a + yuanping_outward*17))
yuanping_paving = Polygon((yuanping_a, yuanping_b, yuanping_b + yuanping_outward*32,
                           yuanping_a + yuanping_outward*32)).difference(Point(291,-400).buffer(12))
yuanping_flat_area = yuanping_approach.buffer(6)
yuanping_inner, yuanping_outer = prep(yuanping_flat_area), prep(yuanping_flat_area.buffer(12))
stone_level = elevation.value(115,115)+2.0
pool_level = stone_level-2.4
stone_site = unary_union((stone_plaza,stone_house,south_pool,south_pool_house)).buffer(11)
stone_site_interior, stone_site_outer = prep(stone_site),prep(stone_site.buffer(14))
# 粗高程对相邻小水域可能给出相差数米的采样，导致湖边地形穿过水面。
# 将缓冲区相接的水域分组，统一采用组内最大水面的近似高程。
lake_heights = []
remaining_waters = list(waters)
while remaining_waters:
    group = [remaining_waters.pop(0)]
    pending = 0
    while pending < len(group):
        current = group[pending][2]
        connected = [item for item in remaining_waters if current.distance(item[2]) <= 24]
        group.extend(connected)
        remaining_waters = [item for item in remaining_waters if item not in connected]
        pending += 1
    largest = max(group, key=lambda item: item[2].area)[2].representative_point()
    level = elevation.value(largest.x, largest.y)
    for _, _, geom in group:
        lake_heights.append((prep(geom.buffer(12)), geom, level))

# 2023 年湖区照片与卫星图可辨认两处弧形亲水台阶；范围、高差均为视觉估计。
upper_lake = next(g for key,_,g in waters if key == 'way/234739309')
upper_lake_level = next(y for _,g,y in lake_heights if g.equals(upper_lake))
wenshan_west_site = Point(-377,190).buffer(18)
wenshan_east_site = affinity.scale(Point(-231,211).buffer(1),xfact=15,yfact=21)
trefoil_centers = ((-420,192),(-413,187),(-406,182))
trefoil_patio = unary_union([LineString(trefoil_centers).buffer(3.2,cap_style='flat')]
                           + [Point(x,z).buffer(3.6) for x,z in trefoil_centers])
wenshan_sites = unary_union((wenshan_west_site,wenshan_east_site,trefoil_patio))
wenshan_site_inner = prep(wenshan_sites.buffer(5))
wenshan_site_outer = prep(wenshan_sites.buffer(18))
ancient_stone_config=json.loads((DATA/'wenshan-ancient-stone.json').read_text(encoding='utf-8'))
ancient_stone_site=Point(*ancient_stone_config['position_xz_m'])
# 旧地图的步行面与按照片估计的石刻草地重叠；暂保留石刻周围开敞绿地。
ancient_stone_reserve=ancient_stone_site.buffer(8.5)
kaifeng_config=json.loads((DATA/'kaifeng-canteen.json').read_text(encoding='utf-8'))
kaifeng_points=way_points(ways[kaifeng_config['osm_way']])
kaifeng_a,kaifeng_b,kaifeng_u,kaifeng_v=campus_kaifeng.front_geometry(kaifeng_points,kaifeng_config)
kaifeng_forecourt=unary_union([
    Polygon([kaifeng_a+kaifeng_u*s+kaifeng_v*d for s,d in corners])
    for corners in (((13.3,0),(23.7,0),(23.7,9.1),(13.3,9.1)),
                    ((29.7,0),(46.7,0),(46.7,5.4),(29.7,5.4)),
                    ((44.25,5.4),(45.8,5.4),(45.8,11.1),(44.25,11.1)))])
trefoil_inner = prep(trefoil_patio.buffer(5))
trefoil_outer = prep(trefoil_patio.buffer(15))

# 田径场应是平整场地，低分辨率高程在看台和跑道附近有明显起伏。
flat_sports = []
for _, t, geom in pitches:
    if t.get('leisure') == 'stadium':
        samples = [elevation.value(x, z) for x, z in list(geom.exterior.coords)[::5]]
        flat_sports.append((prep(geom), float(np.median(samples))))

# 跑道采用连续直道和半圆，而不是将不规则看台外轮廓向内缩放。
# 位置依足球场 OSM 轮廓和航拍近似；半径及长度为标准场地量级，不是竣工测量。
stadium_footprint = way_polygon(ways['230848519'])
stadium_field = way_polygon(ways['231813499'])
stadium_center = stadium_field.centroid
stadium_angle = -.7
stadium_half_straight, stadium_inner_radius, stadium_lane_width, stadium_lanes = 38.1, 39.0, 1.22, 10
stadium_axis = affinity.rotate(LineString(((stadium_center.x,stadium_center.y-stadium_half_straight),
                                         (stadium_center.x,stadium_center.y+stadium_half_straight))),
                               stadium_angle,origin=(stadium_center.x,stadium_center.y))
stadium_infield = stadium_axis.buffer(stadium_inner_radius,quad_segs=48)
stadium_track_outer = stadium_axis.buffer(stadium_inner_radius+stadium_lanes*stadium_lane_width,quad_segs=48)

# 天人广场的雕塑基座位于水景旁的小型硬质平台；粗高程需局部调平。
plaza_flat = prep(Point(-108,36).buffer(19))
plaza_y = next(y for prepared, _, y in reversed(lake_heights) if prepared.covers(Point(-108,36)))


def height_at(x, z):
    p = Point(x, z)
    for prepared, _, y in lake_heights:
        if prepared.covers(p):
            return y
    for prepared, y in flat_sports:
        if prepared.covers(p):
            return y
    if plaza_flat.covers(p):
        return plaza_y
    return elevation.value(x, z)


terrain_grid = None
terrain_normals = None
terrain_patches = []
wenshan_shore_terrain = None
road_building_bases = {}
road_building_groups = {}
zhounan_portal = None


def surface_height(x, z):
    # 建筑、桥梁和树木使用最终网格的插值高度，避免与地形错层。
    if terrain_grid is None:
        raise RuntimeError('Terrain must be generated first')
    for patch in terrain_patches:
        if patch.contains(x,z):
            return patch.height(x,z)
    nz, nx = terrain_grid.shape
    u = max(0, min(nx - 1, (x - X0) / (X1 - X0) * (nx - 1)))
    v = max(0, min(nz - 1, (z - Z0) / (Z1 - Z0) * (nz - 1)))
    i, j = min(int(u), nx - 2), min(int(v), nz - 2)
    a, b = u - i, v - j
    return float((terrain_grid[j, i] * (1 - a) + terrain_grid[j, i + 1] * a) * (1 - b)
                 + (terrain_grid[j + 1, i] * (1 - a) + terrain_grid[j + 1, i + 1] * a) * b)


def rendered_terrain_height(x,z):
    """与实际地形三角面一致的插值，供宽台阶消除边缘穿插。"""
    for patch in terrain_patches:
        if patch.contains(x,z):
            return patch.height(x,z)
    nz,nx=terrain_grid.shape
    u=max(0,min(nx-1,(x-X0)/(X1-X0)*(nx-1)))
    v=max(0,min(nz-1,(z-Z0)/(Z1-Z0)*(nz-1)))
    i,j=min(int(u),nx-2),min(int(v),nz-2)
    a,b=u-i,v-j
    p,q,r,s=terrain_grid[j,i],terrain_grid[j,i+1],terrain_grid[j+1,i],terrain_grid[j+1,i+1]
    if b>=a:
        return float((1-b)*p+(b-a)*r+a*s)
    return float((1-a)*p+b*s+(a-b)*q)


def coarse_terrain_normal(x,z):
    nz,nx=terrain_grid.shape
    u=max(0,min(nx-1,(x-X0)/(X1-X0)*(nx-1)))
    v=max(0,min(nz-1,(z-Z0)/(Z1-Z0)*(nz-1)))
    i,j=min(int(u),nx-2),min(int(v),nz-2)
    a,b=u-i,v-j
    p,q,r,s=terrain_normals[j,i],terrain_normals[j,i+1],terrain_normals[j+1,i],terrain_normals[j+1,i+1]
    return (1-b)*p+(b-a)*r+a*s if b>=a else (1-a)*p+b*s+(a-b)*q


def rendered_terrain_normal(x,z):
    """沿实际地形三角面插值法线，细分补丁与原网格共用边界法线。"""
    for patch in terrain_patches:
        if patch.normals is not None and patch.contains(x,z):
            return patch.sample(patch.normals,x,z)
    return coarse_terrain_normal(x,z)


def terrain_cell_parts(i,j):
    """把单个粗网格单元细分为与可见地形一致的单元，供铺装和地面共用。"""
    nz,nx=terrain_grid.shape
    dx,dz=(X1-X0)/(nx-1),(Z1-Z0)/(nz-1)
    x,z=X0+i*dx,Z0+j*dz
    patch=next((p for p in terrain_patches if p.contains(x+dx*.5,z+dz*.5)),None)
    if patch is None:
        yield x,z,dx,dz
        return
    sx,sz=round(dx/patch.spacing),round(dz/patch.spacing)
    for a in range(sx):
        for b in range(sz):
            yield x+a*patch.spacing,z+b*patch.spacing,patch.spacing,patch.spacing


def smooth_campus_road_terrain(heights):
    """沿地图车行路网缓和粗地形尖折，保留整体高差；全部高程仍属估计。"""
    nz,nx=heights.shape
    dx,dz=(X1-X0)/(nx-1),(Z1-Z0)/(nz-1)
    region=prep(campus.buffer(3))
    points,lookup,edges=[],{},[]
    vehicle_types={'service','residential','unclassified'}
    bridge_refs={n.get('ref') for osm_id in ('574956548','776358616')
                 for n in (ways[osm_id].findall('nd')[0],ways[osm_id].findall('nd')[-1])}
    def vertex(key,p):
        if key not in lookup:
            lookup[key]=len(points)
            points.append(p)
        return lookup[key]
    for osm_id,item in ways.items():
        t=tags(item)
        if (t.get('highway') not in vehicle_types or t.get('bridge') not in (None,'no')
                or t.get('tunnel') not in (None,'no') or t.get('layer','0')!='0'
                or osm_id=='220071397'):
            continue
        refs=[n.get('ref') for n in item.findall('nd') if n.get('ref') in nodes]
        for segment,(ra,rb) in enumerate(zip(refs[:-1],refs[1:])):
            a,b=np.asarray(nodes[ra]),np.asarray(nodes[rb])
            distance=float(np.linalg.norm(b-a))
            if distance<.1 or not region.covers(Point((a+b)*.5)):
                continue
            count=max(1,int(math.ceil(distance/6)))
            previous=vertex(ra,a)
            for step in range(1,count+1):
                p=a+(b-a)*(step/count)
                current=vertex(rb if step==count else f'{osm_id}:{segment}:{step}',p)
                if current!=previous:
                    edges.append((previous,current,distance/count))
                previous=current
    if not edges:
        raise ValueError('No ground-level campus roads available for terrain fitting')
    xy=np.asarray(points)
    edges=np.asarray(edges)
    left,right=edges[:,0].astype(int),edges[:,1].astype(int)
    length=edges[:,2]
    u=np.clip((xy[:,0]-X0)/dx,0,nx-1)
    v=np.clip((xy[:,1]-Z0)/dz,0,nz-1)
    i,j=np.minimum(u.astype(int),nx-2),np.minimum(v.astype(int),nz-2)
    a,b=u-i,v-j
    initial=((1-b)*((1-a)*heights[j,i]+a*heights[j,i+1])
             +b*((1-a)*heights[j+1,i]+a*heights[j+1,i+1]))
    fitted=initial.copy()
    fixed=np.asarray([lookup[r] for r in bridge_refs if r in lookup],dtype=int)
    degree=np.bincount(np.r_[left,right],minlength=len(xy))
    # 共享 OSM 节点只保留一个高度，使多条路在真正相交处一起过渡。
    for _ in range(50):
        neighbors=np.bincount(np.r_[left,right],weights=np.r_[fitted[right],fitted[left]],minlength=len(xy))
        fitted=.35*initial+.65*neighbors/np.maximum(1,degree)
        fitted[fixed]=initial[fixed]
    # .16 为抑制粗栅格极端尖折的视觉拟合参数，不是实际道路设计坡度。
    for _ in range(600):
        diff=fitted[right]-fitted[left]
        excess=np.sign(diff)*np.maximum(0,np.abs(diff)-.16*length)*.5
        adjustment=np.bincount(np.r_[left,right],weights=np.r_[excess,-excess],minlength=len(xy))
        active=np.bincount(np.r_[left,right],weights=np.r_[excess!=0,excess!=0],minlength=len(xy))
        fitted+=adjustment/np.maximum(1,active)
        fitted[fixed]=initial[fixed]
    segments=[LineString((xy[a],xy[b])) for a,b in zip(left,right)]
    road_tree=STRtree(segments)
    xx,zz=np.meshgrid(np.linspace(X0,X1,nx),np.linspace(Z0,Z1,nz))
    queries=shapely.points(xx.ravel(),zz.ravel())
    indexes,distances=road_tree.query_nearest(queries,max_distance=23,all_matches=False,return_distance=True)
    grid_id,edge_id=indexes
    A,B=xy[left[edge_id]],xy[right[edge_id]]
    P=np.c_[xx.ravel()[grid_id],zz.ravel()[grid_id]]
    t=np.clip(np.sum((P-A)*(B-A),axis=1)/np.sum((B-A)**2,axis=1),0,1)
    target=fitted[left[edge_id]]*(1-t)+fitted[right[edge_id]]*t
    weight=np.clip((23-distances)/15,0,1)
    weight=weight*weight*(3-2*weight)
    # 湖水、重点建筑、楼梯与已校准的北门缓坡保持原有关系，并在边缘平滑过渡。
    protected_names=('legong_road_outer','legong_entry_outer','beixia_outer','golf_outer',
                     'huiwen_approach_outer','huizi_outer','huizi_approach_outer','huiyuan_entry_outer',
                     'time_square_outer','yanhui_site_outer','yanhui_step_outer','north_library_step_outer',
                     'zhiyuan_entry_outer','zhiyi_flat_outer','huicui_flat_outer','yuanping_outer',
                     'stone_site_outer','wenshan_site_outer','trefoil_outer','plaza_flat')
    protected=unary_union([unary_union([g for _,_,g in buildings+surrounding]).buffer(1)]
                          +[globals()[name].context for name in protected_names]
                          +[g.buffer(4) for _,_,g in waters])
    protected_tree=STRtree(list(shapely.get_parts(protected)))
    index,distance=protected_tree.query_nearest(shapely.points(P[:,0],P[:,1]),max_distance=6,
                                                all_matches=False,return_distance=True)
    fade=np.ones(len(P))
    fade[index[0]]=np.clip(distance/6,0,1)
    weight*=fade*fade*(3-2*fade)
    original=heights.ravel()[grid_id].copy()
    correction=np.clip((target-original)*weight,-2.5,2.5)
    heights.ravel()[grid_id]=np.maximum(2.0,original+correction)
    profile={'road_graph_vertices':len(xy),'road_graph_segments':len(edges),
             'terrain_cells_changed_over_5cm':int(np.count_nonzero(np.abs(correction)>.05)),
             'max_terrain_adjustment_m':round(float(np.max(np.abs(correction))),2),
             'note':'OSM 地面车行线上的估计纵坡与横向整形，保留山地大尺度起伏；不是测绘高程'}
    (DATA/'road-profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')


def smooth_lakeside_road_terrain(heights):
    """减小文山湖东侧道路受粗地形和湖岸调平叠加形成的尖折。"""
    road=LineString(way_points(ways['231530782']))
    start,end=50.0,215.0
    positions=np.linspace(start,end,67)
    # 保留道路整体高差，以邻点平滑估计纵坡，再限制过于尖锐的局部折线。
    original=np.asarray([rendered_terrain_height(road.interpolate(float(d)).x,road.interpolate(float(d)).y)
                         for d in positions])
    second=np.diff(np.eye(len(positions)),n=2,axis=0)
    levels=np.linalg.solve(np.eye(len(positions))+16*second.T@second,original)
    for _ in range(400):
        difference=np.diff(levels)
        excess=np.sign(difference)*np.maximum(0,np.abs(difference)-.17*(positions[1]-positions[0]))*.5
        levels[:-1]+=excess
        levels[1:]-=excess
        levels[0],levels[-1]=original[0],original[-1]
    protected=prep(unary_union([g.buffer(2) for _,_,g in buildings+waters]))
    before=heights.copy()
    nz,nx=heights.shape
    for j,z in enumerate(np.linspace(Z0,Z1,nz)):
        for i,x in enumerate(np.linspace(X0,X1,nx)):
            point=Point(float(x),float(z))
            along=road.project(point)
            if not start<along<end:
                continue
            distance=road.distance(point)
            if distance>=22 or protected.covers(point):
                continue
            across=np.clip((22-distance)/12,0,1)
            across=across*across*(3-2*across)
            blend=np.clip(min(along-start,end-along)/18,0,1)
            blend=blend*blend*(3-2*blend)
            # 湖岸台阶与水面仍由独立模型决定，越接近湖岸保护区权重越低。
            protection=min(wenshan_sites.distance(point)/7,1)
            correction=(float(np.interp(along,positions,levels))-before[j,i])*across*blend*protection
            heights[j,i]=max(2.0,before[j,i]+np.clip(correction,-7.0,7.0))


def calibrate_qiaoyuan_terrain(heights):
    """以 2013 年设计图的标高点约束乔苑，平滑连接相邻路面与建筑。"""
    # 历史图纸与 OSM 用三栋楼定位，点位有配准误差，也不是当前实测高程。
    reference=json.loads((DATA/'qiaoyuan-terrain-2013.json').read_text(encoding='utf-8'))
    road=LineString(way_points(ways['488453361'])[-10:])
    anchors=[(road.project(Point(x,z)),y) for x,z,y in reference['roadside_height_labels_estimate_m']]
    anchors += [(0,surface_height(*road.coords[0])),
                (road.length,surface_height(*road.coords[-1]))]
    anchors.sort()
    positions,levels=zip(*anchors)
    def road_y(p):
        return float(np.interp(road.project(p),positions,levels))

    towers={key:(way_polygon(ways[key]),level) for key,level in
            reference['building_reference_levels_m'].items()}
    # 路边老建筑未见绝对标高，随道路近似调整底部；楼体内部依旧保持平整。
    neighbor_ids=('230935217','231065255','231304390','231304398','231304400')
    neighbors={key:way_polygon(ways[key]) for key in neighbor_ids}
    neighbor_y={key:road_y(geom.representative_point())+.45 for key,geom in neighbors.items()}
    region=prep(unary_union([road.buffer(42)]+[geom.buffer(28) for geom,_ in towers.values()]))
    tower_pads=[(prep(geom.buffer(3)),level) for geom,level in towers.values()]
    neighbor_pads=[(prep(geom.buffer(2)),neighbor_y[key]) for key,geom in neighbors.items()]
    original=heights.copy()
    values=heights.copy()
    nz,nx=heights.shape
    fixed=np.ones((nz,nx),dtype=bool)
    for j,z in enumerate(np.linspace(Z0,Z1,nz)):
        for i,x in enumerate(np.linspace(X0,X1,nx)):
            if not (-500<x<-130 and 435<z<680):
                continue
            p=Point(x,z)
            if not region.covers(p):
                continue
            fixed[j,i]=False
            for pad,y in tower_pads+neighbor_pads:
                if pad.covers(p):
                    fixed[j,i]=True
                    values[j,i]=y
                    break
            if not fixed[j,i] and road.distance(p)<6.7:
                fixed[j,i]=True
                values[j,i]=road_y(p)
    # 在旧地形、道路和楼宇之间作连续插值；保留整体坡势而非平整整片校园。
    for _ in range(350):
        p=np.pad(values,1,mode='edge')
        average=(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])*.25
        values[~fixed]=(.98*average+.02*original)[~fixed]
    heights[:]=values


def smooth_xiyuan_shop_street(heights):
    """保留阁区至 7-11 一带的坡势，缓和粗高程在街口产生的突跳。"""
    # 这段坡路由使用者确认，但没有实测坡度；端点取当前粗高程，沿线只作近似插值。
    street=LineString(way_points(ways['231664125']))
    z0,z1=street.bounds[1],street.bounds[3]
    start=street.interpolate(0)
    end=street.interpolate(street.length)
    if start.y>end.y:
        start,end=end,start
    y0=rendered_terrain_height(start.x,start.y)
    y1=rendered_terrain_height(end.x,end.y)
    original=heights.copy()
    values=heights.copy()
    fixed=np.ones(heights.shape,dtype=bool)
    nz,nx=heights.shape
    for j,z in enumerate(np.linspace(Z0,Z1,nz)):
        for i,x in enumerate(np.linspace(X0,X1,nx)):
            if not (z0-20<z<z1+20 and -458<x<-396):
                continue
            point=Point(x,z)
            distance=street.distance(point)
            if distance>25:
                continue
            fixed[j,i]=False
            # 街面及两侧窄人行带共用连续坡面，不把道路摊平成水平面。
            if distance<8.5 and z0<=z<=z1:
                fixed[j,i]=True
                values[j,i]=y0+(y1-y0)*(z-z0)/(z1-z0)
    for _ in range(300):
        p=np.pad(values,1,mode='edge')
        average=(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])*.25
        values[~fixed]=(.97*average+.03*original)[~fixed]
    heights[:]=values


def uv(x, z):
    return (x - X0) / (X1 - X0), (z - Z0) / (Z1 - Z0)


def pixel(point):
    u, v = uv(*point)
    return int(u * (RES - 1)), int(v * (RES - 1))


def draw_polygon(draw, geom, fill, hole_fill=None):
    for poly in polygons(geom):
        draw.polygon([pixel(p) for p in poly.exterior.coords], fill=fill)
        if hole_fill is not None:
            for interior in poly.interiors:
                draw.polygon([pixel(p) for p in interior.coords], fill=hole_fill)


def draw_line(draw, geom, fill, width):
    width = max(1, round(width * RES / (X1 - X0)))
    for line in lines(geom):
        coords = [pixel(p) for p in line.coords]
        if len(coords) > 1:
            draw.line(coords, fill=fill, width=width, joint='curve')


def make_ground_texture():
    # 背景色取材于可再分发的草地材质；道路和水体依据矢量数据绘制。
    grass = Image.open(ASSETS / 'grass.ppm').convert('RGB').resize((512, 512))
    grass = ImageEnhance.Color(grass).enhance(.72)
    grass = ImageEnhance.Brightness(grass).enhance(1.10)
    tile = Image.blend(Image.new('RGB',(512,512),'#78966a'),grass,.22)
    ground = Image.new('RGB', (RES, RES))
    for x in range(0, RES, 512):
        for y in range(0, RES, 512):
            ground.paste(tile, (x, y))
    d = ImageDraw.Draw(ground)
    # 中庭主要为浅色铺装；随后以 OSM 草坪多边形恢复其中的绿地。
    draw_polygon(d,south_courtyard,'#c7c4b7')
    # 规划图在艺术楼北侧标明硬质铺装广场和若干绿化带；颜色为照片近似。
    draw_polygon(d,huicui_forecourt,'#c9c6ba')
    for _, t, geom in parks:
        if t.get('natural') == 'wood' or t.get('landuse') == 'forest':
            draw_polygon(d, geom, '#6d8d5d')
        else:
            draw_polygon(d, geom, '#8da96e')
    # 从卫星图和练习场照片保留草坪及边缘树带；地图道路仍在随后绘制。
    draw_polygon(d,golf_clear,'#769b51')
    draw_polygon(d,zhai_courtyard,'#b9b7ab')
    # 航拍显示广场中心是绿篱分区的圆形草地，周边为浅色硬质铺装。
    draw_polygon(d,time_square_paving,'#c1c0b8')
    draw_polygon(d,time_square_lawn,'#799052')
    # 汇紫楼中庭航拍未见草坪或高大树冠，院内铺装色调为概括。
    for court in huizi_courtyards:
        draw_polygon(d,court,'#c1c0b4')
    for garden in huixing_gardens:
        draw_polygon(d,garden,'#87a66f')
    for garden in huicui_greenbeds:
        draw_polygon(d,garden,'#8da96e')
    # 致工楼停车位范围取地图轮廓，铺装及逐个车位线以卫星图概括。
    draw_polygon(d,zhigong_parking,'#8e918a')
    # 只绘制地面路段；标有桥梁的道路留给三维高架几何，避免凭空横穿白石路。
    pedestrian_areas=[]
    for group in ('path','footway','pedestrian','steps','cycleway','track'):
        for key, t, geom in roadways:
            if t.get('highway') != group or t.get('bridge') not in (None, 'no'):
                continue
            if t.get('area')=='yes' and geom.geom_type=='LineString' and geom.is_ring:
                if key not in campus_road_network.MODELED_PEDESTRIAN_AREAS:
                    # 优先尊重地图中的面状广场；石刻附近留给独立景观模型。
                    pedestrian_areas.append(clean(Polygon(geom.coords).difference(ancient_stone_reserve)))
            else:
                draw_line(d,geom,'#d4cfbc',road_width(key,t))
    if pedestrian_areas:
        # 利用蒙版保留内环与原有底色，不能把面状铺装的孔洞直接填满。
        mask=Image.new('L',(RES,RES),0)
        draw_polygon(ImageDraw.Draw(mask),unary_union(pedestrian_areas),'white','black')
        ground.paste('#d4cfbc',mask=mask)
    # 桥端下挖处由立体平台铺装，擦除落在边坡上的旧平面步道色块。
    mask=Image.new('L',(RES,RES),0)
    draw_polygon(ImageDraw.Draw(mask),zhounan_portal['reserved'],'white',0)
    ground.paste('#78966a',mask=mask)
    # 联合面一次性画路缘和沥青，避免道路交叉时路缘横穿另一条路。
    mask=Image.new('L',(RES,RES),0)
    draw_polygon(ImageDraw.Draw(mask),vehicle_surface.buffer(.22),'white',0)
    ground.paste('#b8b9aa',mask=mask)
    mask=Image.new('L',(RES,RES),0)
    draw_polygon(ImageDraw.Draw(mask),vehicle_surface,'white',0)
    ground.paste('#777d79',mask=mask)
    for key, t, geom in roadways:
        if t.get('highway') not in ('primary', 'secondary', 'tertiary') or t.get('bridge') not in (None, 'no'):
            continue
        # OSM 只给出道路中心线；按主干路的常见划线方式近似，不当作实测标线。
        for line in lines(geom):
            offsets=(-3.6,0,3.6) if key in shennan_primary else (0,)
            for offset in offsets:
                marking=line.offset_curve(offset) if offset else line
                for section in lines(marking):
                    for dist in np.arange(0.0, section.length, 10.0):
                        a, b = section.interpolate(dist), section.interpolate(min(dist + 4.0, section.length))
                        if a.distance(b) > .1:
                            draw_line(d, LineString((a, b)), '#d8d9c9', .20)
    # 环道中线和 OSM 明确标注的横道以实际路面位置绘制，线型为照片近似。
    draw_polygon(d,road_markings,'#e1e2d8')
    # 2022 年航拍中立德门两道车行线在门口汇成一片连续铺装，中央有交通岛。
    # 两侧线位来自 OSM；汇合区边缘和白线依照片近似，不是测绘结果。
    draw_polygon(d,box(-446,-63.5,-419,-44.5),'#777d79')
    draw_polygon(d,box(-419,-62.25,-397,-45.75),'#777d79')
    for z in (-62.0,-46.0):
        draw_line(d,LineString(((-417.5,z),(-399,z))),'#d8d9cf',.35)
    for x in (-418,-414,-410,-406,-402):
        draw_line(d,LineString(((x,-54),(x+2.0,-54))),'#dddcd2',.35)
    # 右侧红色步道由独立网格绘制，不再烘焙到地面底图。
    # 白石路两侧校门不止是一条 6.5 米路：入口铺装尺度依航拍近似。
    # 避开白石路本身；通行线位仍与 OSM 路网一致。
    draw_polygon(d,liyan_entry,'#c0c1b8')
    draw_line(d,liyan_entry_axis,'#8a8e88',11.0)
    draw_polygon(d,zhiyuan_entry,'#bfc2bb')
    # 两栋研究教学楼之间为开放硬质庭院；铺装范围依航拍近似。
    draw_polygon(d,zhizhi_courtyard,'#c9c7bb')
    draw_polygon(d,zhili_a_entry,'#c5c4b9')
    draw_polygon(d,zhigong_forecourt,'#c4c4bc')
    for z in (649.0,663.0,677.0,691.0):
        for x in np.arange(81.0,155.0,2.6):
            stripe=LineString(((x,z),(x,z+5.0))).intersection(zhigong_parking.buffer(-.5))
            draw_line(d,stripe,'#d8d9ce',.14)
        edge=LineString(((79.0,z),(156.0,z))).intersection(zhigong_parking.buffer(-.5))
        draw_line(d,edge,'#c7c9bd',.12)
    for key, t, geom in pitches:
        # 这两组球场位于建筑顶层，不能再画到地面纹理上。
        if key in ('way/231692661', 'way/231692664', 'way/231692665',
                   'way/1081134204', 'way/1081134205', 'way/1081134206'):
            continue
        if t.get('leisure') == 'stadium':
            # 跑道、草坪与标线另用细网格绘制；此处保留绿色的场边缓冲区。
            draw_polygon(d, geom, '#658c79')
        elif key == 'way/231813499':
            # 足球草坪在细网格中绘制，避免与底图的粗像素边界重叠。
            continue
        elif t.get('sport') in ('soccer', 'futsal'):
            draw_polygon(d, geom, '#739b5d')
            draw_line(d, geom.boundary, '#e4ead9', .6)
        elif t.get('sport') in ('basketball', 'tennis'):
            draw_polygon(d, geom, '#a96557' if t.get('sport') == 'basketball' else '#698b92')
            draw_line(d, geom.boundary, '#efecdb', .5)

    # 2023 年天人广场航拍显示中心为开敞草坪、两条主轴与曲线汀步。
    # 以下根据照片与浑天仪/水体相对位置近似配准，不能当作测绘图。
    plaza = box(-127,-45,-53,78)
    draw_polygon(d,plaza,'#8da96e')
    def plaza_point(u,v):
        z=36-(u-298)/5.0
        return (-108+(v-374)/5.0+.072*(z-36),z)
    for endpoints,width in (((90,374,700,374),4.3),((298,374,333,650),3.2),
                            ((315,360,475,180),3.5)):
        u0,v0,u1,v1=endpoints
        draw_line(d,LineString((plaza_point(u0,v0),plaza_point(u1,v1))),'#dedbcf',width)
    # 将航拍主轴端部接到 OSM 记载的北馆方向步道。
    draw_line(d,LineString((plaza_point(700,374),(-114,-47))),'#dedbcf',4.0)
    for control in (((160,325),(145,510),(350,495),(550,600)),
                    ((410,390),(402,465),(460,500),(470,560)),
                    ((402,269),(580,335),(510,430),(690,560))):
        points=[]
        for i in range(81):
            t=i/80; s=1-t
            u=sum(w*p[0] for w,p in zip((s**3,3*s*s*t,3*s*t*t,t**3),control))
            v=sum(w*p[1] for w,p in zip((s**3,3*s*s*t,3*s*t*t,t**3),control))
            points.append(plaza_point(u,v))
        line=LineString(points)
        # 小段铺石而非连续宽路，空隙在近景中仍可分辨。
        for dist in np.arange(0,line.length,.95):
            a,b=line.interpolate(dist),line.interpolate(min(dist+.68,line.length))
            if a.distance(b)>.05:
                draw_line(d,LineString((a,b)),'#e0dfd4',1.0)
    for _, _, geom in waters:
        draw_polygon(d, geom, '#6e9e9a', '#839c72')
    # 航拍及地面照片可见散落的练习球；点位由固定种子生成，并非实测。
    golf_rng = random.Random(231522459)
    minx,minz,maxx,maxz=golf_clear.bounds
    prepared_golf = prep(golf_clear.buffer(-2))
    for _ in range(850):
        point=Point(golf_rng.uniform(minx,maxx),golf_rng.uniform(minz,maxz))
        if prepared_golf.covers(point):
            x,y=pixel((point.x,point.y))
            d.ellipse((x-1,y-1,x+1,y+1),fill='#e2e5cc')
    # 体育馆前的浅色铺装保留与星空广场相连的路线；尺寸按照片估计。
    draw_polygon(d,yuanping_paving,'#c7c5bb')
    # 仅在硬质铺装内部画稀疏分缝，真实石材尺寸和走向未实测。
    for distance in np.arange(0,33,3):
        stripe=LineString((yuanping_a+yuanping_outward*distance,
                           yuanping_b+yuanping_outward*distance)).intersection(yuanping_paving)
        draw_line(d,stripe,'#b9b9b1',.18)
    for along in np.arange(0,np.linalg.norm(yuanping_b-yuanping_a),4):
        start=yuanping_a+yuanping_tangent*along
        draw_line(d,LineString((start,start+yuanping_outward*32)).intersection(yuanping_paving),'#b9b9b1',.16)
    # 广场自身为连续浅色铺装；旧道路底图不能在圆形广场内部留下暗色斑块。
    draw_polygon(d,time_square_paving,'#c1c0b8')
    draw_polygon(d,time_square_lawn,'#799052')
    ground.resize((GROUND_TEXTURE_SIZE,GROUND_TEXTURE_SIZE),Image.Resampling.LANCZOS).save(ASSETS / 'campus-ground.ppm')


def facade_texture(name, wall, glass, border, pattern='windows'):
    # 一格纹理对应约 4m × 3.4m；不同材质均为本项目生成。
    img = Image.new('RGB', (256, 256), wall)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 255, 14), fill=border)
    d.rectangle((0, 238, 255, 255), fill=border)
    if pattern == 'windows':
        for x in (17, 136):
            d.rectangle((x, 36, x + 103, 216), fill=glass)
            d.rectangle((x + 4, 39, x + 99, 212), outline=border, width=5)
            d.line((x + 52, 40, x + 52, 211), fill=border, width=4)
            d.line((x + 5, 132, x + 98, 132), fill=border, width=4)
            d.line((x + 8, 46, x + 86, 120), fill=tuple(min(255, c + 10) for c in glass), width=2)
    else:
        for x, y, w, h in ((19, 40, 66, 74), (141, 45, 70, 78), (86, 137, 66, 69), (183, 145, 45, 59)):
            d.rectangle((x, y, x + w, y + h), fill=glass, outline=border, width=5)
    img.save(ASSETS / (name + '.ppm'))


def huizi_roof_textures():
    # 依据航拍生成浅色、略有风化的屋面，不把实景照片作为贴图。
    size=512
    axis=np.arange(size,dtype=np.float32)/size
    u,v=np.meshgrid(axis,axis)
    rng=random.Random(230847102)
    noise=np.zeros((size,size),dtype=np.float32)
    for _ in range(24):
        fx,fz=rng.randint(-9,9),rng.randint(-9,9)
        if fx==0 and fz==0: continue
        amplitude=.65/(1+.16*math.hypot(fx,fz))
        noise+=amplitude*np.sin(math.tau*(fx*u+fz*v)+rng.random()*math.tau)
    base=np.array((216,214,204),dtype=np.float32)
    # 轻微颗粒保留混凝土质感，避免远看变成大块云纹。
    grain=np.random.default_rng(230847102).normal(0,.75,(size,size)).astype(np.float32)
    color=np.clip(base+(noise+grain)[:,:,None],0,255).astype(np.uint8)
    Image.fromarray(color,'RGB').save(ASSETS/'huizi-roof.ppm')
    # 两组屋顶设备的深色格栅由程序绘制，具体机械类型尚未确认。
    img=Image.new('RGB',(256,128),(211,213,207));d=ImageDraw.Draw(img)
    d.rectangle((5,13,250,116),fill=(85,91,86),outline=(183,187,178),width=5)
    for x in range(15,250,26):
        d.rectangle((x,20,x+17,109),fill=(48,56,54))
        d.line((x+2,27,x+15,27),fill=(128,137,130),width=2)
    img.save(ASSETS/'huizi-equipment.ppm')


def landmark_facades():
    # 不复制实景照片：以照片中可见的窗洞规律制作可重复的示意材质。
    # 汇研楼由浅蓝玻璃与连续的白色竖向构件组成，按校方 2026 年照片重新绘制。
    huiyan = Image.new('RGB',(256,256),(139,174,181))
    d = ImageDraw.Draw(huiyan)
    for x in (0,128):
        d.rectangle((x+12,0,x+125,255),fill=(143,177,184))
        d.rectangle((x+17,5,x+120,120),fill=(153,185,189))
        d.rectangle((x+17,132,x+120,250),fill=(136,171,179))
        d.line((x+7,0,x+7,255),fill=(237,240,232),width=18)
        d.line((x+16,0,x+16,255),fill=(104,128,130),width=3)
    for y in (0,128,255):
        d.line((0,y,255,y),fill=(88,117,124),width=3)
    huiyan.save(ASSETS/'facade-huiyan.ppm')

    # 汇星楼的浅色面板与细密横向窗带。窗框、玻璃反光和个别空调由程序绘制。
    rng = random.Random(230847103)
    hx = Image.new('RGB',(512,256),(214,213,201))
    d = ImageDraw.Draw(hx)
    for y in (12,244):
        d.line((0,y,511,y),fill=(193,193,184),width=2)
    for x in range(0,512,64):
        d.line((x,13,x,87),fill=(197,197,187),width=2)
        d.line((x,170,x,243),fill=(197,197,187),width=2)
        d.rectangle((x+2,87,x+62,170),fill=(175,184,177))
        for k in range(2):
            left=x+5+k*28
            pane=rng.choice(((104,125,123),(118,139,136),(140,154,147),(91,111,112),(177,177,159)))
            d.rectangle((left,93,left+24,164),fill=pane,outline=(190,197,186),width=2)
            d.line((left+1,128,left+23,128),fill=(165,174,164),width=2)
            if rng.random()<.18:
                d.rectangle((left+5,144,left+17,157),fill=(216,218,205),outline=(125,136,130))
    hx.save(ASSETS/'facade-huixing.ppm')

    # 中央交通筒内立面的细网格；外露塔身另用真正的透明面和立体金属框架。
    core = Image.new('RGB',(256,256),(124,151,153))
    d = ImageDraw.Draw(core)
    d.rectangle((0,0,255,255),outline=(48,63,62),width=7)
    d.line((128,0,128,255),fill=(66,82,79),width=5)
    d.line((0,128,255,128),fill=(66,82,79),width=5)
    d.line((10,18,75,90),fill=(166,184,180),width=3)
    core.save(ASSETS/'facade-huixing-core.ppm')

    # 汇元楼入口白色石面及暗铜色校名，以字体重新绘制，不使用实景照片作纹理。
    entrance = Image.new('RGB',(512,256),(226,225,217))
    d = ImageDraw.Draw(entrance)
    for x in range(0,513,128):
        d.line((x,0,x,255),fill=(203,203,196),width=2)
    for y in (0,128,255):
        d.line((0,y,511,y),fill=(203,203,196),width=2)
    title=ImageFont.truetype('C:/Windows/Fonts/STXINGKA.TTF',76)
    english=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',20)
    d.text((99,53),'深圳大学',font=title,fill=(113,55,29))
    d.text((126,153),'SHENZHEN UNIVERSITY',font=english,fill=(113,55,29))
    entrance.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'huiyuan-sign.ppm')

    # 门廊内石壁以黑色脚印与浅色校训字样示意，图案非原雕刻复制。
    motto=Image.new('RGB',(512,256),(160,154,140))
    d=ImageDraw.Draw(motto)
    for x in range(0,513,85):
        d.line((x,0,x,255),fill=(126,121,111),width=3)
    for y in (0,85,170,255):
        d.line((0,y,511,y),fill=(126,121,111),width=3)
    mfont=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',47)
    for x,ch in zip((30,109,188,267,346,425),'自立自律自强'):
        d.text((x,20),ch,font=mfont,fill=(132,126,114))
    d.ellipse((154,147,205,221),fill=(32,33,30))
    for j in range(4):
        d.ellipse((159+j*11,123-j*2,169+j*11,139-j*2),fill=(32,33,30))
    d.text((225,115),'实地',font=ImageFont.truetype('C:/Windows/Fonts/STXINGKA.TTF',82),fill=(32,33,30))
    motto.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'huiyuan-motto.ppm')

    podium = Image.new('RGB',(256,256),(182,185,177))
    d = ImageDraw.Draw(podium)
    for y in range(10,160,12):
        d.line((0,y,255,y),fill=(109,117,115),width=3)
    d.rectangle((0,174,255,244),fill=(102,122,121))
    for x in range(0,256,32):
        d.line((x,174,x,244),fill=(196,204,196),width=3)
    podium.save(ASSETS/'facade-huixing-podium.ppm')

    # 航拍可见裙房屋顶的长条深色分区和浅色边框；用途及精确数量未核实。
    roof = Image.new('RGB',(256,256),(139,140,134))
    d = ImageDraw.Draw(roof)
    d.rectangle((10,12,246,244),fill=(99,101,98),outline=(199,199,189),width=5)
    d.line((15,84,241,84),fill=(136,138,131),width=2)
    d.line((15,169,241,169),fill=(136,138,131),width=2)
    roof.save(ASSETS/'huixing-podium-roof.ppm')

    # 汇文楼多栋楼的立面是密集浅色框架与内缩窗洞，并非连续深色横窗。
    huiwen = Image.new('RGB',(256,256),(229,230,222))
    d = ImageDraw.Draw(huiwen)
    d.rectangle((0,17,255,38),fill=(241,241,234))
    d.rectangle((0,217,255,241),fill=(242,241,233))
    for x in (10,92,174):
        d.rectangle((x,43,x+70,210),fill=(159,171,166))
        d.rectangle((x+6,49,x+64,204),fill=(88,109,109))
        d.rectangle((x+10,55,x+60,93),fill=(130,147,143))
        d.line((x+35,49,x+35,204),fill=(205,212,204),width=4)
        for y in (99,153):
            d.line((x+6,y,x+64,y),fill=(211,215,206),width=5)
    huiwen.save(ASSETS/'facade-huiwen.ppm')
    # 入口红墙上的狭长窗洞参照照片概括，窗位和数量不是施工图。
    huiwen_red = Image.new('RGB',(256,256),(166,53,46))
    d = ImageDraw.Draw(huiwen_red)
    d.rectangle((100,45,151,210),fill=(223,214,199))
    d.rectangle((107,52,144,203),fill=(88,101,99))
    d.line((125,52,125,203),fill=(170,179,170),width=3)
    huiwen_red.save(ASSETS/'facade-huiwen-red.ppm')

    strip = Image.new('RGB',(256,256),(222,224,219))
    d = ImageDraw.Draw(strip)
    d.rectangle((0,72,255,163),fill=(71,91,92))
    d.line((0,77,255,77),fill=(145,161,157),width=5)
    for x in range(0,256,64):
        d.rectangle((x,72,x+5,163),fill=(195,203,198))
    strip.save(ASSETS / 'facade-strip.ppm')

    holes = Image.new('RGB',(256,256),(225,227,223))
    d = ImageDraw.Draw(holes)
    d.rectangle((91,68,164,172),fill=(169,182,178))
    d.rectangle((97,74,158,166),fill=(61,82,85))
    d.line((127,75,127,165),fill=(123,139,137),width=3)
    holes.save(ASSETS / 'facade-perforated.ppm')

    # 致艺楼外墙散布小型穿孔设备格栅；不把它们误画成普通窗户。
    vent=Image.new('RGB',(128,128),(231,232,226))
    d=ImageDraw.Draw(vent)
    for y in range(10,119,11):
        for x in range(10,119,11):
            d.ellipse((x,y,x+3,y+3),fill=(115,122,118))
    vent.save(ASSETS/'facade-zhiyi-vent.ppm')

    # 裙房呈浅色厚墙及窄窗，纹理仅按照片概括窗形，不代表具体窗位。
    podium=Image.new('RGB',(256,256),(228,229,223))
    d=ImageDraw.Draw(podium)
    for x in (50,168):
        d.rectangle((x,40,x+30,216),fill=(169,181,176))
        d.rectangle((x+5,44,x+25,212),fill=(79,105,106))
        d.line((x+15,45,x+15,211),fill=(166,180,174),width=2)
    d.line((0,15,255,15),fill=(211,214,207),width=3)
    d.line((0,238,255,238),fill=(211,214,207),width=3)
    podium.save(ASSETS/'facade-zhiyi-podium.ppm')

    teng = Image.new('RGB',(256,256),(207,200,188))
    d = ImageDraw.Draw(teng)
    d.rectangle((13,92,243,145),fill=(103,121,120))
    d.line((0,90,255,90),fill=(230,225,215),width=5)
    for x in (64,128,192):
        d.line((x,92,x,145),fill=(190,190,181),width=4)
    teng.save(ASSETS / 'facade-teng.ppm')

    atrium = Image.new('RGB',(256,256),(79,111,111))
    d = ImageDraw.Draw(atrium)
    d.rectangle((0,0,255,255),outline=(133,154,151),width=5)
    d.line((128,0,128,255),fill=(134,153,149),width=3)
    d.line((0,120,255,120),fill=(105,133,130),width=3)
    atrium.save(ASSETS / 'facade-atrium.ppm')

    # 档案馆的印章侧墙根据“岁月如金”四字重新绘制，字形是示意而非原建筑篆刻复制。
    mask = Image.new('L',(512,512),0)
    d = ImageDraw.Draw(mask)
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',220)
    for ch,x,y in (('岁',17,-30),('月',273,-30),('如',17,226),('金',273,226)):
        d.text((x,y),ch,font=font,fill=255)
    # 用细凹槽轮廓代替深色广告牌；原建筑篆体图案没有被复制。
    outline = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(9)),mask.filter(ImageFilter.MinFilter(9)))
    seal = Image.new('RGB',(512,512),(224,219,197))
    seal.paste((95,72,61),(0,0,512,512),outline)
    ImageDraw.Draw(seal).rectangle((5,5,506,506),outline=(106,84,72),width=7)
    seal.save(ASSETS / 'archive-seal.ppm')

    # 汇昀楼一翼是红色实墙与间隔窗，而不是整面玻璃幕墙。
    img = Image.new('RGB',(256,256),(173,51,62))
    d = ImageDraw.Draw(img)
    for x in (18,138):
        d.rectangle((x,64,x+82,172),fill=(224,225,216))
        d.rectangle((x+5,69,x+77,167),fill=(94,132,143))
        d.line((x+41,69,x+41,167),fill=(211,219,213),width=4)
        d.line((x+5,117,x+77,117),fill=(211,219,213),width=4)
        d.rectangle((x+48,183,x+78,200),fill=(202,204,194),outline=(134,135,128),width=2)
    img.save(ASSETS/'facade-activity-red.ppm')


def sports_textures():
    # 用程序制作座椅纹理，现场照片只作形态参考。
    seat=Image.new('RGB',(128,128),(155,155,155))
    d=ImageDraw.Draw(seat)
    d.rounded_rectangle((8,6,119,90),radius=12,fill=(235,235,235),outline=(105,105,105),width=4)
    d.rectangle((8,91,119,113),fill=(195,195,195))
    d.line((11,92,116,92),fill=(118,118,118),width=4)
    seat.save(ASSETS/'stadium-seat.ppm')

    # 依据航拍绘制红蓝篮球场与网球场的示意线条；不使用实景照片作纹理。
    basketball = Image.new('RGB', (512,320), (64,108,139))
    d = ImageDraw.Draw(basketball)
    d.rectangle((34,28,478,292), fill=(167,78,68), outline=(237,235,222), width=4)
    d.line((256,28,256,292), fill=(237,235,222), width=4)
    d.ellipse((220,124,292,196), outline=(237,235,222), width=4)
    for x in (34, 408):
        d.rectangle((x,107,x+70,213), fill=(64,108,139), outline=(237,235,222), width=4)
    # 航拍能辨认出三分线弧；线条按球场长轴绘制，具体尺度仍为示意。
    d.arc((-54,50,164,270),start=-72,end=72,fill=(237,235,222),width=4)
    d.arc((348,50,566,270),start=108,end=252,fill=(237,235,222),width=4)
    basketball.save(ASSETS/'basketball-court.ppm')

    tennis = Image.new('RGB', (512,320), (84,125,107))
    d = ImageDraw.Draw(tennis)
    d.rectangle((20,28,492,292), fill=(84,125,141))
    d.rectangle((42,44,470,276), outline=(237,237,224), width=4)
    d.line((42,70,470,70), fill=(237,237,224), width=3)
    d.line((42,250,470,250), fill=(237,237,224), width=3)
    d.line((256,44,256,276), fill=(237,237,224), width=4)
    d.line((152,70,152,250), fill=(237,237,224), width=3)
    d.line((360,70,360,250), fill=(237,237,224), width=3)
    d.line((152,160,360,160), fill=(237,237,224), width=3)
    tennis.save(ASSETS/'tennis-court.ppm')

    # 近景照片中的深色菱形围网；透明背景使用 alpha 裁剪，避免画成不透明墙。
    fence = Image.new('RGBA',(64,64),(48,65,61,0))
    d = ImageDraw.Draw(fence)
    for x in range(-128,129,32):
        d.line((x,0,x+64,64),fill=(74,85,77,230),width=2)
        d.line((x,64,x+64,0),fill=(74,85,77,230),width=2)
    with (ASSETS/'sports-fence.rgba').open('wb') as f:
        f.write(struct.pack('<II',fence.width,fence.height))
        f.write(fence.tobytes())


def stone_plaza_textures():
    # 参考 2026 年校园实拍重画橙白放射铺装；不把实景照片作为纹理使用。
    size = 1024
    img = Image.new('RGB',(size,size),(223,207,145))
    d = ImageDraw.Draw(img)
    center, radius = size//2, int(size*23/72)
    bounds = (center-radius,center-radius,center+radius,center+radius)
    for i in range(8):
        d.pieslice(bounds,start=i*45-22.5,end=(i+1)*45-22.5,
                   fill=(214,131,53) if i%2 == 0 else (231,225,189))
    for factor in (.47,.75,1):
        r = int(radius*factor)
        d.ellipse((center-r,center-r,center+r,center+r),outline=(242,237,212),width=4)
    r = int(radius*.20)
    d.ellipse((center-r,center-r,center+r,center+r),fill=(232,226,190),outline=(244,239,219),width=4)
    img.save(ASSETS/'stone-plaza.ppm')

    # 隔壁海滨泳池照片可见浅蓝水色及深色泳道线；泳道精确间距按图示意。
    img = Image.new('RGB',(1024,512),(91,181,199))
    d = ImageDraw.Draw(img)
    for y in range(16,512,8):
        color=(92+(y//8)%4,178+(y//8)%5,196+(y//8)%5)
        d.line((0,y,1023,y),fill=color,width=2)
    for lane in range(1,9):
        y = int(lane*512/9)
        d.line((22,y,1002,y),fill=(30,81,99),width=5)
        d.line((22,y-8,22,y+8),fill=(30,81,99),width=4)
        d.line((1002,y-8,1002,y+8),fill=(30,81,99),width=4)
    img.save(ASSETS/'south-pool.ppm')

    # 石头坞旧楼的石材用不同灰褐色不规则石块生成，可平铺，非实拍贴图。
    img = Image.new('RGB',(512,512),(102,99,91))
    d = ImageDraw.Draw(img)
    rng = random.Random(231572974)
    for row in range(8):
        y=row*64
        offset = -42 if row%2 else 0
        for x in range(offset,512,85):
            c=rng.randint(121,162)
            points=[(x+3,y+5),(x+75,y+4),(x+82,y+17),(x+78,y+54),
                    (x+55,y+60),(x+9,y+57),(x+2,y+42)]
            d.polygon(points,fill=(c,c-4,c-11),outline=(83,81,76),width=3)
    img.save(ASSETS/'facade-stone-fortress.ppm')


def yuanping_textures():
    """参照场馆正面运动浮雕重新绘制图案，不使用现场照片作为纹理。"""
    img = Image.new('RGB',(1600,192),(211,202,181))
    d = ImageDraw.Draw(img)
    # 每幅图使用不同动作与球具，表示人物浮雕；不声称逐幅复制原作。
    poses = [
        [(48,102),(72,69),(100,78),(120,52),(102,79),(83,115),(112,151),(82,126),(50,149)],
        [(42,128),(72,94),(91,62),(112,79),(127,57),(111,80),(97,104),(118,142),(85,117),(60,153)],
        [(38,142),(68,112),(92,103),(115,116),(133,91),(116,117),(97,123),(77,151),(71,118),(48,153)],
        [(42,82),(69,71),(88,92),(106,67),(126,74),(108,68),(90,106),(111,143),(84,123),(58,146)],
        [(38,138),(66,109),(87,82),(106,91),(122,70),(105,94),(96,113),(120,143),(85,123),(62,151)],
        [(43,105),(73,88),(92,60),(108,79),(126,66),(107,82),(94,105),(117,137),(85,121),(53,144)],
        [(38,126),(66,105),(89,76),(108,93),(125,79),(107,95),(95,112),(112,148),(82,127),(54,151)],
        [(38,76),(64,92),(89,88),(111,66),(125,82),(109,70),(91,105),(113,138),(82,122),(52,140)],
        [(41,129),(69,109),(92,77),(111,91),(128,73),(111,95),(99,116),(122,145),(86,126),(59,151)],
        [(42,119),(68,96),(91,79),(111,99),(126,82),(108,100),(92,113),(112,143),(80,124),(56,147)],
    ]
    for i,points in enumerate(poses):
        x=i*160
        d.rectangle((x+3,4,x+156,187),fill=(199,186,162),outline=(176,161,140),width=3)
        for gy in range(28,187,31):
            d.line((x+4,gy,x+155,gy),fill=(192,178,157),width=1)
        points=[(x+px,py) for px,py in points]
        d.line([(px+2,py+3) for px,py in points],fill=(116,100,82),width=13,joint='curve')
        d.line(points,fill=(143,128,106),width=10,joint='curve')
        hx,hy=points[2]
        d.ellipse((hx-9,hy-18,hx+9,hy),fill=(143,128,106))
        if i in (1,4,8):
            d.ellipse((x+125,35,x+141,51),fill=(143,128,106))
        if i == 9:
            d.ellipse((x+29,115,x+66,152),outline=(143,128,106),width=6)
            d.ellipse((x+104,115,x+141,152),outline=(143,128,106),width=6)
    img.save(ASSETS/'yuanping-relief.ppm')
    seal = Image.new('RGB',(256,256),(225,224,218))
    d = ImageDraw.Draw(seal)
    d.ellipse((8,8,248,248),fill=(100,44,75),outline=(239,236,224),width=7)
    d.ellipse((39,39,217,217),outline=(239,236,224),width=5)
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',52)
    d.text((128,99),'深大',font=font,fill=(246,240,228),anchor='mm')
    d.text((128,166),'1983',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',29),
           fill=(246,240,228),anchor='mm')
    seal.save(ASSETS/'yuanping-seal.ppm')
    name = Image.new('RGB',(512,128),(225,225,219))
    d = ImageDraw.Draw(name)
    d.text((256,43),'元平体育馆',font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',52),
           fill=(26,31,31),anchor='mm')
    d.text((256,102),'YUANPING GYMNASIUM',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',23),
           fill=(34,39,39),anchor='mm')
    name.save(ASSETS/'yuanping-name.ppm')

    # 入口楼名使用本机字体重新绘制，仅近似照片中的黑色中英文字样。
    label=Image.new('RGB',(512,128),(255,255,255))
    d=ImageDraw.Draw(label)
    d.text((256,47),'致理楼 A',font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',54),
           fill=(27,31,31),anchor='mm')
    d.text((256,100),'ZHI LI BUILDING',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',19),
           fill=(27,31,31),anchor='mm')
    label.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'zhili-a-sign.ppm')


def golf_texture():
    # 2016 年场内可见黄色距离牌；数字、位置和数量按照片概括。
    img=Image.new('RGB',(1024,256),(245,213,48))
    d=ImageDraw.Draw(img)
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',150)
    for i,distance in enumerate((50,100,150,200)):
        x=i*256
        d.rectangle((x+3,3,x+252,252),outline=(48,48,31),width=5)
        d.text((x+128,128),str(distance),font=font,fill=(30,32,27),anchor='mm')
    img.save(ASSETS/'golf-distance.ppm')


def sundial_textures():
    # 依据学校公开照片重画日晷盘面与石砌基座，不直接使用照片纹理。
    dial = Image.new('RGB',(512,512),(232,229,215))
    d = ImageDraw.Draw(dial)
    for radius in (166,206,248):
        d.ellipse((256-radius,256-radius,256+radius,256+radius),outline=(174,169,155),width=3)
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',30)
    for index,ch in enumerate('子丑寅卯辰巳午未申酉戌亥'):
        angle = 2*math.pi*index/12 - math.pi/2
        x,y = 256+188*math.cos(angle),256+188*math.sin(angle)
        d.text((x,y),ch,font=font,fill=(51,50,46),anchor='mm')
        d.line((256+214*math.cos(angle),256+214*math.sin(angle),
                256+239*math.cos(angle),256+239*math.sin(angle)),fill=(68,66,62),width=5)
    dial.save(ASSETS/'sundial-face.ppm')
    # 校方近景照片中的日晷底座为深灰色砌石，纹理为自行绘制的近似。
    stone = Image.new('RGB',(256,256),(127,130,124))
    d = ImageDraw.Draw(stone)
    rng=random.Random(11833634683)
    for row,y in enumerate(range(0,256,32)):
        for x in range(-32+(row%2)*32,256,64):
            tone=rng.randint(-9,9)
            d.rectangle((x+2,y+2,x+62,y+30),fill=(127+tone,130+tone,124+tone))
            d.line((x,y,x,y+32),fill=(91,94,89),width=2)
        d.line((0,y,255,y),fill=(91,94,89),width=2)
    stone.save(ASSETS/'sundial-stone.ppm')


def qiaoyuan_textures():
    # 依据乔木、乔林航拍概括灰色阳台与黄色竖向色带；每个重复单元约四间。
    image = Image.new('RGB',(512,256),(151,159,155))
    d = ImageDraw.Draw(image)
    d.rectangle((0,0,22,255),fill=(210,188,98))
    d.rectangle((23,0,511,18),fill=(200,205,198))
    d.rectangle((23,235,511,255),fill=(194,199,193))
    for x in range(28,512,120):
        d.rectangle((x,30,min(511,x+104),202),fill=(69,86,88),outline=(205,210,202),width=5)
        d.line((x+52,34,x+52,199),fill=(154,165,160),width=4)
        for y in (145,162,179):
            d.line((x+4,y,min(511,x+101),y),fill=(181,190,184),width=3)
    image.save(ASSETS/'facade-qiaomu.ppm')

    image = Image.new('RGB',(512,256),(224,225,219))
    d = ImageDraw.Draw(image)
    d.rectangle((0,30,511,210),fill=(84,98,98))
    for x in range(0,512,86):
        d.rectangle((x,28,x+12,216),fill=(220,222,216))
        d.line((x+14,150,x+82,150),fill=(178,186,181),width=7)
        d.line((x+14,183,x+82,183),fill=(190,196,190),width=5)
    d.rectangle((0,0,511,24),fill=(232,232,226))
    d.rectangle((0,221,511,255),fill=(222,224,219))
    image.save(ASSETS/'facade-qiaosen.ppm')

    # 聚翰、紫薇和红豆新斋实景以浅墙、深色阳台与密集水平护栏为主。
    image = Image.new('RGB',(512,256),(224,221,210))
    d = ImageDraw.Draw(image)
    d.rectangle((0,0,511,22),fill=(237,235,224))
    d.rectangle((0,226,511,255),fill=(230,228,217))
    for x in range(4,512,128):
        d.rectangle((x,28,x+110,220),fill=(87,96,91),outline=(216,217,207),width=7)
        d.line((x+57,33,x+57,156),fill=(161,170,160),width=4)
        d.rectangle((x+6,147,x+104,217),fill=(145,151,141))
        for y in (159,172,185,198):
            d.line((x+8,y,x+102,y),fill=(228,228,215),width=3)
        d.rectangle((x+77,42,x+99,60),fill=(184,182,163))
    image.save(ASSETS/'facade-zhai.ppm')


def gate_textures():
    # 立德门实际有细密横向分缝及竖直校名牌；字体重新绘制，尺寸为示意。
    stripes = Image.new('RGB',(256,256),(183,186,179))
    d = ImageDraw.Draw(stripes)
    for y in range(0,256,20):
        d.line((0,y,255,y),fill=(120,126,122),width=3)
        d.line((0,y+4,255,y+4),fill=(208,210,202),width=2)
    stripes.save(ASSETS/'facade-gate.ppm')
    sign = Image.new('RGB',(256,512),(231,232,221))
    d = ImageDraw.Draw(sign)
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',76)
    for i,ch in enumerate('深圳大学'):
        d.text((128,72+i*100),ch,font=font,fill=(32,36,34),anchor='mm')
    sign.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'gate-sign.ppm')
    # 照片中的蓝色门牌为南海大道 3688 号；数字由本机字体重绘。
    address=Image.new('RGB',(256,128),(26,93,151))
    d=ImageDraw.Draw(address)
    d.rectangle((4,4,251,123),outline=(229,236,237),width=4)
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',72)
    d.text((128,63),'3688',font=font,fill=(243,246,240),anchor='mm')
    address.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'gate-address.ppm')


def zhiyuan_sign_textures():
    # 参考校方 2024 年照片重画校名装置；不使用合影或照片贴图。
    words=Image.new('RGBA',(1024,256),(255,255,255,0))
    draw=ImageDraw.Draw(words)
    font=ImageFont.truetype('C:/Windows/Fonts/STXINGKA.TTF',222)
    draw.text((512,128),'深圳大学',font=font,fill=(246,246,240,255),anchor='mm')
    words=words.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    with (ASSETS/'zhiyuan-sign-words.rgba').open('wb') as f:
        f.write(struct.pack('<II',*words.size));f.write(words.tobytes())
    base=Image.new('RGB',(1024,128),(104,47,70));draw=ImageDraw.Draw(base)
    font=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',44)
    draw.text((512,64),'S H E N Z H E N   U N I V E R S I T Y',font=font,
              fill=(238,229,224),anchor='mm')
    base.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'zhiyuan-sign-base.ppm')


def canopy_texture():
    # 用 Poly Haven 的 CC0 叶片图生成树冠材质；不是校园树木实拍定位。
    color = Image.open(DATA/'foliage'/'tree-small-leaves-diff.jpg').convert('RGBA')
    alpha = Image.open(DATA/'foliage'/'tree-small-leaves-alpha.png').convert('L')
    color.putalpha(alpha)
    # 原始贴图将相邻枝条与遮罩放在一起，单独裁掉枝条以免冠层出现黑线。
    leaf = color.crop((235,275,375,380))
    mask = Image.new('L',leaf.size,0)
    ImageDraw.Draw(mask).polygon([(17,29),(20,23),(27,19),(39,16),(52,15),(69,15),
        (84,17),(98,22),(108,29),(113,36),(116,44),(118,63),(111,70),(104,75),
        (90,82),(75,86),(59,87),(47,83),(36,77),(26,69),(20,59),(16,48),(16,37)],fill=255)
    leaf.putalpha(ImageChops.multiply(leaf.getchannel('A'),mask))
    leaf = ImageEnhance.Brightness(leaf).enhance(1.30)
    atlas = Image.new('RGBA',(512,512),(73,112,63,255))
    rng = random.Random(20242702)
    for _ in range(210):
        angle = rng.uniform(0,2*math.pi)
        radius = math.sqrt(rng.random())
        x = 256+radius*220*math.cos(angle)
        y = 256+radius*195*math.sin(angle)
        size = rng.randint(34,72)
        part = leaf.resize((size,int(size*.74)),Image.Resampling.LANCZOS)
        part = part.rotate(rng.uniform(0,360),resample=Image.Resampling.BICUBIC,expand=True)
        atlas.alpha_composite(part,(int(x-part.width/2),int(y-part.height/2)))
    with (ASSETS/'campus-canopy.rgba').open('wb') as f:
        f.write(struct.pack('<II',atlas.width,atlas.height))
        f.write(atlas.tobytes())

    # 同一 CC0 叶片拼成透明的枝叶簇，供树冠外缘使用；空白像素保留绿色以减轻 mipmap 黑边。
    cluster = Image.new('RGBA',(512,512),(0,0,0,0))
    rng = random.Random(20242703)
    for _ in range(240):
        angle = rng.uniform(0,2*math.pi)
        radius = math.sqrt(rng.random())
        x = 256+radius*200*math.cos(angle)
        y = 256+radius*160*math.sin(angle)
        size = rng.randint(35,70)
        part = leaf.resize((size,int(size*.74)),Image.Resampling.LANCZOS)
        part = part.rotate(rng.uniform(0,360),resample=Image.Resampling.BICUBIC,expand=True)
        cluster.alpha_composite(part,(int(x-part.width/2),int(y-part.height/2)))
    pixels = np.asarray(cluster).copy()
    pixels[pixels[:,:,3] == 0, :3] = (67,100,57)
    cluster = Image.fromarray(pixels,'RGBA')
    with (ASSETS/'campus-leaf-cluster.rgba').open('wb') as f:
        f.write(struct.pack('<II',cluster.width,cluster.height))
        f.write(cluster.tobytes())


def huizhi_textures():
    # 汇智楼照片显示细密的浅色竖框与深浅不一的玻璃面板；不使用实拍作为纹理。
    facade=Image.new('RGB',(256,512),(126,150,148))
    d=ImageDraw.Draw(facade)
    palette=((123,145,142),(137,155,151),(114,139,138),(145,161,155),
             (128,148,145),(111,134,132),(140,157,153),(120,142,140),(133,151,147),(117,137,135))
    for floor in range(5):
        y0=round(floor*512/5); y1=round((floor+1)*512/5)
        for pane in range(2):
            x0=pane*128; x1=(pane+1)*128
            d.rectangle((x0+3,y0+3,x1-3,y1-3),fill=palette[floor*2+pane])
            d.line((x0+8,y0+8,x1-8,y0+8),fill=(162,177,169),width=2)
            if (floor,pane) in ((1,0),(3,1)):
                d.rectangle((x0+14,y0+12,x1-14,y0+17),fill=(77,102,101))
        d.line((0,y0,255,y0),fill=(199,207,199),width=4)
    d.line((127,0,127,511),fill=(192,201,193),width=3)
    facade.save(ASSETS/'facade-huizhi.ppm')

    # 按学校入口照片重绘文字，仅表达字样，不复制原字形。
    sign=Image.new('RGB',(512,128),(230,232,225))
    d=ImageDraw.Draw(sign)
    main_font=ImageFont.truetype('C:/Windows/Fonts/simkai.ttf',64)
    sub_font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    d.text((40,20),'图书馆',font=main_font,fill=(72,78,76))
    d.text((310,53),'南馆',font=sub_font,fill=(72,78,76))
    sign.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'huizhi-entrance.ppm')


def zhizhen_textures():
    # 演播厅实拍可见深色体量下的白色 STUDIO 字样；文字自行绘制，不复制现场照片。
    sign = Image.new('RGB',(1024,256),(27,30,31))
    draw = ImageDraw.Draw(sign)
    font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',205)
    bounds = draw.textbbox((0,0),'STUDIO',font=font,stroke_width=0)
    draw.text(((1024-(bounds[2]-bounds[0]))/2-bounds[0],
               (256-(bounds[3]-bounds[1]))/2-bounds[1]),
              'STUDIO',font=font,fill=(231,232,225))
    sign.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'studio-sign.ppm')
    # 致理楼低层部分短侧为浅墙和横向狭窗，按照片重绘一个可重复单元。
    side = Image.new('RGB',(256,256),(226,228,222))
    draw = ImageDraw.Draw(side)
    draw.rectangle((52,104,196,127),fill=(49,60,61))
    draw.line((52,104,196,104),fill=(135,142,138),width=3)
    side.save(ASSETS/'facade-zhili-side.ppm')


def zhigong_textures():
    # 百叶和院名按照片重新绘制，照片本身不作为模型纹理。
    vent=Image.new('RGB',(128,128),(49,57,57))
    draw=ImageDraw.Draw(vent)
    for y in range(3,128,10):
        draw.line((0,y,127,y),fill=(139,150,145),width=2)
    vent.save(ASSETS/'zhigong-louver.ppm')
    sign=Image.new('RGB',(1024,160),(235,235,228))
    draw=ImageDraw.Draw(sign)
    draw.ellipse((18,25,126,133),outline=(125,46,48),width=5)
    draw.ellipse((34,41,110,117),outline=(125,46,48),width=2)
    draw.text((146,15),'深圳大学 土木与交通工程学院',
              font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',47),fill=(76,72,65))
    draw.text((150,99),'COLLEGE OF CIVIL AND TRANSPORTATION ENGINEERING',
              font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22),fill=(76,72,65))
    sign.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'zhigong-sign.ppm')


def canteen_textures():
    # 2019 年照片中的楼层名称用本机字体重新绘制；不把照片作为纹理。
    sign=Image.new('RGB',(512,256),(183,39,36))
    draw=ImageDraw.Draw(sign)
    font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',59)
    for y,label in ((64,'南区一楼餐厅'),(192,'南区二楼餐厅')):
        draw.text((256,y),label,font=font,fill=(247,242,230),anchor='mm')
    sign.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'canteen-sign.ppm')


def dujuan_boardwalk_texture():
    # 杜鹃山曾修建 1.3 米塑木栈道；纹理为自行绘制的近似木板。
    image=Image.new('RGB',(256,256),(148,132,109))
    draw=ImageDraw.Draw(image)
    for x in range(0,256,32):
        draw.line((x,0,x,255),fill=(104,94,79),width=2)
        draw.line((x+3,0,x+3,255),fill=(164,147,122),width=1)
    for y in (47,109,171,232):
        draw.line((0,y,255,y),fill=(141,126,106),width=1)
    image.save(ASSETS/'dujuan-boardwalk.ppm')


def campus_asphalt_texture():
    # 自行生成细颗粒沥青，重复使用时不把外部照片当作纹理。
    rng=np.random.default_rng(20261103)
    noise=rng.normal(0,4,(256,256,1))
    color=np.asarray((108,112,108),dtype=float).reshape(1,1,3)
    pixels=np.clip(color+noise,0,255).astype(np.uint8)
    Image.fromarray(pixels,'RGB').save(ASSETS/'campus-asphalt.ppm')


def qiaoyuan_garden_textures():
    # 园路、红砖和立体框架文字均自行绘制；照片只用于观察，不作为纹理。
    rng=np.random.default_rng(20221214)
    noise=rng.normal(0,3.2,(256,256,1))
    sand=np.clip(np.asarray((222,177,125)).reshape(1,1,3)+noise,0,255).astype(np.uint8)
    Image.fromarray(sand,'RGB').save(ASSETS/'qiaoyuan-garden-walk.ppm')
    brick=Image.new('RGB',(256,256),(161,99,78))
    draw=ImageDraw.Draw(brick)
    for y in range(0,256,32):
        draw.line((0,y,255,y),fill=(112,77,65),width=2)
        offset=32 if (y//32)%2 else 0
        for x in range(offset,256,64):
            draw.line((x,y,x,y+31),fill=(112,77,65),width=2)
    brick.save(ASSETS/'qiaoyuan-brick.ppm')
    words=Image.new('RGBA',(1024,160),(220,241,231,0))
    draw=ImageDraw.Draw(words)
    font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',122)
    draw.text((512,76),'森林校巴花园',font=font,fill=(213,237,224,255),anchor='mm')
    words=words.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    with (ASSETS/'qiaoyuan-garden-words.rgba').open('wb') as f:
        f.write(struct.pack('<II',*words.size))
        f.write(words.tobytes())


def xiyuan_shop_textures():
    """程序重绘近似店招与西苑外墙，不把旧照片当作当前店面贴图。"""
    font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',71)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    entries=[('MANNER COFFEE','#70494a','#fff9ec'),('优剪','#1f2525','#ffffff'),
             ('果果切','#124d3b','#ffffff'),('7-ELEVEN','#f8f7ed','#007548'),
             ('茶饮 · 小食','#c79869','#ffffff'),('图文打印','#df8d31','#ffffff'),
             ('修车 · 配钥匙','#f0e5cd','#b02d27'),('顺丰速运','#f5f3ed','#d8322e'),
             ('LAWSON 罗森','#1749a6','#ffffff'),('Hair Code','#9970b0','#ffffff'),
             ('博士眼镜','#161b20','#fff7ed'),('深圳大学 品牌文化店','#8b2044','#ffffff')]
    atlas=Image.new('RGB',(1024,len(entries)*128))
    draw=ImageDraw.Draw(atlas)
    for i,(label,bg,fg) in enumerate(entries):
        y=i*128
        draw.rectangle((0,y,1023,y+127),fill=bg)
        if i==3:
            # 校内 7-11 的灯箱为白底、中间数字与两侧橙绿红色横带。
            draw.rectangle((0,y,85,y+127),fill='#262b28')
            draw.rectangle((938,y,1023,y+127),fill='#262b28')
            for k,color in enumerate(('#e97923','#138344','#c4272d')):
                for x0,x1 in ((95,420),(604,928)):
                    draw.rounded_rectangle((x0,y+17+k*30,x1,y+38+k*30),radius=4,fill=color)
            large=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',91)
            tiny=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',19)
            draw.text((512,y+53),'7',font=large,fill='#c92d2d',anchor='mm')
            draw.text((512,y+97),'ELEVEN',font=tiny,fill='#0c7640',anchor='mm')
            continue
        if i==8:
            draw.line((0,y+7,1023,y+7),fill='#c5e9ff',width=3)
            draw.line((0,y+119,1023,y+119),fill='#c5e9ff',width=3)
        draw.text((512,y+70),label,font=font,fill=fg,anchor='mm',stroke_width=0)
        if i==7:
            draw.text((920,y+100),'EXPRESS',font=small,fill=fg,anchor='mm')
    atlas.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'xiyuan-shop-signs.ppm')

    # 宿舍长翼为浅色外墙、暖橙色竖向构件及连续外廊；窗格比例按照片概括。
    facade=Image.new('RGB',(256,256),(224,221,207))
    d=ImageDraw.Draw(facade)
    d.rectangle((0,0,255,20),fill=(242,240,230))
    d.rectangle((0,220,255,255),fill=(235,232,218))
    d.rectangle((0,21,21,219),fill=(214,151,83))
    d.rectangle((234,21,255,219),fill=(214,151,83))
    for x in (32,137):
        d.rectangle((x,42,x+87,177),fill=(134,148,145))
        d.rectangle((x+6,48,x+81,171),fill=(103,124,121))
        d.line((x+44,48,x+44,171),fill=(218,221,210),width=5)
        d.line((x+6,109,x+81,109),fill=(218,221,210),width=4)
    for y in (190,205):
        d.line((22,y,233,y),fill=(150,157,148),width=3)
    facade.save(ASSETS/'facade-xiyuan.ppm')
    # Manner 上方是裸露的暖橙色墙面；用噪声概括旧墙风化，不使用实拍贴图。
    rng_np=np.random.default_rng(20260806)
    grain=rng_np.normal(0,5,(256,256,1))
    ochre=np.clip(np.asarray((184,126,76)).reshape((1,1,3))+grain,0,255).astype(np.uint8)
    Image.fromarray(ochre,'RGB').save(ASSETS/'xiyuan-manner-wall.ppm')
    # 每间店以简单货架、柜台或工具表达用途，细节为概括而非现场室内复制。
    window=Image.new('RGB',(1024,12*256))
    d=ImageDraw.Draw(window)
    rng=random.Random(71)
    for i in range(12):
        y=i*256
        background=('#cbbda5','#dadbd0','#c2d4b3','#d8e1d5','#d4c2a5','#d6dfdb',
                    '#8a8c80','#deded7','#d9e5e2','#d2c4de','#49a6da','#d9d2ca')[i]
        d.rectangle((0,y,1023,y+255),fill=background)
        d.rectangle((0,y+216,1023,y+255),fill='#a7a99d')
        d.rectangle((0,y,1023,y+17),fill='#cbd4cc')
        if i in (0,4):
            d.rectangle((60,y+42,395,y+148),fill='#6c5140')
            for x in range(88,360,65):
                d.ellipse((x,y+58,x+30,y+88),fill='#d2b797')
            d.rectangle((440,y+135,990,y+220),fill='#765e4f')
            d.rectangle((440,y+128,990,y+142),fill='#ded0b4')
            for x in range(510,930,100):d.rectangle((x,y+102,x+28,y+128),fill='#eee5d3')
        elif i in (1,9):
            for x in (110,360,610,860):
                d.rounded_rectangle((x,y+37,x+130,y+155),radius=18,fill='#a9c1bb',outline='#ecede4',width=8)
                d.rectangle((x+36,y+170,x+100,y+215),fill='#5c6260')
        elif i==2:
            for level in range(2):
                yy=y+122+level*48
                d.rectangle((45,yy+28,975,yy+43),fill='#816744')
                for x in range(60,955,38):
                    color=rng.choice(('#d95e43','#e6b33f','#78a451','#b7b64c'))
                    d.ellipse((x,yy,x+25,yy+28),fill=color)
        elif i in (3,8):
            for x in (30,278,526,774):
                d.rectangle((x,y+50,x+213,y+213),fill='#b4c7ba',outline='#829b90',width=4)
                for level in range(3):
                    yy=y+78+level*43
                    d.line((x+6,yy+24,x+207,yy+24),fill='#f3f4e9',width=5)
                    for xx in range(x+12,x+196,25):
                        d.rectangle((xx,yy,xx+14,yy+22),fill=rng.choice(('#cb765b','#e5c86a','#8ab699','#ebe8d8')))
        elif i==5:
            d.rectangle((45,y+170,970,y+219),fill='#8b9f98')
            for x in (125,400,675):
                d.rectangle((x,y+87,x+160,y+167),fill='#e8ece4',outline='#85948f',width=5)
                d.rectangle((x+20,y+109,x+140,y+135),fill='#96a7a2')
        elif i==6:
            for x in (155,375,640,850):
                d.ellipse((x,y+74,x+120,y+194),outline='#373d3b',width=14)
                d.ellipse((x+36,y+110,x+84,y+158),outline='#b5beb3',width=4)
            d.line((80,y+206,945,y+206),fill='#565e58',width=9)
        elif i==7:
            for x in range(100,900,150):
                d.rectangle((x,y+116,x+105,y+210),fill='#bd9b70',outline='#947652',width=3)
                d.rectangle((x+15,y+130,x+50,y+139),fill='#eee9dc')
        elif i==10:
            # 博士眼镜店的蓝色装饰仅抽象表达实景配色，不复制现场插画。
            d.polygon(((0,y+170),(230,y+120),(380,y+212),(100,y+226)),fill='#f0cc45')
            d.polygon(((590,y+215),(780,y+146),(1023,y+200),(1023,y+232)),fill='#151e2b')
            for x in range(65,960,180):
                d.rectangle((x,y+48,x+130,y+184),fill='#decdb4',outline='#577e8d',width=4)
                for yy in (y+80,y+120,y+160):
                    d.ellipse((x+25,yy-10,x+51,yy+10),outline='#5c6b67',width=3)
                    d.ellipse((x+55,yy-10,x+81,yy+10),outline='#5c6b67',width=3)
        else:
            for x in range(50,950,140):
                d.rectangle((x,y+60,x+100,y+202),fill='#b7a7b9',outline='#8d7c91',width=4)
                for yy in (y+92,y+135,y+178):
                    d.line((x+5,yy,x+95,yy),fill='#ece9df',width=5)
        # 淡色玻璃反光保留遮挡感，同时让用途不同的陈设可辨认。
        d.polygon(((12,y+20),(45,y+20),(190,y+216),(157,y+216)),fill='#b9cfc4')
    window.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ASSETS/'xiyuan-store-window.ppm')
    # 照片中的 7-11 外是深浅交错的长条砖，而非统一的方形地砖。
    paving=Image.new('RGB',(256,256),(192,187,172))
    d=ImageDraw.Draw(paving)
    colors=((210,207,194),(188,181,165),(91,91,86),(224,220,208))
    for y in range(0,256,32):
        offset=(y//32%2)*32
        for x in range(-offset,256,64):
            color=colors[((x+offset)//64+y//32)%len(colors)]
            d.rectangle((x+1,y+1,x+62,y+30),fill=color)
    paving.save(ASSETS/'xiyuan-paving.ppm')


class MeshWriter:
    def __init__(self):
        self.vertices = array('f')

    def v(self, p, n, tex):
        self.vertices.extend((p[0], p[1], p[2], n[0], n[1], n[2], tex[0], tex[1]))

    def triangle(self, a, b, c, normal, uvs):
        for p, uv_ in zip((a, b, c), uvs):
            self.v(p, normal, uv_)

    def quad(self, a, b, c, d, normal, uv4):
        self.triangle(a, b, c, normal, uv4[:3])
        self.triangle(a, c, d, normal, (uv4[0], uv4[2], uv4[3]))

    def horizontal(self, geom, y):
        self.horizontal_mapped(geom,y,lambda x,z:(x/12.0,z/12.0))

    def horizontal_mapped(self, geom, y, uv_map):
        for poly in polygons(geom):
            poly = orient(poly, sign=1.0)
            rings = [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
            rings = [r for r in rings if len(r) >= 3]
            if not rings:
                continue
            vertices = np.asarray([p for ring in rings for p in ring], dtype=np.float64)
            ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
            indices = mapbox_earcut.triangulate_float64(vertices, ends)
            for index in indices:
                x, z = vertices[index]
                self.v((x, y, z), (0, 1, 0), uv_map(x,z))

    def save(self, path):
        if len(self.vertices) % 8:
            raise ValueError('Vertex buffer alignment')
        count = len(self.vertices) // 8
        if count < 3 or count > 500000:
            raise ValueError(f'{path.name}: unsupported vertex count {count}')
        if struct.pack('=I', 1) != struct.pack('<I', 1):
            self.vertices.byteswap()
        with path.open('wb') as output:
            output.write(b'SZM1' + struct.pack('<I', count))
            self.vertices.tofile(output)
        return count


def make_terrain():
    global terrain_grid, terrain_normals, terrain_patches, huizi_roof_levels, beixia_level, golf_level, wenshan_shore_terrain, road_building_bases, road_building_groups, zhounan_portal
    terrain_patches=[]
    mesh = MeshWriter()
    nx, nz = 152, 169
    dx, dz = (X1 - X0) / nx, (Z1 - Z0) / nz
    heights = np.asarray([[height_at(X0 + i * dx, Z0 + j * dz) for i in range(nx + 1)] for j in range(nz + 1)], dtype=np.float32)
    # 平滑低分辨率地形带来的孤立尖峰，保留大尺度起伏。
    for _ in range(2):
        p = np.pad(heights, 1, mode='edge')
        heights = (p[1:-1,1:-1]*4 + p[:-2,1:-1] + p[2:,1:-1] + p[1:-1,:-2] + p[1:-1,2:]) / 8
    # 两套公开全球高程在白石路附近存在明显分歧，Mapzen 局部出现负值。
    # 以附近低地最小高程 2 米作为保守的视觉修正，而非声称这是实测地面。
    np.maximum(heights, 2.0, out=heights)
    terrain_grid = heights
    # 粗 DEM 不能表示建筑基座，按实有建筑轮廓建立近似平整平台，减少穿插与悬空。
    for key, t, geom in buildings + surrounding:
        if geom.area < 250 or t.get('building') == 'stadium':
            continue
        point = geom.representative_point()
        level = surface_height(point.x,point.y)
        mask = Image.new('L',(nx+1,nz+1),0)
        draw = ImageDraw.Draw(mask)
        def grid_pt(q):
            return (round((q[0]-X0)/(X1-X0)*nx),round((q[1]-Z0)/(Z1-Z0)*nz))
        for poly in polygons(geom):
            draw.polygon([grid_pt(q) for q in poly.exterior.coords],fill=255)
            for inner in poly.interiors:
                draw.polygon([grid_pt(q) for q in inner.coords],fill=0)
        heights[np.asarray(mask)>0] = level
    # 用户提供的模型和两套粗高程均提示北侧较高；以 B 座为锚点近似台地，非实测。
    anchor = huizi_parts['230847100'].representative_point()
    huizi_lower = surface_height(anchor.x,anchor.y)
    huizi_upper = huizi_lower+4.0
    huizi_roof_levels = {'230847102':huizi_lower+13.2,'230847100':huizi_lower+16.5,
                         '230847098':huizi_upper+16.5,'230847099':huizi_upper+16.5}
    huiyuan_anchor = huiyuan_site.representative_point()
    huiyuan_level = surface_height(huiyuan_anchor.x,huiyuan_anchor.y)
    yanhui_anchor = yanhui_site.representative_point()
    yanhui_level = surface_height(yanhui_anchor.x,yanhui_anchor.y)
    yanhui_low = huiyuan_level
    # 日晷底座附近局部调平，避免粗 DEM 把石砌底座顶穿。
    dial_center = Point(-243.2,-48.6)
    dial_level = surface_height(dial_center.x,dial_center.y)
    dial_flat = prep(dial_center.buffer(13))
    star_center = Point(291,-400)
    star_level = surface_height(star_center.x,star_center.y)
    star_flat = prep(star_center.buffer(11))
    huiwen_point = huiwen_h3.representative_point()
    huiwen_level = surface_height(huiwen_point.x,huiwen_point.y)-1.25
    beixia_anchor = beixia_stand.representative_point()
    beixia_level = surface_height(beixia_anchor.x,beixia_anchor.y)
    golf_anchor = golf_tee.representative_point()
    golf_level = surface_height(golf_anchor.x,golf_anchor.y)
    yuanping_anchor = yuanping_building.representative_point()
    yuanping_level = surface_height(yuanping_anchor.x,yuanping_anchor.y)
    art_point = huicui_site.representative_point()
    art_level = surface_height(art_point.x,art_point.y)
    # 数十米 DEM 无法表示北馆前平台的横向平整台阶；两端高程仍取粗网格估计。
    north_low = surface_height(*north_library_steps.coords[0])
    north_high = surface_height(-126.4,-102.9)
    for j in range(nz + 1):
        for i in range(nx + 1):
            p = Point(X0 + i*dx, Z0 + j*dz)
            for prepared, _, lake_y in lake_heights:
                if prepared.covers(p):
                    heights[j,i] = lake_y
                    break
            for prepared, stadium_y in flat_sports:
                if prepared.covers(p):
                    heights[j,i] = stadium_y
                    break
            if plaza_flat.covers(p):
                heights[j,i] = plaza_y
            if dial_flat.covers(p):
                heights[j,i] = dial_level
            if star_flat.covers(p):
                heights[j,i] = star_level
            # 深南大道并非高山：根据实景、道路走向及 GLO-30 的 24–30 米采样，
            # 修正校外道路附近 Mapzen 的异常起伏；这是近似地面而非实测高程。
            if not campus_interior.covers(p) and shennan_outer.covers(p):
                road_y = 28.0-.01*min(420.0,max(0.0,p.x-300.0))
                if shennan_inner.covers(p):
                    heights[j,i] = road_y
                else:
                    blend=min(1.0,shennan_corridor.distance(p)/20.0)
                    blend=blend*blend*(3-2*blend)
                    heights[j,i]=road_y*(1-blend)+heights[j,i]*blend
            # 低分辨率 DEM 将球场拉成十余米高的波峰；保留极缓坡并平滑接回周边。
            # 坡度与高程仅按照片估计，不能作为测绘地形。
            golf_y = golf_level-.006*(p.x-golf_anchor.x)+.005*(p.y-golf_anchor.y)
            if golf_inner.covers(p):
                heights[j,i] = golf_y
            elif golf_outer.covers(p):
                blend = min(1.0,golf_flat_area.distance(p)/16.0)
                heights[j,i] = golf_y*(1-blend)+heights[j,i]*blend
            # 两个露天池和看台相接，使用共同估计平台，并在周围与粗 DEM 渐变。
            if beixia_inner.covers(p):
                heights[j,i] = beixia_level-.12
            elif beixia_outer.covers(p):
                blend = min(1.0,beixia_site.distance(p)/14.0)
                heights[j,i] = (beixia_level-.12)*(1-blend)+heights[j,i]*blend
            # 场馆正面照片显示入口与广场之间有台阶；1.2 米为建模估计。
            if yuanping_inner.covers(p):
                heights[j,i] = yuanping_level-1.2
            elif yuanping_outer.covers(p):
                blend = min(1.0,yuanping_flat_area.distance(p)/12.0)
                heights[j,i] = (yuanping_level-1.2)*(1-blend)+heights[j,i]*blend
            # 为广场高平台及下方泳池留出空间，并与周边粗地形缓慢衔接。
            if stone_site_interior.covers(p):
                heights[j,i] = pool_level-.15
            elif stone_site_outer.covers(p):
                blend = min(1.0,stone_site.distance(p)/14.0)
                heights[j,i] = (pool_level-.15)*(1-blend)+heights[j,i]*blend
            # 入口实拍可见宽台阶；在粗高程中局部留出近似的铺装及踏步空间。
            if huiwen_approach_inner.covers(p):
                heights[j,i] = huiwen_level
            elif huiwen_approach_outer.covers(p):
                blend = min(1.0,huiwen_flat_area.distance(p)/12.0)
                heights[j,i] = huiwen_level*(1-blend)+heights[j,i]*blend
            # 湖岸的两组弧形踏步高差小于 DEM 网格间距；压低基底再平滑连接岸坡。
            if wenshan_site_inner.covers(p):
                heights[j,i] = upper_lake_level
            elif wenshan_site_outer.covers(p):
                blend = min(1.0,max(0.0,(wenshan_sites.distance(p)-5.0)/13.0))
                heights[j,i] = upper_lake_level*(1-blend)+heights[j,i]*blend
            if trefoil_inner.covers(p):
                heights[j,i] = upper_lake_level+.95
            elif trefoil_outer.covers(p):
                blend = min(1.0,max(0.0,(trefoil_patio.distance(p)-5.0)/10.0))
                heights[j,i] = (upper_lake_level+.95)*(1-blend)+heights[j,i]*blend
            # 两塔和相邻裙房共用平台；外缘逐渐过渡到周围地形，非实测标高。
            if zhiyi_flat_inner.covers(p):
                heights[j,i] = zhiyi_level
            elif zhiyi_flat_outer.covers(p):
                blend = min(1.0,max(0.0,(zhiyi_site.distance(p)-12.0)/18.0))
                heights[j,i] = zhiyi_level*(1-blend)+heights[j,i]*blend
            # 粗 DEM 将主入口广场与相邻楼体拉出数米高差；按连续平台近似修正，非实测。
            if huicui_flat_inner.covers(p):
                heights[j,i] = art_level
            elif huicui_flat_outer.covers(p):
                blend = min(1.0,huicui_flat_area.distance(p)/10.0)
                heights[j,i] = art_level*(1-blend)+heights[j,i]*blend
            if huizi_outer.covers(p):
                ab_distance = huizi_ab.distance(p)
                cd_distance = huizi_cd.distance(p)
                # 建筑外侧按到两组体量的距离过渡，避免相邻道路出现突然的高差。
                total=ab_distance+cd_distance
                blend_cd=ab_distance/total if total>1e-6 else .5
                level=huizi_lower+(huizi_upper-huizi_lower)*blend_cd
                if huizi_inner.covers(p):
                    heights[j,i] = level
                else:
                    blend = min(1.0,max(0.0,(huizi_site.distance(p)-5.0)/15.0))
                    heights[j,i] = level*(1-blend)+heights[j,i]*blend
            # 校园道路由地图中心线确定；路面高程沿楼侧缓慢变化并与坡地混合，非实测。
            if huizi_side_road_outer.covers(p) and not huizi_site_guard.covers(p):
                road_y=huizi_lower-1.5+2.1*huizi_side_road.project(p)/huizi_side_road.length
                blend=min(1.0,max(0.0,(huizi_side_road.distance(p)-6.0)/16.0))
                blend=blend*blend*(3-2*blend)
                heights[j,i]=road_y*(1-blend)+heights[j,i]*blend
            # 宽台阶下方留出估计高差，并在外侧与周边地面渐变衔接。
            if huizi_approach_inner.covers(p):
                heights[j,i] = huizi_lower-1.15
            elif huizi_approach_outer.covers(p):
                blend = min(1.0,max(0.0,(huizi_approach.distance(p)-2.0)/6.0))
                heights[j,i] = (huizi_lower-1.15)*(1-blend)+heights[j,i]*blend
            # 入口通道的粗网格原先比门廊下沉近两米，需留出连续铺装并平滑过渡。
            if huiyuan_entry_inner.covers(p):
                heights[j,i] = huiyuan_level-.08
            elif huiyuan_entry_outer.covers(p):
                blend = min(1.0,huiyuan_entry_area.distance(p)/12.0)
                heights[j,i] = (huiyuan_level-.08)*(1-blend)+heights[j,i]*blend
            # 航拍中日晷与办公楼入口处在连续铺装空间；以入口粗高程作锚点局部调平。
            # 该平台与边缘渐变均为建模推定，不能作为实测高程。
            if time_square_inner.covers(p):
                heights[j,i] = huiyuan_level
            elif time_square_outer.covers(p):
                blend=min(1.0,max(0.0,(time_square_paving.distance(p)-4.0)/14.0))
                heights[j,i] = huiyuan_level*(1-blend)+heights[j,i]*blend
            # 演会中心是坡地上的单层半开放厅，平台及台阶高程为粗地形约束下的估计。
            if yanhui_site_inner.covers(p):
                heights[j,i] = yanhui_level
            elif yanhui_site_outer.covers(p):
                blend = min(1.0,max(0.0,(yanhui_site.distance(p)-4.0)/12.0))
                heights[j,i] = yanhui_level*(1-blend)+heights[j,i]*blend
            if yanhui_step_inner.covers(p) or yanhui_step_outer.covers(p):
                along=yanhui_step_line.project(p)/yanhui_step_line.length
                level=yanhui_low*(1-along)+yanhui_level*along-.12
                if yanhui_step_inner.covers(p):
                    heights[j,i]=level
                else:
                    blend=min(1.0,yanhui_step_area.distance(p)/7.0)
                    heights[j,i]=level*(1-blend)+heights[j,i]*blend
            if north_library_step_inner.covers(p) or north_library_step_outer.covers(p):
                along = north_library_steps.project(p)/north_library_steps.length
                stepped_level = north_low*(1-along)+north_high*along
                if north_library_step_inner.covers(p):
                    heights[j,i] = stepped_level
                else:
                    blend = min(1.0,north_library_step_site.distance(p)/7.0)
                    heights[j,i] = stepped_level*(1-blend)+heights[j,i]*blend
            # 校门硬质前场应连续平整；具体标高没有实测，只平滑粗 DEM 的局部起伏。
            if zhiyuan_entry_inner.covers(p):
                heights[j,i]=2.1
            elif zhiyuan_entry_outer.covers(p):
                blend=min(1.0,max(0.0,(zhiyuan_entry.distance(p)-6.0)/12.0))
                blend=blend*blend*(3-2*blend)
                heights[j,i]=2.1*(1-blend)+heights[j,i]*blend
            # 北门及两侧人行道采用连续的近似纵坡，向两侧缓慢接回已有地形。
            # 这也避免低分辨率栅格给照片中的岗亭造成数米的地基高差。
            if legong_road_outer.covers(p) or legong_entry_outer.covers(p):
                road_distance=legong_road.distance(p)
                cross_distance=legong_crossroad.distance(p)
                road_weight=1.0-min(1.0,max(0.0,(road_distance-16.0)/18.0))
                cross_weight=1.0-min(1.0,max(0.0,(cross_distance-8.0)/12.0))
                entry_weight=1.0-min(1.0,legong_entry_area.distance(p)/25.0)
                entry_weight=entry_weight*entry_weight*(3-2*entry_weight)
                weight=max(road_weight,cross_weight,entry_weight)
                weight=weight*weight*(3-2*weight)
                level=float(np.interp(p.y,[-760,-740,-720,-700,-680,-650,-620,-590,-560,-530,-500],
                                      [28,26,24,23,22,20.5,19,18,17,17,18.2]))
                # 东侧向邻近练习场的平台高程过渡，消除道路旁的尖锐三角深沟。
                east_blend=entry_weight*min(1.0,max(0.0,(p.x-300.0)/40.0))
                level=level*(1-east_blend)+golf_y*east_blend
                heights[j,i]=level*weight+heights[j,i]*(1-weight)
    smooth_campus_road_terrain(heights)
    calibrate_qiaoyuan_terrain(heights)
    smooth_xiyuan_shop_street(heights)
    smooth_lakeside_road_terrain(heights)
    wenshan_shore_terrain=campus_wenshan_shore.smooth_east_bank(
        heights,x_bounds=(X0,X1),z_bounds=(Z0,Z1),lake=upper_lake,lake_level=upper_lake_level,
        road=LineString(way_points(ways['231530782'])),buildings=buildings)
    # 石头坞与学生活动中心之间，粗 DEM 和广场平台造成突兀的路面折角。
    # 保留平台与建筑，沿地图道路局部平滑；配置参数和结果均不代表实测高程。
    slope_config=json.loads((DATA/'road-slope-adjustments.json').read_text(encoding='utf-8'))
    slope_results=[]
    road_building_bases={}
    road_building_groups={}
    for route in slope_config['routes']:
        road_geometry=next(g for key,_,g in ground_vehicle_roads if key=='way/'+route['osm_way'])
        # 裁剪后的地图道路可能分成多段；只拟合最靠近配置中心的连续段。
        road=min(lines(road_geometry),key=lambda g:g.distance(Point(*route['center_xz_m'])))
        edge_buildings=set(route.get('road_edge_buildings',()))
        for key in route.get('retain_base_buildings',()):
            geom=next(g for ident,_,g in buildings if ident==key)
            p=geom.representative_point()
            road_building_bases.setdefault(key,surface_height(p.x,p.y))
            road_building_groups[key]=route['foundation_group']
        # 仅在名单中的临路楼体边缘放宽地形限制，楼体核心及其他地标仍受到保护。
        extra_ids={'way/'+item['osm_way'] for item in route.get('additional_roads',())}
        edge_corridor=unary_union([road]+[g for key,_,g in ground_vehicle_roads if key in extra_ids]).buffer(
            route.get('road_edge_buffer_m',8.0))
        parts=[g.difference(edge_corridor) if key in edge_buildings else g
               for key,_,g in buildings]
        parts.extend((stone_plaza,south_pool))
        # 接入已调整好的主路时固定其附近地形，避免支路修坡改变交叉口的既有标高。
        for item in route.get('protected_roads',()):
            crossing=next(g for key,_,g in ground_vehicle_roads if key=='way/'+item['osm_way'])
            parts.append(crossing.buffer(item['buffer_m']))
        if route.get('protect_time_square'):
            parts.extend((time_square_paving,huiyuan_entry_area,yanhui_step_area,yanhui_site))
        if route.get('detail_bounds_m'):
            parts.extend([beixia_deck.buffer(3),beixia_stand.buffer(3)]
                         +[g.buffer(2) for _,_,g in waters+pitches])
        elif edge_buildings:
            parts.extend([beixia_deck,beixia_stand]+[g for _,_,g in waters+pitches])
        protected=unary_union(parts)
        if route.get('detail_bounds_m'):
            additional=[]
            for extra in route.get('additional_roads',()):
                geometry=next(g for key,_,g in ground_vehicle_roads if key=='way/'+extra['osm_way'])
                path=min(lines(geometry),key=lambda g:g.distance(Point(*extra['center_xz_m'])))
                additional.append({'osm_way':extra['osm_way'],'line':path,'center':extra['center_xz_m'],
                                   'half_length':extra['half_length_m'],'target_grade':extra['target_grade'],
                                   'lock_protected_profile':extra.get('lock_protected_profile',False),
                                   'keep_endpoint_fit':extra.get('keep_endpoint_fit',False),
                                   'fit_spacing':extra.get('fit_spacing_m'),
                                   'core_weight':extra.get('core_weight',5.0),
                                   'edge_weight':extra.get('edge_weight',2.0)})
            patch,result=campus_terrain_patch.smooth_road_patch(
                heights,coarse_bounds=(X0,X1,Z0,Z1),bounds=route['detail_bounds_m'],
                spacing=route['detail_spacing_m'],line=road,center=route['center_xz_m'],
                half_length=route['half_length_m'],target_grade=route['target_grade'],protected=protected,
                additional_routes=additional,lock_protected_profile=route.get('lock_protected_profile',False),
                keep_endpoint_fit=route.get('keep_endpoint_fit',False),fit_spacing=route.get('fit_spacing_m'),
                core_weight=route.get('core_weight',5.0),edge_weight=route.get('edge_weight',2.0),
                landscape_smoothing_steps=route.get('landscape_smoothing_steps',0))
            for previous in terrain_patches:
                if box(patch.x0,patch.z0,patch.x1,patch.z1).intersection(
                        box(previous.x0,previous.z0,previous.x1,previous.z1)).area>.001:
                    raise ValueError('Terrain detail patches must not overlap')
            terrain_patches.append(patch)
        else:
            result=campus_road_terrain.smooth_segment(
                heights,line=road,center=route['center_xz_m'],half_length=route['half_length_m'],
                target_grade=route['target_grade'],x_bounds=(X0,X1),z_bounds=(Z0,Z1),protected=protected)
        slope_results.append({'name':route['name'],'osm_way':route['osm_way'],**result})
    (DATA/'road-alignment-profile.json').write_text(
        json.dumps({'adjustments':road_alignment_profiles},ensure_ascii=False,indent=2),encoding='utf-8')
    (DATA/'road-slope-calibration.json').write_text(
        json.dumps({'routes':slope_results},ensure_ascii=False,indent=2),encoding='utf-8')
    terrain_grid = heights
    zhounan_portal = campus_zhounan_portal.layout(
        way_points(ways['574956548']),way_points(ways['984608622']),
        vehicle_surface,terrain_patches,rendered_terrain_height)
    # 自由相机使用相同高程网格，避免移动时穿到地表下方。
    with (ASSETS / 'campus-heights.bin').open('wb') as output:
        output.write(struct.pack('<II4f', nx + 1, nz + 1, X0, X1, Z0, Z1))
        for row in heights:
            output.write(struct.pack('<' + 'f' * len(row), *row))
    normals = []
    for j in range(nz + 1):
        row = []
        for i in range(nx + 1):
            gx = (heights[j][min(i + 1, nx)] - heights[j][max(i - 1, 0)]) / (2 * dx)
            gz = (heights[min(j + 1, nz)][i] - heights[max(j - 1, 0)][i]) / (2 * dz)
            length = math.sqrt(gx * gx + 1 + gz * gz)
            row.append((-gx / length, 1 / length, -gz / length))
        normals.append(row)
    terrain_normals=np.asarray(normals,dtype=np.float32)
    for patch in terrain_patches:
        patch.make_normals(coarse_terrain_normal)
    campus_terrain_patch.write_patches(ASSETS/'campus-terrain-patches.bin',terrain_patches)
    for j in range(nz):
        for i in range(nx):
            for x,z,sx,sz in terrain_cell_parts(i,j):
                if sx==dx and sz==dz:
                    for di,dj in ((0,0),(0,1),(1,1),(0,0),(1,1),(1,0)):
                        px,pz=x+di*dx,z+dj*dz
                        mesh.v((px,heights[j+dj][i+di],pz),normals[j+dj][i+di],uv(px,pz))
                else:
                    for di,dj in ((0,0),(0,1),(1,1),(0,0),(1,1),(1,0)):
                        px,pz=x+di*sx,z+dj*sz
                        mesh.v((px,rendered_terrain_height(px,pz),pz),rendered_terrain_normal(px,pz),uv(px,pz))
    return mesh.save(ASSETS / 'campus-terrain.mesh')


def city_surface(x,z):
    # 城市带只作背景：与校园接缝一致，向外平滑过渡至约 26 米。
    edge_x=min(x,X1)
    if z >= Z0:
        edge=surface_height(edge_x,z)
    else:
        t=min(1.0,(Z0-z)/80.0)
        t=t*t*(3-2*t)
        edge=surface_height(edge_x,Z0)*(1-t)+26.0*t
    t=min(1.0,max(0.0,(x-X1)/80.0))
    t=t*t*(3-2*t)
    return edge*(1-t)+26.0*t


def make_north_city_ground():
    # 贴图沿 OSM 中心线绘制；车道宽度与标线只是常见道路的近似。
    width,height=4096,2048
    grass=Image.open(ASSETS/'grass.ppm').convert('RGB').resize((512,512))
    grass=ImageEnhance.Brightness(ImageEnhance.Color(grass).enhance(.72)).enhance(1.10)
    tile=Image.blend(Image.new('RGB',(512,512),'#78966a'),grass,.22)
    ground=Image.new('RGB',(width,height))
    for x in range(0,width,512):
        for y in range(0,height,512): ground.paste(tile,(x,y))
    draw=ImageDraw.Draw(ground)
    def pixel_city(point):
        return (round((point[0]-X0)/(CITY_EAST-X0)*(width-1)),
                round((CITY_SOUTH-point[1])/(CITY_SOUTH-CITY_NORTH)*(height-1)))
    roads=[]
    road_width={'footway':2.3,'path':1.8,'pedestrian':4,'cycleway':2.7,'service':5,
                'residential':6,'unclassified':6.5,'tertiary':8,'secondary':10,
                'primary':12,'primary_link':7,'secondary_link':7,'tertiary_link':7}
    for way in city_ways:
        t=tags(way); p=city_points(way)
        if len(p)<2: continue
        if t.get('highway') in road_width and t.get('bridge') in (None,'no') and t.get('tunnel') in (None,'no'):
            geom=LineString(p).intersection(city_extent)
            if not geom.is_empty: roads.append((way.get('id'),t,geom))
    # 先画小路，再画干道，避免支路覆盖主路。
    for group in road_width:
        for osm_id,t,geom in roads:
            if t['highway']!=group: continue
            w=14.5 if osm_id in shennan_ways and group=='primary' else road_width[group]
            pedestrian=group in ('footway','path','pedestrian','cycleway')
            for line in lines(geom):
                coords=[pixel_city(p) for p in line.coords]
                if len(coords)<2: continue
                if not pedestrian: draw.line(coords,fill='#b8b9aa',width=round((w+1.6)*width/(CITY_EAST-X0)),joint='curve')
                draw.line(coords,fill='#d4cfbc' if pedestrian else '#777d79',width=max(1,round(w*width/(CITY_EAST-X0))),joint='curve')
                if group in ('primary','secondary','tertiary'):
                    offsets=(-3.6,0,3.6) if osm_id in shennan_ways and group=='primary' else (0,)
                    for offset in offsets:
                        mark=line.offset_curve(offset) if offset else line
                        for section in lines(mark):
                            for dist in np.arange(0,section.length,10):
                                a,b=section.interpolate(dist),section.interpolate(min(dist+4,section.length))
                                if a.distance(b)>.1: draw.line((pixel_city((a.x,a.y)),pixel_city((b.x,b.y))),fill='#d8d9c9',width=1)
    # 沿北、东两条接缝继承校园贴图像素，再向外渐变，消除色块断层。
    old=Image.open(ASSETS/'campus-ground.ppm').convert('RGB')
    sx,sy=pixel_city((X1,Z0))
    north_edge=old.crop((0,0,old.width,1)).resize((sx+1,1))
    for k in range(34):
        y=sy-1+k
        row=ground.crop((0,y,sx+1,y+1))
        ground.paste(Image.blend(north_edge,row,max(0,k-1)/32.0),(0,y))
    old_end=round((CITY_SOUTH-Z0)/(Z1-Z0)*(old.height-1))
    east_edge=old.crop((old.width-1,0,old.width,old_end+1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((1,sy+1))
    for k in range(34):
        x=sx-1+k
        column=ground.crop((x,0,x+1,sy+1))
        ground.paste(Image.blend(east_edge,column,max(0,k-1)/32.0),(x,0))
    ground.save(ASSETS/'campus-north-city-ground.ppm')
    mesh=MeshWriter()
    # 两个矩形组成 L 形远景，避免覆盖原有校园地形。
    for left,right,south,north in ((X0,CITY_EAST,Z0,CITY_NORTH),(X1,CITY_EAST,CITY_SOUTH,Z0)):
        nx,nz=round((right-left)/10),round((south-north)/10)
        dx,dz=(right-left)/nx,(south-north)/nz
        for j in range(nz):
            for i in range(nx):
                for di,dj in ((0,0),(1,0),(1,1),(0,0),(1,1),(0,1)):
                    x,z=left+(i+di)*dx,south-(j+dj)*dz
                    y=city_surface(x,z)
                    gx=(city_surface(x+1,z)-city_surface(x-1,z))/2
                    gz=(city_surface(x,z+1)-city_surface(x,z-1))/2
                    length=math.sqrt(gx*gx+1+gz*gz)
                    mesh.v((x,y,z),(-gx/length,1/length,-gz/length),
                           ((x-X0)/(CITY_EAST-X0),(CITY_SOUTH-z)/(CITY_SOUTH-CITY_NORTH)))
    return mesh.save(ASSETS/'campus-north-city-ground.mesh')


def height_for(key, t, geom):
    if key in HEIGHTS:
        return HEIGHTS[key]
    try:
        raw = t.get('height', '').replace('m', '').strip()
        if raw and 2 <= float(raw) <= 150:
            return float(raw), 'OSM height'
    except ValueError:
        pass
    try:
        levels = float(t.get('building:levels', ''))
        if 1 <= levels <= 40:
            return levels * 3.3, f'OSM {levels:g}层，层高估算3.3米'
    except ValueError:
        pass
    kind = t.get('building', '')
    estimate = 18.0 if kind in ('dormitory', 'apartments') else 14.0 if geom.area > 650 else 9.0
    return estimate, '未见明确高度，按用途和轮廓估算'


def building_style(key, t, geom):
    name = NAMES.get(key, t.get('name', ''))
    if key == 'way/1085792762':
        return 'huiyan'
    if name.startswith(('汇文楼', '文科楼办公区')):
        return 'huiwen'
    # 中区部分教学楼仍使用白墙和深色水平窗带。
    if name.startswith(('汇紫楼', '汇元楼')):
        return 'strip'
    # 部分白墙建筑使用点窗；致艺楼与致真楼在下方另有独立几何。
    if key in ('way/629940889', 'way/629940890', 'way/629940892', 'way/319101984', 'way/231086125'):
        return 'perforated'
    if key == 'way/319101983':
        return 'teng'
    # 校方宿舍建筑资料称南区四栋是白色竖线条，乔苑为灰白色；不把地图中的红色图例当墙色。
    if name in ('春笛','夏筝','秋瑟','冬筑','高音','低音'):
        return 'perforated'
    if key in ('way/1258216924','way/1258216925'):
        return 'qiaomu'
    if key in ('way/629940893','way/629940894','way/629940895'):
        return 'qiaosen'
    if key in ('way/1558576826','way/1558576825','way/1558576824'):
        return 'zhai'
    if name.startswith('乔'):
        return 'cream'
    if t.get('building') in ('dormitory', 'apartments') or name.endswith(('斋', '轩', '阁')):
        return 'red'
    return 'cream'


def extrude(mesh, roof, geom, base, top, unit_width=4.0, unit_height=3.3, cap=True):
    if cap:
        roof.horizontal(geom, top)
    for p in polygons(geom):
        p = orient(p, sign=1.0)
        for ring in [p.exterior] + list(p.interiors):
            coords = list(ring.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                dx, dz = b[0] - a[0], b[1] - a[1]
                length = math.hypot(dx, dz)
                if length < .05:
                    continue
                normal = (dz / length, 0.0, -dx / length)
                repeat = length / unit_width
                rows = (top - base) / unit_height
                mesh.quad((a[0], base, a[1]), (b[0], base, b[1]),
                          (b[0], top, b[1]), (a[0], top, a[1]), normal,
                          ((0,0), (repeat,0), (repeat,rows), (0,rows)))


def surface_normal(a, b, c):
    normal = np.cross(np.asarray(b)-a,np.asarray(c)-a)
    normal /= max(np.linalg.norm(normal),.0001)
    return -normal if normal[1] < 0 else normal


def extrude_teng(walls, roof, glazed, geom, ground, height):
    # 对照致腾楼照片，将圆角入口中段向内退让，形成真实的凹进式玻璃入口。
    roof.horizontal(geom,ground+height)
    for poly in polygons(geom):
        poly = orient(poly,sign=1.0)
        for ring_index,ring in enumerate([poly.exterior]+list(poly.interiors)):
            coords = list(ring.coords)
            for a,b in zip(coords[:-1],coords[1:]):
                dx,dz = b[0]-a[0],b[1]-a[1]
                length = math.hypot(dx,dz)
                if length < .05:
                    continue
                nx,nz = dz/length,-dx/length
                normal = (nx,0,nz)
                mx,mz = (a[0]+b[0])*.5,(a[1]+b[1])*.5
                portal = ring_index == 0 and ((mx < -25 and mz > 736) or (mx < -7 and mz > 757))
                def wall_band(bottom,top):
                    walls.quad((a[0],bottom,a[1]),(b[0],bottom,b[1]),
                               (b[0],top,b[1]),(a[0],top,a[1]),normal,
                               ((0,(bottom-ground+2)/3.3),(length/4,(bottom-ground+2)/3.3),
                                (length/4,(top-ground+2)/3.3),(0,(top-ground+2)/3.3)))
                if not portal:
                    wall_band(ground-2,ground+height)
                    continue
                bottom,top = ground+4,ground+25
                wall_band(ground-2,bottom)
                wall_band(top,ground+height)
                aa,bb = (a[0]-nx*2.0,a[1]-nz*2.0),(b[0]-nx*2.0,b[1]-nz*2.0)
                glazed.quad((aa[0],bottom,aa[1]),(bb[0],bottom,bb[1]),
                            (bb[0],top,bb[1]),(aa[0],top,aa[1]),normal,
                            ((0,0),(length/4,0),(length/4,6),(0,6)))
                for y in (bottom,top):
                    roof.quad((a[0],y,a[1]),(b[0],y,b[1]),(bb[0],y,bb[1]),(aa[0],y,aa[1]),
                              (0,1,0),((0,0),(1,0),(1,1),(0,1)))


def glass_accent(mesh, geom, bottom, top, length_filter, fraction=.60, limit=None):
    # 照片中的局部深色幕墙，按建筑边界放置；窗框和具体开窗位置未测量。
    edges = []
    for poly in polygons(geom):
        coords = list(orient(poly,sign=1.0).exterior.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            dx,dz = b[0]-a[0],b[1]-a[1]
            length = math.hypot(dx,dz)
            if length_filter(length):
                edges.append((length,a,b,dx,dz))
    if limit is not None:
        edges = sorted(edges,reverse=True)[:limit]
    for length,a,b,dx,dz in edges:
        nx,nz = dz/length,-dx/length
        start,end = (1-fraction)*.5,(1+fraction)*.5
        aa = (a[0]+dx*start+nx*.12,a[1]+dz*start+nz*.12)
        bb = (a[0]+dx*end+nx*.12,a[1]+dz*end+nz*.12)
        mesh.quad((aa[0],bottom,aa[1]),(bb[0],bottom,bb[1]),
                  (bb[0],top,bb[1]),(aa[0],top,aa[1]),(nx,0,nz),
                  ((0,0),(length*fraction/4,0),(length*fraction/4,(top-bottom)/3.3),(0,(top-bottom)/3.3)))


def huiwen_roof_detail(roof_detail, geom, y):
    # 航拍中多栋屋顶可见白色条形格架：用足迹裁剪格栅，间距为视觉估计。
    rectangle = geom.minimum_rotated_rectangle
    p = list(rectangle.exterior.coords)
    angle = math.degrees(math.atan2(p[1][1]-p[0][1],p[1][0]-p[0][0]))
    aligned = affinity.rotate(geom,-angle,origin=geom.centroid)
    inset = aligned.buffer(-1.0)
    if inset.is_empty:
        return
    xmin,zmin,xmax,zmax = inset.bounds
    for x in np.arange(xmin+1,xmax,4.0):
        for section in lines(inset.intersection(LineString(((x,zmin-1),(x,zmax+1))))):
            coords = list(affinity.rotate(section,angle,origin=geom.centroid).coords)
            for a,b in zip(coords[:-1],coords[1:]):
                beam(roof_detail,(a[0],y,a[1]),(b[0],y,b[1]),.11)
    for z in np.arange(zmin+1,zmax,8.0):
        for section in lines(inset.intersection(LineString(((xmin-1,z),(xmax+1,z))))):
            coords = list(affinity.rotate(section,angle,origin=geom.centroid).coords)
            for a,b in zip(coords[:-1],coords[1:]):
                beam(roof_detail,(a[0],y,a[1]),(b[0],y,b[1]),.09)


def zhiyi_facade(shell, glass, vents, rails, roof, geom, base, height, osm_id):
    """致艺楼的厚墙、内凹窗洞和少量开放窗洞；具体洞口位置按照片估计。"""
    # 墙与窗分开生成，洞口四周有实际侧壁；不是把深色方块画在平面上。
    level_height=height/10.0
    podium=way_polygon(ways['776358621'])
    ground_path=LineString(way_points(ways['776358640']))
    tower_seed=int(osm_id[-3:])
    landing=zhiyi_bridge_ends[0 if osm_id.endswith('889') else 1]
    uv4=((0,0),(1,0),(1,1),(0,1))
    for poly in polygons(geom):
        coords=list(orient(poly,sign=1.0).exterior.coords)
        for edge_index,(a,b) in enumerate(zip(coords[:-1],coords[1:])):
            dx,dz=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dz)
            if length<.1:
                continue
            tx,tz=dx/length,dz/length
            nx,nz=tz,-tx
            normal=(nx,0,nz)
            edge_line=LineString((a,b))
            landing_s=edge_line.project(landing) if edge_line.distance(landing)<.1 else None
            crossing=edge_line.intersection(ground_path)
            crossings=([crossing] if crossing.geom_type=='Point' else
                       list(crossing.geoms) if crossing.geom_type=='MultiPoint' else [])
            ground_openings=[]
            for p in crossings:
                d=ground_path.project(p)
                before=ground_path.interpolate(max(0,d-.4))
                after=ground_path.interpolate(min(ground_path.length,d+.4))
                wx,wz=after.x-before.x,after.y-before.y
                wl=max(.001,math.hypot(wx,wz))
                sine=max(.16,abs(tx*wz/wl-tz*wx/wl))
                half=min(9.,2.25/sine+.25)
                center=edge_line.project(p)
                ground_openings.append((max(0.,center-half),min(length,center+half)))
            adjacent=podium.distance(Point((a[0]+b[0])*.5,(a[1]+b[1])*.5))<.25
            def point(s,y,depth=0):
                return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
            def front(mesh,left,right,bottom,top,depth=0):
                if right-left>.01 and top-bottom>.01:
                    mesh.quad(point(left,bottom,depth),point(right,bottom,depth),
                              point(right,top,depth),point(left,top,depth),normal,uv4)
            # 掩埋在地面以下的基座保留完整墙体。
            front(shell,0,length,base-2,base)
            columns=max(1,round(length/3.7))
            bay=length/columns
            for floor in range(10):
                y0,y1=base+floor*level_height,base+(floor+1)*level_height
                if adjacent and floor<3:
                    # 与裙房共墙的低层由裙房墙体覆盖，避免重合闪烁。
                    continue
                if floor==0 and ground_openings:
                    # 召南桥地图步线穿过塔楼边界；两侧墙和门楣保留，中央留出通道。
                    end=0.0
                    for left,right in sorted(ground_openings):
                        front(shell,end,left,y0,y1)
                        front(shell,left,right,min(y1,y0+3.1),y1)
                        end=max(end,right)
                    front(shell,end,length,y0,y1)
                    continue
                if bay<2.5:
                    front(shell,0,length,y0,y1)
                    continue
                for col in range(columns):
                    s0,s1=col*bay,(col+1)*bay
                    key=tower_seed+edge_index*37+floor*19+col*11
                    bridge_door=landing_s is not None and floor in (4,6,8) and s0<=landing_s<s1
                    opened=(not bridge_door and floor in (3,6) and columns>=5 and 1<col<columns-2 and key%17==0)
                    width=min(bay-.16,2.6) if bridge_door else min(bay*.67,2.1 if opened else (.95+.16*(key%4)))
                    win_height=2.65 if bridge_door else 2.05 if opened else (1.05+.14*((key//5)%4))
                    center=landing_s if bridge_door else (s0+s1)*.5
                    left=max(s0+.08,min(s1-.08-width,center-width*.5))
                    right=left+width
                    bottom=y0+(0 if bridge_door else .48 if opened else .92)
                    top=min(y1-.18,bottom+win_height)
                    depth=1.65 if opened else .72
                    front(shell,s0,s1,y0,bottom)
                    front(shell,s0,s1,top,y1)
                    front(shell,s0,left,bottom,top)
                    front(shell,right,s1,bottom,top)
                    shell.quad(point(left,bottom),point(left,bottom,depth),point(left,top,depth),point(left,top),
                               (tx,0,tz),uv4)
                    shell.quad(point(right,bottom,depth),point(right,bottom),point(right,top),point(right,top,depth),
                               (-tx,0,-tz),uv4)
                    shell.quad(point(left,top),point(right,top),point(right,top,depth),point(left,top,depth),
                               (0,-1,0),uv4)
                    shell.quad(point(left,bottom,depth),point(right,bottom,depth),point(right,bottom),point(left,bottom),
                               (0,1,0),uv4)
                    if bridge_door:
                        # 连桥正对的三层开口保持可通行，不以玻璃或栏杆封死。
                        pass
                    elif opened:
                        # 个别较大洞口有可看见的栏杆和内侧平台，数量与位置为近似。
                        for h in (.38,.90):
                            beam(rails,point(left+.08,bottom+h,.45),point(right-.08,bottom+h,.45),.025)
                        for s in (left+.08,right-.08):
                            beam(rails,point(s,bottom,.45),point(s,bottom+.90,.45),.025)
                    else:
                        front(glass,left,right,bottom,top,depth)
                        # 小方格是遮蔽空调设备的格栅，不与窗洞重叠。
                        if key%3 != 0 and left-s0>.66:
                            size=.46
                            front(vents,s0+.12,s0+.12+size,y0+.90,y0+.90+size,-.012)
    # 开放洞口后保留楼层平台，用于表达公共空间的深度关系。
    for floor in range(1,10):
        roof.horizontal(geom,base+floor*level_height)
    roof.horizontal(geom,base+height)


def zhizhi_building(shell,glass,frame,roof,geom,base,height):
    """致知楼折线轮廓及七层玻璃格架，门廊和窗格尺寸为航拍估计。"""
    outline=orient(geom.simplify(.1,preserve_topology=True),sign=1.0)
    roof.horizontal(outline,base+height)
    coords=list(outline.exterior.coords)
    for edge_index,(a,b) in enumerate(zip(coords[:-1],coords[1:])):
        dx,dz=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dz)
        if length<.1: continue
        tx,tz=dx/length,dz/length
        nx,nz=tz,-tx
        normal=(nx,0,nz)
        def pt(s,y,depth=0):
            return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
        def face(mesh,s0,s1,y0,y1,depth=0,uv=((0,0),(1,0),(1,1),(0,1))):
            mesh.quad(pt(s0,y0,depth),pt(s1,y0,depth),pt(s1,y1,depth),pt(s0,y1,depth),normal,uv)
        face(shell,0,length,base-2,base+.10)
        if length<26:
            # 短端白色实体墙与少量狭窗，位置按照片概括。
            face(shell,0,length,base+.10,base+height)
            if length>12:
                for i in range(3):
                    left=length*(i+1)/4-.28
                    face(glass,left,left+.56,base+3.5,base+height-2,-.025)
            continue
        # 玻璃位于外框内侧；白色层间梁、立柱为可投影的真实三维构件。
        face(glass,0,length,base+.12,base+height-.15,.17,
             ((0,0),(length/4,0),(length/4,7),(0,7)))
        for floor in range(8):
            y=base+height*floor/7
            beam(frame,pt(0,y,-.05),pt(length,y,-.05),.17 if floor in (0,7) else .12)
        bays=max(1,round(length/3.2))
        for i in range(bays+1):
            s=length*i/bays
            beam(frame,pt(s,base,-.05),pt(s,base+height,-.05),.10)
        # 窗间细横框用较细梁呈现，按七层重复，具体模数非实测。
        for floor in range(7):
            for fraction in (.34,.68):
                y=base+height*(floor+fraction)/7
                beam(frame,pt(0,y,.02),pt(length,y,.02),.035)
        if edge_index==0:
            # 朝庭院的两层入口有浅色外框；具体门宽、位置由照片估计。
            s0,s1=max(0,length-13),max(0,length-4)
            for s in (s0,s1):
                beam(frame,pt(s,base,-.48),pt(s,base+7.1,-.48),.32)
            beam(frame,pt(s0,base+7.1,-.48),pt(s1,base+7.1,-.48),.32)
            beam(frame,pt(s0,base+3.5,-.48),pt(s1,base+3.5,-.48),.12)
            # 双扇门和入口雨棚仅依据航拍推定，凸出幕墙以便近处辨识。
            left,right=s0+.8,s1-.8
            face(glass,left,right,base+.04,base+3.05,-.30)
            for s in (left,(left+right)/2,right):
                beam(frame,pt(s,base+.04,-.37),pt(s,base+3.05,-.37),.07)
            beam(frame,pt(left,base+3.05,-.37),pt(right,base+3.05,-.37),.08)
            shell.quad(pt(s0,base+3.48,-.25),pt(s1,base+3.48,-.25),
                       pt(s1,base+3.48,-1.30),pt(s0,base+3.48,-1.30),
                       (0,1,0),((0,0),(1,0),(1,1),(0,1)))
            beam(frame,pt(s0,base+3.48,-1.30),pt(s1,base+3.48,-1.30),.10)


def south_l4_building(glass,frame,garden,equipment,roof,geom,base,height):
    """邻近 L4 的深色横向立面、屋顶绿化和设备按 2022 年航拍概括。"""
    outline=orient(geom.simplify(.1,preserve_topology=True),sign=1.0)
    roof.horizontal(outline,base+height)
    coords=list(outline.exterior.coords)
    for a,b in zip(coords[:-1],coords[1:]):
        dx,dz=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dz)
        if length<.1: continue
        tx,tz=dx/length,dz/length
        nx,nz=tz,-tx
        def pt(s,y,depth=0):
            return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
        glass.quad(pt(0,base-1,.08),pt(length,base-1,.08),
                   pt(length,base+height,.08),pt(0,base+height,.08),(nx,0,nz),
                   ((0,0),(length/4,0),(length/4,4),(0,4)))
        for floor in range(5):
            y=base+height*floor/4
            beam(frame,pt(0,y,-.08),pt(length,y,-.08),.16)
        for i in range(1,max(2,round(length/5.0))):
            s=length*i/max(2,round(length/5.0))
            beam(frame,pt(s,base,-.04),pt(s,base+height,-.04),.055)
        # 每层的细横线体现深色水平窗带，并保持与屋顶边缘分离。
        for floor in range(4):
            for fraction in (.30,.60,.83):
                y=base+height*(floor+fraction)/4
                beam(frame,pt(0,y,.005),pt(length,y,.005),.035)
    inner=outline.buffer(-1.2)
    if not inner.is_empty:
        garden.horizontal(inner,base+height+.04)
    # 顶部三个方形设备在航拍中可见，尺寸与功能没有公开测量资料。
    for x in (194,212,230):
        pad=box(x-2.1,570.2,x+2.1,574.0).intersection(inner)
        if not pad.is_empty:
            extrude(equipment,equipment,pad,base+height,base+height+1.9)


def zhigong_building(shell,glass,dark,perforated,louvers,rails,steps,sign,roof,geom,base,height):
    """致工楼的白色格窗、架空门廊、实验大厅和台阶，尺寸按公开照片近似。"""
    body=clean(geom.intersection(box(219.4,650,253,679)))
    corner=clean(geom.intersection(box(205.5,650,219.4,679)))
    porch=clean(geom.intersection(box(183.5,650,205.5,679)))
    # 楼体地图轮廓覆盖台阶附近；保留一个估计的开敞楼梯缺口。
    hall=clean(geom.intersection(box(169,679,230,746)).difference(box(169,679,181,688)))
    uv4=((0,0),(1,0),(1,1),(0,1))

    def grid_walls(part,bottom,floors):
        roof.horizontal(part,base+height)
        for poly in polygons(part):
            coords=list(orient(poly,sign=1.0).exterior.coords)
            for a,b in zip(coords[:-1],coords[1:]):
                dx,dz=b[0]-a[0],b[1]-a[1]
                length=math.hypot(dx,dz)
                if length<.2:
                    continue
                tx,tz=dx/length,dz/length
                nx,nz=tz,-tx
                def p(s,y,depth=0):
                    return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
                def face(mesh,l,r,lo,hi,depth=0):
                    if r-l>.02 and hi-lo>.02:
                        mesh.quad(p(l,lo,depth),p(r,lo,depth),p(r,hi,depth),p(l,hi,depth),
                                  (nx,0,nz),uv4)
                bays=max(1,round(length/3.6))
                for floor in range(floors):
                    y0=bottom+floor*3.3
                    y1=min(base+height,y0+3.3)
                    for i in range(bays):
                        s0,s1=length*i/bays,length*(i+1)/bays
                        l,r=s0+.23*(s1-s0),s1-.12*(s1-s0)
                        low,high=y0+1.08,y1-.35
                        face(shell,s0,s1,y0,low)
                        face(shell,s0,s1,high,y1)
                        face(shell,s0,l,low,high)
                        face(shell,r,s1,low,high)
                        # 窗洞及侧壁为实体几何；设备格栅的位置按照片概括。
                        shell.quad(p(l,low),p(l,low,.48),p(l,high,.48),p(l,high),
                                   (tx,0,tz),uv4)
                        shell.quad(p(r,low,.48),p(r,low),p(r,high),p(r,high,.48),
                                   (-tx,0,-tz),uv4)
                        shell.quad(p(l,high),p(r,high),p(r,high,.48),p(l,high,.48),
                                   (0,-1,0),uv4)
                        face(glass,l,r,low,high,.48)
                        face(perforated if (i+floor)%3==0 else louvers,
                             l,r,y0+.20,y0+.83,-.015)

    # 长翼一层为玻璃，临门廊部分下部有深灰色横带。
    extrude(glass,roof,body,base-1,base+3.3,cap=False)
    grid_walls(body,base+3.3,4)
    for floor in range(3):
        y=base+3.3*floor
        extrude(glass,roof,corner,y-1 if floor==0 else y,y+2.2,cap=False)
        extrude(dark,roof,corner,y+2.2,y+3.3,cap=False)
    grid_walls(corner,base+9.9,2)
    # 入口角部为多层玻璃窗，加入细立框，避免近看只剩一整块暗色平面。
    for x in np.arange(206.1,252,1.8):
        z=659.6-.6*(x-184.3)/67.5-.03
        floors=3 if x<219.4 else 1
        for floor in range(floors):
            y=base+3.3*floor
            beam(rails,(x,y,z),(x,y+2.2,z),.035)
    for z in np.arange(660.2,678,1.8):
        for floor in range(3):
            y=base+3.3*floor
            beam(rails,(205.46,y,z),(205.46,y+2.2,z),.035)

    # 开敞门廊：上部为厚框、纵向格栅和内退的浅色墙，下部由四根白柱支撑。
    extrude(shell,roof,porch,base+9.9,base+10.3)
    extrude(shell,roof,porch,base+15.9,base+height)
    # 照片中悬挑体的长侧仍延续两层格窗，而不是一整片无窗白墙。
    north_band=clean(porch.intersection(box(183,658,206,660.3)))
    if not north_band.is_empty:
        grid_walls(north_band,base+9.9,2)
    south_band=clean(porch.intersection(box(183,677.4,206,680)))
    if not south_band.is_empty:
        extrude(shell,roof,south_band,base+10.3,base+15.9)
    for z in (660.4,665.0,669.6,674.2,677.5):
        beam(shell,(184.8,base-.15,z),(184.8,base+9.9,z),.38)
    for z in np.arange(660.9,668.3,1.10):
        beam(shell,(184.5,base+10.5,z),(184.5,base+15.5,z),.12)
    # 后墙和小窗为架空空间提供可见深度；精确窗位尚无图纸。
    x=205.35
    dark.quad((x,base+10.5,660.5),(x,base+10.5,677.2),
              (x,base+15.5,677.2),(x,base+15.5,660.5),(-1,0,0),uv4)
    for z in (669.8,673.8):
        glass.quad((x-.025,base+11.6,z),(x-.025,base+11.6,z+1.8),
                   (x-.025,base+13.6,z+1.8),(x-.025,base+13.6,z),(-1,0,0),uv4)
    platform=clean(porch.intersection(box(191.5,660.5,205.5,677.2)))
    roof.horizontal(platform,base+3.35)
    for z in np.arange(661,677.1,1.6):
        beam(rails,(191.6,base+3.35,z),(191.6,base+4.45,z),.028)
    beam(rails,(191.6,base+4.45,661),(191.6,base+4.45,677),.035)
    # 标志与中文由本机字体重绘，图案和位置只概括照片。
    sign.quad((205.31,base+3.9,661.2),(205.31,base+3.9,675.0),
              (205.31,base+5.6,675.0),(205.31,base+5.6,661.2),(-1,0,0),uv4)

    # 实验大厅外侧以白墙、灰色基座和少量狭窗区分于教学翼。
    extrude(dark,roof,hall,base-1,base+3.3,cap=False)
    extrude(shell,roof,hall,base+3.3,base+height)
    for i in range(5):
        z=704.5+(.45 if i%2 else 0)
        y=base+5.0+i*2.15
        x=170.5+(z-678.9)*(.5/48.7)-.03
        glass.quad((x,y,z),(x,y,z+1.9),(x,y+.62,z+1.9),(x,y+.62,z),(-1,0,0),uv4)
    # 从门廊西南侧下到路面的台阶，级数、位置及高度为近似。
    for i in range(20):
        z1=688-i*.65
        tread=box(170.2,z1-.65,180.5,z1)
        extrude(steps,steps,tread,base-.12,base+(i+1)*.17)
    landing=box(178.5,674.5,193,677.5)
    extrude(steps,steps,landing,base-.12,base+3.4)
    for x in (170.6,180.1):
        beam(rails,(x,base+1.05,688),(x,base+4.45,675),.04)
        for i in range(0,21,4):
            z=688-i*.65
            y=base+i*.17
            beam(rails,(x,y,z),(x,y+1.05,z),.035)
    # 卫星影像可辨大厅屋面纵向格架及长翼屋顶设备，位置按图估计。
    for x in (197,200.5,204,207.5):
        beam(shell,(x,base+height+.20,683),(x,base+height+.20,722),.10)
    for z in np.arange(683,723,6.5):
        beam(shell,(197,base+height+.20,z),(207.5,base+height+.20,z),.10)
    for x in (226,234,242):
        pad=clean(box(x-1.1,665,x+1.1,667.5).intersection(body))
        if not pad.is_empty:
            extrude(dark,roof,pad,base+height,base+height+1.1)


def zhili_l1_low_building(shell,glass,frame,sign,roof,geom,base,height):
    """L1 低层教学体量的水平窗带与入口凹廊，构件位置参照照片近似。"""
    outline=orient(geom.simplify(.1,preserve_topology=True),sign=1.0)
    roof.horizontal(outline,base+height)
    coords=list(outline.exterior.coords)
    for a,b in zip(coords[:-1],coords[1:]):
        dx,dz=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dz)
        if length<.1:
            continue
        tx,tz=dx/length,dz/length
        nx,nz=tz,-tx
        def p(s,y,depth=0):
            return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
        # 入口面在朝草坪一侧，具体门洞位置按照片和轮廓估计。
        entry=length>30 and LineString((a,b)).distance(Point(279.5,527.5))<2
        left,right=length*.46,length*.72
        def face(mesh,l,r,lo,hi,depth=0):
            pieces=[(l,r,lo,hi)]
            if entry:
                clipped=[]
                for x0,x1,y0,y1 in pieces:
                    il,ir=max(x0,left),min(x1,right)
                    ib,it=max(y0,base-1),min(y1,base+4.5)
                    if il>=ir or ib>=it:
                        clipped.append((x0,x1,y0,y1))
                    else:
                        clipped.extend(((x0,il,y0,y1),(ir,x1,y0,y1),
                                        (il,ir,y0,ib),(il,ir,it,y1)))
                pieces=clipped
            for x0,x1,y0,y1 in pieces:
                if x1-x0>.01 and y1-y0>.01:
                    mesh.quad(p(x0,y0,depth),p(x1,y0,depth),p(x1,y1,depth),p(x0,y1,depth),
                              (nx,0,nz),((0,0),(1,0),(1,1),(0,1)))
        face(shell,0,length,base-2,base)
        for floor in range(7):
            y0=base+height*floor/7
            y1=base+height*(floor+1)/7
            if entry and floor<2:
                # 入口照片中门洞上方和两侧为连续浅色墙面。
                face(shell,0,length,y0,y1)
                continue
            sill,head=y0+.55,y1-.70
            face(shell,0,length,y0,sill)
            face(glass,0,length,sill,head,.12)
            face(shell,0,length,head,y1)
            bays=max(1,round(length/3.1))
            for i in range(bays+1):
                s=length*i/bays
                if entry and left-.1<s<right+.1 and y0<base+4.5:
                    continue
                beam(frame,p(s,sill,-.04),p(s,head,-.04),.06)
        if entry:
            # 门廊侧壁与顶部真实后退，不用整面贴图伪造入口。
            uv=((0,0),(1,0),(1,1),(0,1))
            floor_corners=(p(left,base,0),p(right,base,0),p(right,base,3.0),p(left,base,3.0))
            roof.horizontal(Polygon([(q[0],q[2]) for q in floor_corners]),base+.04)
            shell.quad(p(left,base,0),p(left,base,3.0),p(left,base+4.5,3.0),p(left,base+4.5,0),
                       (tx,0,tz),uv)
            shell.quad(p(right,base,3.0),p(right,base,0),p(right,base+4.5,0),p(right,base+4.5,3.0),
                       (-tx,0,-tz),uv)
            shell.quad(p(left,base+4.5,0),p(right,base+4.5,0),p(right,base+4.5,3.0),
                       p(left,base+4.5,3.0),(0,-1,0),uv)
            glass.quad(p(left,base,3.0),p(right,base,3.0),p(right,base+4.0,3.0),
                       p(left,base+4.0,3.0),(nx,0,nz),uv)
            for s in (left,(left+right)/2,right):
                beam(frame,p(s,base,2.9),p(s,base+4,2.9),.07)
            # 实拍门廊顶部可见两个圆环灯；直径和间距按图估计。
            for depth in (.8,2.15):
                for i in range(20):
                    angles=(math.tau*i/20,math.tau*(i+1)/20)
                    ends=[p((left+right)/2+.58*math.cos(t),base+4.42,depth+.58*math.sin(t)) for t in angles]
                    beam(shell,ends[0],ends[1],.035)
            sign.quad(p(max(.2,left-5),base+4.7,-.04),p(left-.6,base+4.7,-.04),
                      p(left-.6,base+5.8,-.04),p(max(.2,left-5),base+5.8,-.04),
                      (nx,0,nz),((1,0),(0,0),(0,1),(1,1)))


def south_grid_facade(shell, glass, side_wall, frame, podium, roof, geom, base, height,
                      walkway_id, podium_height, upper_levels, high_walkways=()):
    """致真楼和致理楼 L3 按照片区分白色格构长立面、深灰短侧与低层。"""
    # OSM 的一条长边有共线分段，合并后每面立面沿完整边长均匀排窗。
    outline = orient(geom.simplify(.1,preserve_topology=True),sign=1.0)
    coords = list(outline.exterior.coords)
    walkway = LineString(way_points(ways[walkway_id])) if walkway_id else LineString()
    podium_top, roof_base = base+podium_height, base+height-1.6
    level_height = (roof_base-podium_top)/upper_levels
    roof.horizontal(outline,podium_top)
    roof.horizontal(outline,roof_base)
    uv4=((0,0),(1,0),(1,1),(0,1))

    for a,b in zip(coords[:-1],coords[1:]):
        dx,dz = b[0]-a[0],b[1]-a[1]
        length = math.hypot(dx,dz)
        if length < .1:
            continue
        tx,tz = dx/length,dz/length
        nx,nz = tz,-tx
        normal = (nx,0,nz)
        edge = LineString((a,b))
        high_portals=[]
        for path_id,floor in high_walkways:
            path=LineString(way_points(ways[path_id]))
            endpoints=(Point(path.coords[0]),Point(path.coords[-1]))
            endpoint=min(endpoints,key=edge.distance)
            if edge.distance(endpoint)>.15:
                continue
            wx,wz=path.coords[-1][0]-path.coords[0][0],path.coords[-1][1]-path.coords[0][1]
            wl=math.hypot(wx,wz)
            crossing_sine=max(.2,abs(tx*wz/wl-tz*wx/wl))
            center=edge.project(endpoint)
            half=1.55/crossing_sine+.15
            y=base+(floor-1)*3.3
            high_portals.append((max(0,center-half),min(length,center+half),y-.05,y+2.9))
        def point(s,y,depth=0):
            return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
        def face(mesh,left,right,bottom,top,depth=0,repeat=False):
            # 楼间高层连桥穿过短侧墙；按桥面高度切出示意通道，避免被墙或玻璃封死。
            pieces=[(left,right,bottom,top)]
            for pl,pr,pb,pt in high_portals:
                clipped=[]
                for l,r,lo,hi in pieces:
                    il,ir,ib,it=max(l,pl),min(r,pr),max(lo,pb),min(hi,pt)
                    if il>=ir or ib>=it:
                        clipped.append((l,r,lo,hi))
                    else:
                        clipped.extend(((l,il,lo,hi),(ir,r,lo,hi),(il,ir,lo,ib),(il,ir,it,hi)))
                pieces=clipped
            for l,r,lo,hi in pieces:
                if r-l < .01 or hi-lo < .01:
                    continue
                uv = (((l/5,(lo-base)/3.2),(r/5,(lo-base)/3.2),
                       (r/5,(hi-base)/3.2),(l/5,(hi-base)/3.2)) if repeat else uv4)
                mesh.quad(point(l,lo,depth),point(r,lo,depth),
                          point(r,hi,depth),point(l,hi,depth),normal,uv)

        # 地图上高架人行线穿过裙房一角；在穿越的两面墙上留出口，
        # 门洞宽度与高度按通行净空估计，并不表示取得了建筑平面图。
        portal = None
        crossing = edge.intersection(walkway)
        if crossing.geom_type == 'Point' and not crossing.is_empty:
            along = walkway.project(crossing)
            p0,p1 = walkway.interpolate(max(0,along-.4)),walkway.interpolate(min(walkway.length,along+.4))
            wx,wz = p1.x-p0.x,p1.y-p0.y
            wl = max(math.hypot(wx,wz),.001)
            crossing_sine = max(.16,abs(tx*wz/wl-tz*wx/wl))
            half = min(9.0,2.25/crossing_sine+.25)
            center = edge.project(crossing)
            deck_y = surface_height(crossing.x,crossing.y)+4.2
            portal = (max(0,center-half),min(length,center+half),deck_y-.1,deck_y+3.1)
        if portal:
            left,right,bottom,top = portal
            face(podium,0,left,base-2,podium_top,repeat=True)
            face(podium,right,length,base-2,podium_top,repeat=True)
            face(podium,left,right,base-2,bottom,repeat=True)
            face(podium,left,right,top,podium_top,repeat=True)
        else:
            face(podium,0,length,base-2,podium_top,repeat=True)

        if length > 40:
            # 低层的水平白框和斜撑参考工务署照片，遇到连廊开口则让出通道。
            levels = [base+.15] + list(np.arange(base+3.15,podium_top-.3,3.0)) + [podium_top-.12]
            for y in levels:
                intervals = [(0,length)]
                if portal and portal[2] < y < portal[3]:
                    intervals = [(0,portal[0]),(portal[1],length)]
                for left,right in intervals:
                    if right-left>.2:
                        beam(frame,point(left,y,-.14),point(right,y,-.14),.23)
            for i in range(1,round(length/6)):
                s = length*i/round(length/6)
                if portal and portal[0]-.2<s<portal[1]+.2:
                    continue
                beam(frame,point(s,base+.15,-.12),point(s+.75,podium_top-.15,-.12),.17)
            columns = max(1,round(length/4.0))
            for floor in range(upper_levels):
                y0 = podium_top+floor*level_height
                y1 = y0+level_height
                for col in range(columns):
                    s0,s1 = length*col/columns,length*(col+1)/columns
                    left,right = s0+(s1-s0)*.19,s1-(s1-s0)*.19
                    bottom,top = y0+.75,y1-.48
                    face(shell,s0,s1,y0,bottom)
                    face(shell,s0,s1,top,y1)
                    face(shell,s0,left,bottom,top)
                    face(shell,right,s1,bottom,top)
                    # 窗口是真正内凹的平面，四周有窗侧壁，阳光和阴影可显示深度。
                    shell.quad(point(left,bottom),point(left,bottom,.36),point(left,top,.36),point(left,top),
                               (tx,0,tz),uv4)
                    shell.quad(point(right,bottom,.36),point(right,bottom),point(right,top),point(right,top,.36),
                               (-tx,0,-tz),uv4)
                    shell.quad(point(left,top),point(right,top),point(right,top,.36),point(left,top,.36),
                               (0,-1,0),uv4)
                    shell.quad(point(left,bottom,.36),point(right,bottom,.36),point(right,bottom),point(left,bottom),
                               (0,1,0),uv4)
                    face(glass,left,right,bottom,top,.36)
            for i in range(columns+1):
                s = length*i/columns
                beam(frame,point(s,podium_top,-.12),point(s,roof_base,-.12),.095)
            for floor in range(upper_levels+1):
                y = podium_top+floor*level_height
                beam(frame,point(0,y,-.12),point(length,y,-.12),.115)
        else:
            # 致理楼靠演播厅的端墙是白色塔身，上部连接 OSM 记录的悬挑体；
            # 其余短侧参照照片使用深灰墙面和一列小窗。
            is_white_core = (walkway_id == '1272661381'
                             and edge.intersection(way_polygon(ways['776358623']).boundary).length > 10)
            if is_white_core:
                columns=5
                for floor in range(upper_levels):
                    y0,y1=podium_top+floor*level_height,podium_top+(floor+1)*level_height
                    for col in range(columns):
                        s0,s1=length*col/columns,length*(col+1)/columns
                        # 小窗数量和宽度是对照片的概括，并非逐窗测量。
                        if (floor*7+col*3)%11 in (0,1):
                            face(shell,s0,s1,y0,y1)
                            continue
                        width=.85+.17*((floor+col)%3)
                        left,right=(s0+s1-width)*.5,(s0+s1+width)*.5
                        bottom,top=y0+.85,min(y1-.35,y0+2.05)
                        face(shell,s0,s1,y0,bottom)
                        face(shell,s0,s1,top,y1)
                        face(shell,s0,left,bottom,top)
                        face(shell,right,s1,bottom,top)
                        face(glass,left,right,bottom,top,.22)
                continue
            for floor in range(upper_levels):
                y0,y1 = podium_top+floor*level_height,podium_top+(floor+1)*level_height
                left,right = length*.46,length*.54
                bottom,top = y0+.76,y1-.45
                face(side_wall,0,length,y0,bottom)
                face(side_wall,0,length,top,y1)
                face(side_wall,0,left,bottom,top)
                face(side_wall,right,length,bottom,top)
                face(glass,left,right,bottom,top,.18)

        # 高出屋面的开放式白色屋顶框架，间距与高度为照片近似。
        beam(frame,point(0,base+height),point(length,base+height),.13)
        for i in range(max(1,round(length/6))+1):
            s = length*i/max(1,round(length/6))
            beam(frame,point(s,roof_base),point(s,base+height),.10)


def zhili_cantilever(shell, glass, roof_frame, geom, tower, base):
    """地图第 8—13 层悬挑体：白色大框、内退幕墙与可见底面。"""
    # OSM 未给精确标高；采用 L3 相同的 3.3 米估计层高，并与 L3 屋顶齐平。
    bottom,top=base+8*3.3,base+13*3.3
    outline=orient(geom,sign=1.0)
    coords=list(outline.exterior.coords)
    edges=list(zip(coords[:-1],coords[1:]))
    def separation(edge):
        a,b=edge
        return Point((a[0]+b[0])*.5,(a[1]+b[1])*.5).distance(tower)
    front_edge=max(edges,key=separation)
    # 航拍与近景顶部可见露天格构：屋面只封闭周边框，中央保留开口。
    roof_inset=outline.buffer(-.55,join_style='mitre')
    shell.horizontal(outline.difference(roof_inset) if not roof_inset.is_empty else outline,top)
    start=len(shell.vertices)
    shell.horizontal(outline,bottom)
    for i in range(start,len(shell.vertices),8):
        shell.vertices[i+4]=-1.0
    uv4=((0,0),(1,0),(1,1),(0,1))
    for a,b in edges:
        dx,dz=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dz)
        if length<.1:
            continue
        tx,tz=dx/length,dz/length
        nx,nz=tz,-tx
        normal=(nx,0,nz)
        edge=LineString((a,b))
        def point(s,y,depth=0):
            return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
        def face(mesh,left,right,low,high,depth=0,uv=uv4):
            if right-left>.01 and high-low>.01:
                mesh.quad(point(left,low,depth),point(right,low,depth),
                          point(right,high,depth),point(left,high,depth),normal,uv)
        # 与 L3 主体共用的边不重复绘墙；其余边作为白色悬挑体外壳。
        if edge.intersection(tower.boundary).length>10:
            continue
        if (a,b)!=front_edge:
            face(shell,0,length,bottom,top)
        else:
            left,right=length*.12,length*.88
            low,high=bottom+6.0,top-3.4
            slit_low,slit_high=bottom+1.6,bottom+3.1
            face(shell,0,length,bottom,slit_low)
            face(shell,0,length,slit_high,low)
            face(shell,0,length,high,top-1.6)
            # 顶框下方留出窄长的透空部位，与照片中的屋顶桁架相呼应。
            face(shell,0,length,top-.26,top)
            face(shell,0,1.2,top-1.6,top-.26)
            face(shell,length-1.2,length,top-1.6,top-.26)
            for y0,y1 in ((slit_low,slit_high),(low,high)):
                face(shell,0,left,y0,y1)
                face(shell,right,length,y0,y1)
                # 洞口四周有真实侧壁，玻璃退至白色框架内部。
                depth=5.0
                shell.quad(point(left,y0),point(left,y0,depth),point(left,y1,depth),point(left,y1),
                           (tx,0,tz),uv4)
                shell.quad(point(right,y0,depth),point(right,y0),point(right,y1),point(right,y1,depth),
                           (-tx,0,-tz),uv4)
                shell.quad(point(left,y1),point(right,y1),point(right,y1,depth),point(left,y1,depth),
                           (0,-1,0),uv4)
                shell.quad(point(left,y0,depth),point(right,y0,depth),point(right,y0),point(left,y0),
                           (0,1,0),uv4)
                u,v=(right-left)/3.2,(y1-y0)/2.0
                face(glass,left,right,y0,y1,depth,((0,0),(u,0),(u,v),(0,v)))
        # 顶部横梁与两根斜撑概括照片中的露天格架，间距未实测。
        beam(roof_frame,point(0,top),point(length,top),.12)
        if (a,b)==front_edge:
            for t in (.22,.78):
                beam(roof_frame,point(length*t,top-.12),point(length*(t+.08),top-.20,5.0),.07)


def huicui_building(shell,glass,upper,stage,frames,skylights,roof,geom,base,height):
    """汇萃楼：按规划屋顶平面重建三层底座、L 形八层部位及舞台屋面。"""
    low_top=base+15.0  # 三层剧场及大厅层高不同，15 米为航拍比例估计。
    roof.horizontal(geom,low_top)
    # 入口门厅的浅色地面与外侧铺装相接，避免从门口看见程序化草地。
    roof.horizontal(huicui_rect(0,0,1,.07).intersection(geom),base+.16)
    uv4=((0,0),(1,0),(1,1),(0,1))
    for poly in polygons(geom):
        coords=list(orient(poly,sign=1.0).exterior.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            dx,dz=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dz)
            if length<.1:
                continue
            tx,tz=dx/length,dz/length
            nx,nz=tz,-tx
            normal=(nx,0,nz)
            is_main=length>100 and nz<-.8
            def point(s,y,depth=0):
                return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
            def face(mesh,left,right,bottom,top,depth=0,uv=uv4):
                if right-left>.01 and top-bottom>.01:
                    mesh.quad(point(left,bottom,depth),point(right,bottom,depth),
                              point(right,top,depth),point(left,top,depth),normal,uv)
            def reveal(left,right,bottom,top,depth):
                shell.quad(point(left,bottom),point(left,bottom,depth),point(left,top,depth),point(left,top),
                           (tx,0,tz),uv4)
                shell.quad(point(right,bottom,depth),point(right,bottom),point(right,top),point(right,top,depth),
                           (-tx,0,-tz),uv4)
                shell.quad(point(left,top),point(right,top),point(right,top,depth),point(left,top,depth),
                           (0,-1,0),uv4)
                shell.quad(point(left,bottom,depth),point(right,bottom,depth),point(right,bottom),point(left,bottom),
                           (0,1,0),uv4)
            if is_main:
                # 照片与总平面显示北侧连续高柱、深色入口玻璃；两处主入口位置按图近似。
                columns=max(1,round(length/8.25))
                for col in range(columns):
                    s0,s1=length*col/columns,length*(col+1)/columns
                    left,right=s0+(s1-s0)*.18,s1-(s1-s0)*.18
                    bottom,top=base+.08,low_top-1.4
                    face(shell,s0,s1,base-2,bottom)
                    face(shell,s0,s1,top,low_top)
                    face(shell,s0,left,bottom,top)
                    face(shell,right,s1,bottom,top)
                    reveal(left,right,bottom,top,1.1)
                    doors=[length*u for u in (.29,.59) if left+1.25<length*u<right-1.25]
                    if doors:
                        center=doors[0]
                        # 地面至约 3.5 米保留通道，上方及两侧仍为深色幕墙。
                        face(glass,left,right,base+3.5,top,1.1,
                             ((0,0),((right-left)/3,0),((right-left)/3,3),(0,3)))
                        face(glass,left,center-1.2,bottom,base+3.5,1.1)
                        face(glass,center+1.2,right,bottom,base+3.5,1.1)
                        # 用整个柱间宽度的内退玻璃概括门厅，从斜侧观察也不会穿透到背后的草地。
                        face(glass,s0,s1,base-1,base+3.5,4.0)
                    else:
                        face(glass,left,right,bottom,top,1.1,
                             ((0,0),((right-left)/3,0),((right-left)/3,3.5),(0,3.5)))
                    # 白色竖向构架在窗前实际突出，形成与航拍相似的节奏。
                    for s in (left,right):
                        beam(frames,point(s,base+.1,-.12),point(s,low_top-.2,-.12),.12)
            else:
                # 其余外墙以白色实体为主，保留照片可见的少量深色窗洞。
                columns=max(1,round(length/7.5))
                for col in range(columns):
                    s0,s1=length*col/columns,length*(col+1)/columns
                    for floor in range(3):
                        y0,y1=base+floor*5.0,base+(floor+1)*5.0
                        if (col*7+floor*5)%4==0:
                            face(shell,s0,s1,y0,y1)
                            continue
                        width=min(1.6,(s1-s0)*.35)
                        left,right=(s0+s1-width)*.5,(s0+s1+width)*.5
                        bottom,top=y0+1.35,y0+3.15
                        face(shell,s0,s1,y0,bottom)
                        face(shell,s0,s1,top,y1)
                        face(shell,s0,left,bottom,top)
                        face(shell,right,s1,bottom,top)
                        reveal(left,right,bottom,top,.55)
                        face(glass,left,right,bottom,top,.55)
                face(shell,0,length,base-2,base)

    # 将总平面图中标记 8F 的南侧、东侧条形屋面映射到 OSM 外轮廓。
    high=clean(unary_union((huicui_rect(.134,.794,1,1),huicui_rect(.894,.074,1,.794))).intersection(geom))
    high_roof=base+height-1.5
    extrude(upper,roof,high,low_top,high_roof,unit_width=6,unit_height=4)
    for poly in polygons(high):
        coords=list(orient(poly,sign=1.0).exterior.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            length=math.dist(a,b)
            if length<.3:
                continue
            beam(frames,(a[0],base+height,a[1]),(b[0],base+height,b[1]),.11)
            for i in range(max(1,round(length/7))+1):
                t=i/max(1,round(length/7))
                x,z=a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t
                beam(frames,(x,high_roof,z),(x,base+height,z),.09)

    # 屋顶总平面图中的两块主台及设备区；各自高度未公布，按航拍近似。
    stages=((huicui_rect(.38,.39,.57,.79),23.5),(huicui_rect(.67,.56,.89,.79),21.0))
    for block,block_height in stages:
        block=clean(block.intersection(geom).difference(high).buffer(-.15))
        if block.is_empty:
            continue
        extrude(stage,roof,block,low_top,base+block_height)
        # 屋顶机房外缘可见白色水平格架；构造细节为近似。
        for poly in polygons(block):
            coords=list(orient(poly,sign=1.0).exterior.coords)
            for a,b in zip(coords[:-1],coords[1:]):
                if math.dist(a,b)>.5:
                    beam(frames,(a[0],base+block_height+.6,a[1]),
                         (b[0],base+block_height+.6,b[1]),.11)
                    beam(frames,(a[0],base+block_height,a[1]),
                         (a[0],base+block_height+.6,a[1]),.10)
    # 3F 屋面条形天窗的位置依据总平面图概括，不复制图纸或照片纹理。
    for v in (.10,.17,.28,.42,.54,.66,.75):
        light=huicui_rect(.22,v,.34,v+.011).intersection(geom).difference(high)
        for block,_ in stages:
            light=light.difference(block)
        if not light.is_empty:
            skylights.horizontal(light,low_top+.08)
    # 航拍中的西侧浅色楼体顶部有开放式白色梁架，位置与尺度按照片近似。
    cage=huicui_rect(.025,.035,.35,.76)
    ring=list(orient(cage,sign=1.0).exterior.coords)
    for a,b in zip(ring[:-1],ring[1:]):
        length=math.dist(a,b)
        beam(frames,(a[0],low_top+2.4,a[1]),(b[0],low_top+2.4,b[1]),.14)
        for i in range(max(1,round(length/9))+1):
            t=i/max(1,round(length/9))
            x,z=a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t
            beam(frames,(x,low_top,z),(x,low_top+2.4,z),.12)
    for u in (.13,.25):
        a,b=huicui_point(u,.035),huicui_point(u,.76)
        beam(frames,(a[0],low_top+2.4,a[1]),(b[0],low_top+2.4,b[1]),.12)


def huicui_neighbor_tower(shell,glass,fins,roof,geom,base,height,levels,bay_width,slender_fins=False):
    """吴玉章楼与汇语楼依据照片表达白色竖向立面；窗数和层高为估计。"""
    roof.horizontal(geom,base+height)
    level_height=height/levels
    uv4=((0,0),(1,0),(1,1),(0,1))
    for poly in polygons(geom):
        coords=list(orient(poly,sign=1.0).exterior.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            dx,dz=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dz)
            if length<.1:
                continue
            tx,tz=dx/length,dz/length
            nx,nz=tz,-tx
            normal=(nx,0,nz)
            def point(s,y,depth=0):
                return (a[0]+tx*s-nx*depth,y,a[1]+tz*s-nz*depth)
            def face(mesh,left,right,bottom,top,depth=0):
                if right-left>.01 and top-bottom>.01:
                    mesh.quad(point(left,bottom,depth),point(right,bottom,depth),
                              point(right,top,depth),point(left,top,depth),normal,uv4)
            face(shell,0,length,base-2,base)
            columns=max(1,round(length/bay_width))
            for floor in range(levels):
                y0,y1=base+floor*level_height,base+(floor+1)*level_height
                for col in range(columns):
                    s0,s1=length*col/columns,length*(col+1)/columns
                    side=.25 if slender_fins else .18
                    left,right=s0+(s1-s0)*side,s1-(s1-s0)*side
                    bottom,top=y0+.35,y1-.30
                    face(shell,s0,s1,y0,bottom)
                    face(shell,s0,s1,top,y1)
                    face(shell,s0,left,bottom,top)
                    face(shell,right,s1,bottom,top)
                    depth=.32
                    shell.quad(point(left,bottom),point(left,bottom,depth),point(left,top,depth),point(left,top),
                               (tx,0,tz),uv4)
                    shell.quad(point(right,bottom,depth),point(right,bottom),point(right,top),point(right,top,depth),
                               (-tx,0,-tz),uv4)
                    shell.quad(point(left,top),point(right,top),point(right,top,depth),point(left,top,depth),
                               (0,-1,0),uv4)
                    shell.quad(point(left,bottom,depth),point(right,bottom,depth),point(right,bottom),point(left,bottom),
                               (0,1,0),uv4)
                    face(glass,left,right,bottom,top,depth)
            if slender_fins:
                # 汇语楼照片可见密集外凸白色竖向遮阳片；间距按照片估计。
                for col in range(columns+1):
                    s=length*col/columns
                    fins.quad(point(s,base+.15,-.02),point(s,base+.15,-.42),
                              point(s,base+height-.2,-.42),point(s,base+height-.2,-.02),(tx,0,tz),uv4)


def huiyi_original_building(shell,glass,frames,strip,roof,geom,base,height):
    """老院馆依照片表现南翼悬挑、柱廊、条窗和圆弧形交通体；细部为近似。"""
    # 南翼朝向网球馆和运动场；OSM 只提供总体外轮廓。
    south=clean(geom.intersection(box(170,-637.5,230,-620)))
    cylinder=clean(Point(220,-635).buffer(5.0,resolution=24).intersection(geom))
    upper=clean(south.difference(cylinder))
    rear=clean(geom.difference(south.union(cylinder)))
    left_support=clean(upper.intersection(box(170,-640,187,-619)))
    # 中部上层跨在二层平台之上；上层和后部条窗沿用单独的条窗纹理。
    extrude(strip,roof,rear,base-2,base+height,7,3.8)
    extrude(strip,roof,upper,base+10.0,base+height,7,3.8)
    extrude(strip,roof,left_support,base-2,base+10.0,7,3.8)
    extrude(shell,roof,cylinder,base-2,base+height)
    # 照片中的平台和两根细柱承接悬挑体，底层短墙与栏杆均只按外观近似。
    platform=clean(south.difference(cylinder))
    roof.horizontal(platform,base+3.5)
    plinth=clean(platform.intersection(box(188,-625.5,215,-619)))
    extrude(shell,roof,plinth,base-1,base+3.5)
    for x in (195,207):
        if platform.buffer(-.3).covers(Point(x,-623)):
            beam(shell,(x,base+.1,-623),(x,base+10,-623),.32)
    for x in np.arange(188,215.1,3):
        beam(frames,(x,base+3.5,-620.9),(x,base+4.5,-620.9),.045)
    beam(frames,(188,base+4.5,-620.9),(215,base+4.5,-620.9),.045)
    # 左侧外挑百叶盒子参考原院馆照片，叶片间距不是实测。
    vent=box(177,-624,187,-619.7)
    extrude(shell,roof,vent,base+4.0,base+9.8)
    for x in np.arange(177.4,186.9,.7):
        beam(glass,(x,base+4.15,-619.68),(x,base+9.65,-619.68),.055)


def huiyi_teaching_lab(shell,glass,frames,roof,geom,base,height):
    """教学实验楼按核查总高及总平面退台重建，屋面尺寸和层高为近似。"""
    a,b,c,d=[np.asarray(p,dtype=float) for p in way_points(ways['1455957244'])[:4]]
    def point(u,v):
        return (1-v)*((1-u)*a+u*b)+v*((1-u)*d+u*c)
    def part(u0,v0,u1,v1):
        return clean(Polygon((point(u0,v0),point(u1,v0),point(u1,v1),point(u0,v1))).intersection(geom))
    # 三层基座上按规划图的 5F–9F 标记做退台，分段边界从图面估计。
    podium=11.7
    huicui_neighbor_tower(shell,glass,frames,roof,geom,base,podium,3,3.6)
    sections=((.36,0,1,.29,34.35,6),(.36,.29,1,.50,24.5,3),
              (.36,.50,1,.81,28.6,4),(.36,.81,1,1,32.5,5),
              (0,.81,.36,.95,34.35,6))
    for u0,v0,u1,v1,top,levels in sections:
        area=part(u0,v0,u1,v1)
        if not area.is_empty:
            huicui_neighbor_tower(shell,glass,frames,roof,area,base+podium,top-podium,levels,3.0)
    # 照片中最高白色体量有开放屋顶格架；顶部与官方高度 36.75 米对齐。
    cage=part(.36,0,1,.29)
    for poly in polygons(cage):
        coords=list(poly.exterior.coords)
        for p,q in zip(coords[:-1],coords[1:]):
            beam(frames,(p[0],base+height,p[1]),(q[0],base+height,q[1]),.16)
            length=math.dist(p,q)
            n=max(1,round(length/7))
            for i in range(n+1):
                t=i/n
                x,z=p[0]+(q[0]-p[0])*t,p[1]+(q[1]-p[1])*t
                beam(frames,(x,base+34.35,z),(x,base+height,z),.15)


def studio_facade(shell, glazing, sign_mesh, columns, roof, geom, base, height):
    """STUDIO 深灰盒体及其下方的架空平台；比例按现场照片估计。"""
    platform = base+4.2
    inset = geom.buffer(-1.1,join_style='mitre')
    upper_inset = geom.buffer(-.7,join_style='mitre')
    if inset.is_empty or upper_inset.is_empty:
        return
    # 透明幕墙退到柱列后方；演播厅黑盒高于连廊平台，底部有标志字。
    extrude(glazing,roof,inset,base-.2,platform,4,3.5)
    roof.horizontal(geom,platform)
    extrude(shell,roof,upper_inset,platform,base+6.6)
    extrude(shell,roof,geom,base+6.6,base+height)
    outline = orient(geom,sign=1.0)
    coords = list(outline.exterior.coords)
    for a,b in zip(coords[:-1],coords[1:]):
        dx,dz=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dz)
        if length<.1:
            continue
        tx,tz=dx/length,dz/length
        nx,nz=tz,-tx
        for i in range(max(1,round(length/7))):
            x,z=a[0]+dx*i/max(1,round(length/7))-nx*.35,a[1]+dz*i/max(1,round(length/7))-nz*.35
            beam(columns,(x,base-.1,z),(x,platform,z),.22)
        # 朝向庭院、紧邻高架连廊的短边可见 STUDIO 标识。
        if 19 < length < 22 and nx < -.4 and nz < -.4:
            left,right=length*.05,length*.95
            def at(s,y):
                return (a[0]+tx*s-nx*.66,y,a[1]+tz*s-nz*.66)
            sign_mesh.quad(at(left,platform+.12),at(right,platform+.12),
                           at(right,base+6.48),at(left,base+6.48),(nx,0,nz),
                           ((1,0),(0,0),(0,1),(1,1)))


def extrude_with_walkway_openings(mesh, roof, geom, base, top, path_ids, frame=None, side_wall=None,
                                 deck_y_override=None, cap=True, extra_paths=(), leveled_paths=(),
                                 unit_width_override=None, unit_height=3.3):
    """在地图人行线穿过墙体处预留可通行开口；开口大小与平台高度均为估计。"""
    if cap:
        roof.horizontal(geom,top)
    paths = [(LineString(way_points(ways[osm_id])),None) for osm_id in path_ids]
    paths += [(path,None) for path in extra_paths] + list(leveled_paths)
    for poly in polygons(geom):
        poly = orient(poly,sign=1.0)
        for ring in [poly.exterior]+list(poly.interiors):
            coords = list(ring.coords)
            for a,b in zip(coords[:-1],coords[1:]):
                dx,dz = b[0]-a[0],b[1]-a[1]
                length = math.hypot(dx,dz)
                if length < .05:
                    continue
                tx,tz = dx/length,dz/length
                nx,nz = tz,-tx
                current_mesh = side_wall if side_wall is not None and nx>.7 else mesh
                unit_width = (unit_width_override if unit_width_override is not None else
                              8 if current_mesh is side_wall else 4)
                edge = LineString((a,b))
                openings = []
                for path,path_y in paths:
                    cross = edge.intersection(path)
                    points = [cross] if cross.geom_type == 'Point' else list(cross.geoms) if cross.geom_type == 'MultiPoint' else []
                    for p in points:
                        if p.is_empty:
                            continue
                        d = path.project(p)
                        p0,p1 = path.interpolate(max(0,d-.4)),path.interpolate(min(path.length,d+.4))
                        wx,wz = p1.x-p0.x,p1.y-p0.y
                        wl = max(math.hypot(wx,wz),.001)
                        sine = max(.16,abs(tx*wz/wl-tz*wx/wl))
                        half = min(9.0,2.25/sine+.25)
                        center = edge.project(p)
                        deck_y = (path_y if path_y is not None else
                                  surface_height(p.x,p.y)+4.2 if deck_y_override is None else deck_y_override)
                        openings.append([max(0,center-half),min(length,center+half),deck_y-.1,deck_y+3.1])
                openings.sort()
                merged = []
                for opening in openings:
                    if merged and opening[0] <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1],opening[1])
                        merged[-1][2] = min(merged[-1][2],opening[2])
                        merged[-1][3] = max(merged[-1][3],opening[3])
                    else:
                        merged.append(opening)
                def point(s,y,offset=0):
                    return (a[0]+tx*s+nx*offset,y,a[1]+tz*s+nz*offset)
                def wall(left,right,bottom,upper):
                    if right-left<.01 or upper-bottom<.01:
                        return
                    current_mesh.quad(point(left,bottom),point(right,bottom),point(right,upper),point(left,upper),
                                      (nx,0,nz),((left/unit_width,(bottom-base)/unit_height),(right/unit_width,(bottom-base)/unit_height),
                                                 (right/unit_width,(upper-base)/unit_height),(left/unit_width,(upper-base)/unit_height)))
                at=0.0
                for left,right,bottom,upper in merged:
                    wall(at,left,base,top)
                    wall(left,right,base,min(top,bottom))
                    wall(left,right,max(base,upper),top)
                    at=right
                wall(at,length,base,top)
                if frame is not None and current_mesh is mesh and length>8:
                    # 幕墙的白色层间梁避开门洞，不遮挡连廊。
                    for y in np.arange(base+2,top+.1,3.3):
                        spans = [(0,length)]
                        for left,right,bottom,upper in merged:
                            if bottom < y < upper:
                                spans = [(l,min(r,left)) for l,r in spans if l < left] + [(max(l,right),r) for l,r in spans if r > right]
                        for left,right in spans:
                            if right-left>.2:
                                beam(frame,point(left,y,.10),point(right,y,.10),.10)


def huizi_building(wall,roof,roof_surface,shell,ground,geom,key,height):
    """用真实轮廓、退进底层和柱廊近似旧照片中的汇紫楼。"""
    top = huizi_roof_levels[key]
    base = top-height
    arcade_top = base+3.3
    if key == '230847099':
        # 连廊入口沿用地图线位；不能被重新生成的上部外墙封住。
        deck = surface_height(-126.4,-102.9)+4.35
        extrude_with_walkway_openings(wall,roof,geom,arcade_top,top,
                                     ('231782436',),deck_y_override=deck,cap=False)
    else:
        extrude(wall,roof,geom,arcade_top,top,cap=False)
    roof_surface.horizontal_mapped(geom,top,lambda x,z:(x/26.0,z/26.0))
    # 老照片能辨认到外柱与深处的暗色底层；哪些柱跨敞开尚无完整实测。
    inset = clean(geom.buffer(-2.2,join_style=2))
    if not inset.is_empty:
        extrude(ground,roof,inset,base-2,arcade_top)
    shell.horizontal(geom,arcade_top-.04)
    for poly in polygons(geom):
        points = list(orient(poly,sign=1.0).exterior.coords)
        for a,b in zip(points[:-1],points[1:]):
            dx,dz=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dz)
            midpoint=Point((a[0]+b[0])*.5,(a[1]+b[1])*.5)
            # 相邻 A/B/C/D 分段共有的边并非外墙，不在此处重复放柱。
            if length<3 or huizi_site.boundary.distance(midpoint)>.25:
                continue
            nx,nz=dz/length,-dx/length
            count=max(1,round(length/5.5))
            for i in range(count):
                t=(i+.5)/count
                x,z=a[0]+dx*t-nx*.45,a[1]+dz*t-nz*.45
                foot=min(base-.1,surface_height(x,z)-.1)
                beam(shell,(x,foot,z),(x,arcade_top,z),.26)


def huizi_roof_details(shell,canopy,roof,equipment):
    """屋顶收边和小型设备体量参考航拍，尺寸、位置为近似。"""
    for geom,key in ((huizi_parts['230847102'],'230847102'),(huizi_parts['230847100'],'230847100'),
                     (huizi_cd,'230847098')):
        top=huizi_roof_levels[key]
        parapet=clean(geom.boundary.buffer(.23,join_style=2).intersection(geom))
        extrude(shell,roof,parapet,top-.04,top+.65)
    # 外端的白色竖向楼梯体按航拍轮廓概括，非施工图。
    for x,z in ((-73.9,-104.6),(-17.7,-114.8),(-13.5,41.5),(35.6,32.6)):
        key='230847102' if huizi_a.distance(Point(x,z))<.5 else '230847099'
        top=huizi_roof_levels[key]
        corner=clean(Point(x,z).buffer(3.1,cap_style=3).intersection(huizi_site))
        if not corner.is_empty:
            # 微量外扩防止白色端部与主体外墙共面闪烁。
            corner=corner.buffer(.08,join_style=2)
            base=top-(13.2 if huizi_a.distance(Point(x,z))<.5 else 16.5)
            extrude(shell,roof,corner,base-1,top+2.5)
    # 可见的白色小机房与设备台；避免生成在三个院落之外的空气中。
    for x,z in ((-65,-101),(-9,-79),(7,-36),(-15,27),(27,27)):
        pad=affinity.rotate(box(x-2.1,z-1.4,x+2.1,z+1.4),-10,origin=(x,z))
        groups=[key for key,geom in huizi_parts.items() if geom.buffer(-.3).covers(pad)]
        if groups:
            top=huizi_roof_levels[groups[0]]
            extrude(shell,roof,pad,top+.05,top+1.7)
            for offset in (-.8,0,.8):
                beam(shell,(x+offset,top+1.7,z),(x+offset,top+2.3,z),.05)
    # 北侧中庭和中段中庭之间可见两组带深色格栅的大型设备；仅按航拍近似其位置与尺寸。
    top=huizi_roof_levels['230847098']
    for x,z in ((-36.4,-47.7),(-46.6,-45.1)):
        pad=affinity.rotate(box(x-2.25,z-1.45,x+2.25,z+1.45),-10,origin=(x,z))
        body=affinity.rotate(box(x-2.0,z-1.2,x+2.0,z+1.2),-10,origin=(x,z))
        if not huizi_site.buffer(-.1).covers(pad):
            continue
        extrude(shell,roof,pad,top+.02,top+.20)
        extrude(equipment,roof,body,top+.65,top+2.55,unit_width=4.0,unit_height=1.9)
        for px,pz in list(pad.exterior.coords)[:-1]:
            beam(shell,(px,top+.18,pz),(px,top+.70,pz),.10)
    # 屋面细管与小型通风帽为照片可见构件的简化，线位和数量并非测绘。
    for x,z in ((-59,-99),(-54,-99.8),(-49,-100.5)):
        vent=Point(x,z).buffer(.48,resolution=12)
        if huizi_site.buffer(-.2).covers(vent):
            extrude(shell,roof,vent,top+.05,top+.55)
    for points,key in ((((-48,-42.8),(-38,-44.7),(-28,-46.5)),'230847098'),
                       (((-59,-100),(-48,-102),(-38,-104)),'230847099')):
        top=huizi_roof_levels[key]
        route=LineString(points).intersection(huizi_parts[key].buffer(-.45))
        for part in lines(route):
            for a,b in zip(list(part.coords)[:-1],list(part.coords)[1:]):
                beam(shell,(a[0],top+.18,a[1]),(b[0],top+.18,b[1]),.07)
    # 航拍的北端院落可见蓝灰色格状覆盖物；用途与精确高度仍未核实。
    top=huizi_roof_levels['230847098']
    cover=affinity.scale(huizi_north_courtyard,xfact=.78,yfact=.74,origin='centroid')
    cover_y=top-7.2
    canopy.horizontal(cover,cover_y)
    aligned=affinity.rotate(cover,10,origin=cover.centroid)
    xmin,zmin,xmax,zmax=aligned.bounds
    for x in np.arange(xmin,xmax+1,2.6):
        line=affinity.rotate(LineString(((x,zmin),(x,zmax))).intersection(aligned),-10,origin=cover.centroid)
        for part in lines(line):
            a,b=part.coords[0],part.coords[-1]
            beam(shell,(a[0],cover_y+.04,a[1]),(b[0],cover_y+.04,b[1]),.035)
    for z in np.arange(zmin,zmax+1,2.6):
        line=affinity.rotate(LineString(((xmin,z),(xmax,z))).intersection(aligned),-10,origin=cover.centroid)
        for part in lines(line):
            a,b=part.coords[0],part.coords[-1]
            beam(shell,(a[0],cover_y+.04,a[1]),(b[0],cover_y+.04,b[1]),.035)


def huizi_entry(shell,paving):
    """旧照中的浅色入口平台和低踏步，宽度及高差均为估计。"""
    level=huizi_roof_levels['230847102']-13.2
    extrude(shell,paving,huizi_entry_strip(0,5),level-1.1,level+.03)
    for i in range(4):
        step=huizi_entry_strip(5+i*.8,5+(i+1)*.8)
        extrude(shell,paving,step,level-1.1,level-.18*(i+1)+.03)


def huiyuan_building(wall,roof,shell,ground,sign,motto,paving,planting,entry_roof,geom,base,height):
    """按实景照片和 OSM 人行线重建汇元楼底层、入口与连廊。"""
    arcade_top=base+3.3
    bridge_level=surface_height(-126.4,-102.9)+4.35
    extrude_with_walkway_openings(wall,roof,geom,arcade_top,base+height,
                                 ('1274485928',),deck_y_override=bridge_level)
    inset=clean(geom.buffer(-2.2,join_style=2))
    if not inset.is_empty:
        # 地面人行线进入退后的界面时同样留门洞。
        extrude_with_walkway_openings(ground,roof,inset,base-2,arcade_top,
                                     ('237893526',),deck_y_override=base+.05)
    shell.horizontal(geom,arcade_top-.04)
    for poly in polygons(geom):
        points=list(orient(poly,sign=1.0).exterior.coords)
        for a,b in zip(points[:-1],points[1:]):
            dx,dz=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dz)
            if length<3:
                continue
            nx,nz=dz/length,-dx/length
            count=max(1,round(length/6.0))
            for i in range(count):
                t=(i+.5)/count
                x,z=a[0]+dx*t-nx*.45,a[1]+dz*t-nz*.45
                if huiyuan_entry_path.distance(Point(x,z))<3.4:
                    continue
                foot=min(base-.1,surface_height(x,z)-.1)
                beam(shell,(x,foot,z),(x,arcade_top,z),.27)
    # 一层石质门廊：入口横梁、左侧立柱和带校名的右侧墙体分开建模。
    extrude(shell,roof,huiyuan_rect(-3.8,-3.15,0,6.0),base-1,base+5.1)
    extrude(shell,roof,huiyuan_rect(3.15,10.6,5.1,6.0),base-1,base+5.1)
    # 2023 年航拍显示入口门廊顶面为红褐色。
    extrude(shell,entry_roof,huiyuan_rect(-3.8,10.6,0,6.0),base+4.55,base+5.25)
    # 保留少量离地间隙，避免广场平整后与地形面发生闪烁。
    paving.horizontal(huiyuan_rect(-3.1,3.1,-2.8,7.2),base+.12)
    # 贴图仅为程序重绘的校名；四角顺序使其面向时光广场。
    def p(side,depth,y):
        x,z=huiyuan_point(side,depth)
        return (x,y,z)
    nx,nz=huiyuan_out
    sign.quad(p(3.35,6.02,base+1.05),p(10.35,6.02,base+1.05),
              p(10.35,6.02,base+4.45),p(3.35,6.02,base+4.45),
              (nx,0,nz),((0,0),(1,0),(1,1),(0,1)))
    # 西侧石刻朝向通道内部，具体位置和雕刻字形未经测量。
    rx,rz=huiyuan_right
    motto.quad(p(-3.13,.5,base+.55),p(-3.13,5.0,base+.55),
               p(-3.13,5.0,base+3.7),p(-3.13,.5,base+3.7),
               (rx,0,rz),((1,0),(0,0),(0,1),(1,1)))
    # 照片中校名墙前是低矮绿化池；宽度和植被均按外观概括。
    planter=huiyuan_rect(4.0,10.8,6.4,9.2)
    extrude(shell,planting,planter,base-.7,base+.60)
    # 屋面白色女儿墙与少量设备箱参照航拍，位置为估计。
    rim=clean(geom.boundary.buffer(.20,join_style=2).intersection(geom))
    extrude(shell,roof,rim,base+height-.04,base+height+.55)
    for x,z in ((-220,32),(-187,-99),(-198,-38)):
        pad=box(x-1.4,z-1.1,x+1.4,z+1.1)
        if geom.buffer(-.2).covers(pad):
            extrude(shell,roof,pad,base+height+.05,base+height+1.55)


def yanhui_building(roof,underside,steel,stone,floor,seats,seat_detail,stage,screen,steps,seams,geom,base,height):
    """在地图屋顶范围内近似重建演会中心的半开放厅、钢桁架和坐席。"""
    eave=base+height-.8
    peak=base+height+.6
    center=geom.centroid
    perimeter=list(orient(geom,sign=1.0).exterior.coords)
    # 航拍的低坡白色四面屋顶沿真实 OSM 边界保留不规则挑檐。
    for a,b in zip(perimeter[:-1],perimeter[1:]):
        pa,pb,pc=(a[0],eave,a[1]),(b[0],eave,b[1]),(center.x,peak,center.y)
        # 每个屋面方向共用法线，避免带锯齿挑檐的三角剖分产生扇形明暗条纹。
        midpoint=(np.asarray(a)+np.asarray(b))*.5-yanhui_center
        u,v=np.dot(midpoint,yanhui_u),np.dot(midpoint,yanhui_v)
        horizontal=(yanhui_u*np.sign(u) if abs(u)/32>abs(v)/28 else yanhui_v*np.sign(v))*.05
        normal=np.asarray((horizontal[0],1.0,horizontal[1])); normal/=np.linalg.norm(normal)
        roof.triangle(pa,pb,pc,normal,
                      ((a[0]/12,a[1]/12),(b[0]/12,b[1]/12),(center.x/12,center.y/12)))
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if length<.05:
            continue
        def edge(t,y):
            return (a[0]+(b[0]-a[0])*t,y,a[1]+(b[1]-a[1])*t)
        # 入口照片中的红褐色外围三角桁架；杆数、尺寸按照片概括。
        beam(steel,edge(0,eave-.15),edge(1,eave-.15),.075)
        beam(steel,edge(0,eave-1.75),edge(1,eave-1.75),.075)
        bays=max(1,round(length/4.6))
        for i in range(bays):
            beam(steel,edge(i/bays,eave-.15),edge((i+.5)/bays,eave-1.75),.06)
            beam(steel,edge((i+.5)/bays,eave-1.75),edge((i+1)/bays,eave-.15),.06)
    # 底面略深色，室内能看到与檐口一致的空间桁架。
    underside.horizontal(geom,eave-.08)
    def point(u,v,y):
        x,z=yanhui_point(u,v)
        return (x,y,z)
    # 屋面一圈细深色边线来自航拍，宽度和准确位置为估计。
    seam_y=eave+(peak-eave)*.20+.02
    for u in (-25,25):
        beam(seams,point(u,-22,seam_y),point(u,22,seam_y),.035)
    for v in (-22,22):
        beam(seams,point(-25,v,seam_y),point(25,v,seam_y),.035)
    for u in np.arange(-24,25,6):
        beam(steel,point(u,-24,eave-.38),point(u,24,eave-.38),.065)
        beam(steel,point(u,-24,eave-1.55),point(u,24,eave-1.55),.055)
        for v in np.arange(-24,24,4):
            beam(steel,point(u,v,eave-.38),point(u,v+2,eave-1.55),.045)
            beam(steel,point(u,v+2,eave-1.55),point(u,v+4,eave-.38),.045)
    for v in np.arange(-24,25,6):
        beam(steel,point(-25,v,eave-.50),point(25,v,eave-.50),.06)
    # 粗地形平台内设石墙与柱，外侧保持敞开；位置不是结构图坐标。
    floor.horizontal(geom.buffer(-1),base+.08)
    for u in (-24,24):
        for v in (-20,0,20):
            beam(stone,point(u,v,base-1),point(u,v,eave-1.4),.42)
    for u0,u1 in ((-25,-20),(20,25)):
        side=yanhui_rect(u0,-24,u1,-20)
        extrude(stone,floor,side,base-1,base+4.3)
    # 台前橙色座椅与阶梯式地坪按室内照片近似；坐席排数和台阶高差未经实测。
    row_depth=1.48
    row_front=-10.2
    for row in range(19):
        v0=row_front+row*row_depth
        y=base+.19+row*.13
        # 保留中央通道，避免后排地坪形成横挡入口的整面高墙。
        for u0,u1 in ((-19,-1.4),(1.4,19)):
            terrace=yanhui_rect(u0,v0,u1,v0+row_depth)
            extrude(stone,floor,terrace,base+.06,y)
        floor.horizontal(yanhui_rect(-1.4,v0,1.4,v0+row_depth),y+.01)
        for u in np.arange(-17.5,18,1.08):
            if abs(u)<1.05:
                continue
            seats.horizontal(yanhui_rect(u-.34,v0+.18,u+.34,v0+.75),y+.45)
            normal=(yanhui_v[0],0,yanhui_v[1])
            seats.quad(point(u-.34,v0+.75,y+.43),point(u+.34,v0+.75,y+.43),
                       point(u+.34,v0+.84,y+1.06),point(u-.34,v0+.84,y+1.06),
                       normal,((0,0),(1,0),(1,1),(0,1)))
            # 椅背的浅色塑料面与深色边缘分层，远处仍可辨认单个座椅。
            seat_detail.quad(point(u-.27,v0+.766,y+.50),point(u+.27,v0+.766,y+.50),
                             point(u+.27,v0+.845,y+.98),point(u-.27,v0+.845,y+.98),
                             normal,((0,0),(1,0),(1,1),(0,1)))
    # 舞台、暖色背景及幕布仅表达照片中的整体关系，不复制活动屏幕内容。
    platform=yanhui_rect(-17,-22,17,-12)
    extrude(stone,stage,platform,base-.3,base+.65)
    backdrop=yanhui_rect(-16,-22,16,-21.65)
    extrude(stage,stage,backdrop,base+.65,base+6.0)
    screen.quad(point(-7,-21.60,base+2.0),point(7,-21.60,base+2.0),
                point(7,-21.60,base+6.0),point(-7,-21.60,base+6.0),
                (yanhui_v[0],0,yanhui_v[1]),((0,0),(1,0),(1,1),(0,1)))
    # 后排与大屋顶边缘之间有近似阶梯通道，连接室外入口和坐席。
    rear_v=row_front+19*row_depth
    rear_y=base+.19+18*.13
    for i in range(15):
        v0=rear_v+(27-rear_v)*i/15
        v1=rear_v+(27-rear_v)*(i+1)/15
        y0=rear_y+(base+.08-rear_y)*i/15
        y1=rear_y+(base+.08-rear_y)*(i+1)/15
        extrude(stone,floor,yanhui_rect(-1.4,v0,1.4,v1),min(y0,y1)-.15,max(y0,y1))
    # 地图中的台阶线延至最近的门廊边界；宽度、级数及高差均为视觉估计。
    a=np.asarray(yanhui_step_line.coords[0]); b=np.asarray(yanhui_step_line.coords[-1])
    direction=(b-a)/np.linalg.norm(b-a)
    side=np.asarray((-direction[1],direction[0]))
    low=surface_height(*a)+.04
    high=base+.08
    count=max(12,int(math.ceil((high-low)/.17)))
    def edge(p,offset,y):
        return (p[0]+side[0]*offset,y,p[1]+side[1]*offset)
    levels=[]
    for i in range(count):
        p0=a+(b-a)*i/count
        p1=a+(b-a)*(i+1)/count
        samples=[p0+(p1-p0)*t+side*offset for t in (0,.25,.5,.75,1)
                 for offset in np.linspace(-3,3,7)]
        clearance=max(rendered_terrain_height(p[0],p[1]) for p in samples)+.10
        levels.append(max(low+(high-low)*(i+1)/count,clearance,levels[-1] if levels else low))
    for i in range(count):
        p0=a+(b-a)*i/count
        p1=a+(b-a)*(i+1)/count
        y0=low if i==0 else levels[i-1]
        y1=levels[i]
        steps.quad(edge(p0,-3,y1),edge(p1,-3,y1),edge(p1,3,y1),edge(p0,3,y1),
                   (0,1,0),((0,0),(1,0),(1,1),(0,1)))
        if i==0:
            left=p0-side*3; right=p0+side*3
            bottom_left=rendered_terrain_height(left[0],left[1])-.08
            bottom_right=rendered_terrain_height(right[0],right[1])-.08
        else:
            bottom_left=bottom_right=y0
        steps.quad(edge(p0,-3,bottom_left),edge(p0,3,bottom_right),edge(p0,3,y1),edge(p0,-3,y1),
                   (-direction[0],0,-direction[1]),((0,0),(1,0),(1,1),(0,1)))
        # 两侧石质边墙跟随地表，遮住实体踏步与粗地形间的空隙。
        for offset in (-3,3):
            q0,q1=p0+side*offset,p1+side*offset
            bottom0=min(y1,rendered_terrain_height(q0[0],q0[1])-.08)
            bottom1=min(y1,rendered_terrain_height(q1[0],q1[1])-.08)
            normal=(side[0]*np.sign(offset),0,side[1]*np.sign(offset))
            stone.quad(edge(p0,offset,bottom0),edge(p1,offset,bottom1),
                       edge(p1,offset,y1),edge(p0,offset,y1),normal,
                       ((0,0),(1,0),(1,1),(0,1)))


def make_yanhui_pines():
    """入口照片中高而窄的常绿树以程序网格概括，株数和位置未实测。"""
    trunks,crowns=MeshWriter(),MeshWriter()
    rng=random.Random(231885484)
    segments=10
    for index,u in enumerate((-23,-19,-15,-11,9,13,17,21,25)):
        x,z=yanhui_point(u,32.5+rng.uniform(-.5,.5))
        y=surface_height(x,z)
        height=15.0+rng.uniform(-1.2,2.0)
        beam(trunks,(x,y-.2,z),(x,y+height*.82,z),.16)
        for layer in range(10):
            # 照片中树冠有向外伸展的层状枝条，用相互交叠的疏密层概括。
            low=.18+layer*.075
            high=min(1.0,low+.19)
            radius=2.05*(1-layer*.075)*rng.uniform(.92,1.08)
            phase=rng.uniform(0,math.tau)
            for j in range(segments):
                a,b=math.tau*j/segments+phase,math.tau*(j+1)/segments+phase
                def p(h,r,angle):
                    rr=r*(1+.12*math.sin(3*angle+index*1.3+layer))
                    return (x+rr*math.cos(angle),y+h*height,z+rr*math.sin(angle))
                a0,b0=p(low,radius,a),p(low,radius,b)
                a1,b1=p(high,radius*.08,a),p(high,radius*.08,b)
                n=np.asarray((math.cos((a+b)*.5),.25,math.sin((a+b)*.5))); n/=np.linalg.norm(n)
                crowns.quad(a0,b0,b1,a1,n,((0,0),(1,0),(1,1),(0,1)))
    return {'trunks':trunks.save(ASSETS/'campus-yanhui-pine-trunks.mesh'),
            'crowns':crowns.save(ASSETS/'campus-yanhui-pine-crowns.mesh')}


def xiyuan_dorm(wall, frame, railing, roof, geom, altitude, height):
    """按照片表达阁区长翼、白色开放端廊与栏杆；构件位置和层高为估计。"""
    ends=clean(geom.intersection(box(-510,500,-483.5,660)))
    body=clean(geom.difference(ends))
    extrude(wall,roof,body,altitude-3,altitude+height,4,3.3)
    for poly in polygons(ends):
        minx,minz,maxx,maxz=poly.bounds
        roof.horizontal(poly,altitude+height)
        for x in (minx+.28,maxx-.28):
            for z in (minz+.28,maxz-.28):
                if poly.buffer(.2).covers(Point(x,z)):
                    beam(frame,(x,altitude-2,z),(x,altitude+height,z),.22)
        for floor in range(1,6):
            y=altitude+height*floor/5
            roof.horizontal(poly,y)
            for a,b in zip(list(poly.exterior.coords)[:-1],list(poly.exterior.coords)[1:]):
                beam(frame,(a[0],y,a[1]),(b[0],y,b[1]),.12)
                if floor<5 and math.dist(a,b)>1:
                    beam(railing,(a[0],y+1.0,a[1]),(b[0],y+1.0,b[1]),.025)
                    for t in np.linspace(0,1,max(2,round(math.dist(a,b)/1.5))):
                        p=np.asarray(a)*(1-t)+np.asarray(b)*t
                        beam(railing,(p[0],y,p[1]),(p[0],y+1.0,p[1]),.025)


def building_base_height(key,tags,geom):
    p=geom.representative_point()
    return road_building_bases.get(key,surface_height(p.x,p.y))


def make_dorm_models():
    """楼栋独立生成；L/W 索引统一接入材质加载和相机步行表面。"""
    import campus_qiaoyuan_dorms
    import campus_south_dorms
    result={}
    for name,module in (('qiaoyuan',campus_qiaoyuan_dorms),('south',campus_south_dorms)):
        info=module.make_dorms(buildings=buildings,height_for=height_for,base_height=building_base_height,
            mesh_type=MeshWriter,extrude=extrude,beam=beam,height_at=rendered_terrain_height,
            assets=ASSETS,data=DATA,entrance_paths={'way/629940894':qiaoyuan_tower_path})
        lines=['# L: building material layer; W: walkable entrance surface']
        for layer in info['layers']:
            row=['L',layer['id'],layer['material'],layer['mesh'],layer.get('texture','-'),*layer['color']]
            lines.append('\t'.join(str(value) for value in row))
        for walk in info.get('walking_meshes',()):
            lines.append('\t'.join(('W',walk['id'],walk['mesh'])))
        if name=='south':
            skywalks={building['id']:(building,walk) for building in info['buildings']
                      for walk in building.get('skywalks',())}
            for ident in ('xiazheng','qiuse','dongzhu'):
                building,walk=skywalks[ident]
                eye,target=walk['view_eye'],walk['view_target']
                row=['G',ident+'-skywalk',building['name']+'高架入口',*eye,*target,target[0],target[2]]
                lines.append('\t'.join(str(value) for value in row))
        (ASSETS/f'campus-{name}-dorms.tsv').write_text('\n'.join(lines)+'\n',encoding='utf8')
        (ASSETS/f'campus-{name}-dorms.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
        result[name]={'buildings':len(info['buildings']),'layers':len(info['layers']),
                      'vertices':sum(layer['vertices'] for layer in info['layers'])}
    return result


def make_south_courtyard(*, assets, terrain, height_at):
    return campus_south_spine.make_spine(
        config=SOUTH_SPINE, polygon=lambda key: way_polygon(ways[key.split('/')[1]]),
        line=lambda key: LineString(way_points(ways[key.split('/')[1]])),
        vehicle=vehicle_surface, height_at=height_at, mesh_type=MeshWriter,
        terrain=terrain, assets=assets)


def make_zhiyi_podium(shell, roof, gallery, geom, base, height):
    passage=LineString(way_points(ways[SOUTH_SPINE['building_opening_requests'][0]['existing_passage_osm'].split('/')[1]]))
    campus_south_spine.make_podium(shell,roof,gallery,geom=geom,base=base,height=height,
                                  config=SOUTH_SPINE,passage=passage,
                                  wall=extrude_with_walkway_openings,beam=beam)


def new_zhai_facade(wall, stone, glass, roof, paving, geom, altitude, height, arcade_edge):
    """新斋的深色基座、架空廊和浅色宿舍立面按照片分层。"""
    poly = orient(geom, sign=1.0)
    top = altitude + height
    roof.horizontal(poly, top)
    for a, b in zip(poly.exterior.coords[:-1], poly.exterior.coords[1:]):
        length = math.dist(a,b)
        tangent = (np.asarray(b)-a)/length
        outward = np.asarray((tangent[1],-tangent[0]))
        arcade = arcade_edge is not None and Point(a).distance(arcade_edge)<.05 and Point(b).distance(arcade_edge)<.05
        ground = [rendered_terrain_height(*(np.asarray(a)+s*tangent)) for s in np.linspace(0,length,12)]
        bottom = min(ground)-.55
        ceiling = altitude+4.4

        def position(s, y, recess=0):
            x,z = a+s*tangent-recess*outward
            return x,y,z

        def panel(mesh, left, right, low, high, recess=0, tile=(3.0,3.5)):
            mesh.quad(position(left,low,recess),position(right,low,recess),
                      position(right,high,recess),position(left,high,recess),
                      (outward[0],0,outward[1]),
                      ((0,0),((right-left)/tile[0],0),((right-left)/tile[0],(high-low)/tile[1]),
                       (0,(high-low)/tile[1])))

        # 长边有连续阳台，端部是较宽的浅色墙垛。短边保留疏窗。
        pier = min(5.5,length*.11) if length>30 else 1.2
        panel(roof,0,pier,ceiling,top)
        panel(wall,pier,length-pier,ceiling,top,tile=(12 if length>30 else 20,3.5))
        panel(roof,length-pier,length,ceiling,top)
        if not arcade:
            panel(stone,0,length,bottom,ceiling,tile=(3.2,2.0))
            continue

        # 校园一侧可见柱后玻璃和连续的檐下步行空间，柱距与退深为照片估计。
        depth = 2.5
        open_start,open_end = pier,length-pier
        panel(stone,0,open_start,bottom,ceiling)
        panel(stone,open_end,length,bottom,ceiling)
        panel(glass,open_start,open_end,bottom+.15,ceiling-.32,depth,tile=(2.7,3.6))
        panel(stone,open_start,open_end,ceiling-.32,ceiling,depth)
        soffit = (position(open_start,ceiling),position(open_start,ceiling,depth),
                  position(open_end,ceiling,depth),position(open_end,ceiling))
        stone.quad(*soffit,(0,-1,0),((0,0),(0,1),(length/3,1),(length/3,0)))
        for s,sign in ((open_start,-1),(open_end,1)):
            stone.quad(position(s,bottom),position(s,bottom,depth),
                       position(s,ceiling,depth),position(s,ceiling),
                       (tangent[0]*-sign,0,tangent[1]*-sign),((0,0),(1,0),(1,1),(0,1)))
        bays = max(1,round((open_end-open_start)/5.8))
        for s in np.linspace(open_start,open_end,bays+1)[1:-1]:
            x,_,z = position(s,0,.40)
            beam(stone,(x,rendered_terrain_height(x,z)-.12,z),(x,ceiling+.02,z),.42)
        # 首层外有连续硬质铺装，按楼侧道路边界裁切，避免铺到车行面。
        apron = 5.5
        footprint = Polygon([(x,z) for x,_,z in (
            position(0,0,-apron),position(length,0,-apron),
            position(length,0,0),position(open_end,0,0),position(open_end,0,depth),
            position(open_start,0,depth),position(open_start,0,0),position(0,0,0))])
        drape_on_terrain(paving,footprint.difference(vehicle_surface.buffer(.1)),.065,
                         lambda x,z:(x/2.4,z/2.4))

    # 航拍可见屋顶成排的深色斜面。只表现聚翰斋和红豆斋的阵列，范围和倾角为近似。
    if height>40:
        corners = list(poly.minimum_rotated_rectangle.exterior.coords)[:4]
        a,b = max(zip(corners,corners[1:]+corners[:1]),key=lambda edge:math.dist(*edge))
        length = math.dist(a,b)
        direction = (np.asarray(b)-a)/length
        normal = np.asarray((-direction[1],direction[0]))
        center = np.asarray(poly.centroid.coords[0])
        safe = prep(poly.buffer(-1.1))
        for distance in np.arange(-length*.48,length*.48,3.1):
            for across in range(-25,26,4):
                p = center + direction*distance + normal*across
                corners_xz = (p,p+2.25*direction,p+2.25*direction+2.6*normal,p+2.6*normal)
                if not all(safe.covers(Point(x,z)) for x,z in corners_xz):
                    continue
                panel_top = [(x,top+h,z) for (x,z),h in zip(corners_xz,(.13,.63,.63,.13))]
                stone.quad(*panel_top,surface_normal(*panel_top[:3]),((0,0),(1,0),(1,1),(0,1)))


def make_buildings():
    walls = {k: MeshWriter() for k in ('cream', 'red', 'city', 'strip', 'perforated', 'teng', 'zhai', 'fortress', 'activity', 'huiwen', 'huiyan', 'xiyuan')}
    roofs = MeshWriter()
    glazed = MeshWriter()
    zhai_stone,zhai_paving = MeshWriter(),MeshWriter()
    # 航拍能对应到楼侧内路的两面，地图共边和其余墙面没有首层廊道假设。
    zhai_arcades = {
        'way/1558576825': LineString((nodes['14178203523'],nodes['8987380714'])),
        'way/1558576826': LineString((nodes['8987380714'],nodes['8987380713'])),
    }
    activity_foundation = MeshWriter()
    road_foundations={name:MeshWriter() for name in
                      ('teachers','northsports','southdorm','windzhai','huiwenroad','facultyroad')}
    huiwen_detail, huiwen_terrace, huiwen_red, huiwen_steps = (MeshWriter() for _ in range(4))
    alumni_garden = MeshWriter()
    archive_wall, archive_seal, archive_fins, archive_door = (MeshWriter() for _ in range(4))
    zhiyi_shell,zhiyi_glass,zhiyi_vents,zhiyi_rails,zhiyi_podium,zhiyi_gallery = (MeshWriter() for _ in range(6))
    zhizhen_shell,zhizhen_glass,zhizhen_side,zhizhen_frame,zhizhen_podium = (MeshWriter() for _ in range(5))
    zhizhi_shell,zhizhi_glass,zhizhi_frame,l4_glass,l4_frame,l4_garden,l4_equipment = (MeshWriter() for _ in range(7))
    zhigong_shell,zhigong_glass,zhigong_dark,zhigong_perforated,zhigong_louvers,zhigong_rails,zhigong_steps,zhigong_sign = (MeshWriter() for _ in range(8))
    studio_shell,studio_glass,studio_sign,studio_columns = (MeshWriter() for _ in range(4))
    zhili_glass,zhili_frame,zhili_side = MeshWriter(),MeshWriter(),MeshWriter()
    zhili_a_shell,zhili_a_glass,zhili_a_frame,zhili_a_sign = (MeshWriter() for _ in range(4))
    zhili_crown,zhili_crown_glass,zhili_crown_frame = MeshWriter(),MeshWriter(),MeshWriter()
    huicui_shell,huicui_glass,huicui_upper,huicui_stage,huicui_frames,huicui_skylights = (MeshWriter() for _ in range(6))
    huizi_shell,huizi_ground,huizi_canopy,huizi_paving,huizi_roof,huizi_equipment = (MeshWriter() for _ in range(6))
    huiyuan_shell,huiyuan_ground,huiyuan_sign,huiyuan_motto,huiyuan_paving,huiyuan_planting,huiyuan_entry_roof = (MeshWriter() for _ in range(7))
    yanhui_roof,yanhui_under,yanhui_steel,yanhui_stone,yanhui_floor,yanhui_seats,yanhui_seat_detail,yanhui_stage,yanhui_screen,yanhui_steps,yanhui_seams = (MeshWriter() for _ in range(11))
    canteen_glass,canteen_frame,canteen_roof,canteen_sign = (MeshWriter() for _ in range(4))
    xiyuan_frame,xiyuan_railing=MeshWriter(),MeshWriter()
    info = []
    obstacle = []
    records = [('校园', item) for item in buildings] + [('周边背景', item) for item in surrounding]
    for region, (key, t, geom) in records:
        if key in GATE_EXCLUDED_BUILDINGS:
            continue  # 已由独立校门模型表示的岗亭不再生成通用楼体。
        name = NAMES.get(key,t.get('name', ''))
        p = geom.representative_point()
        altitude = road_building_bases.get(key,surface_height(p.x,p.y))
        height, basis = height_for(key, t, geom)
        if key == 'way/230847099':
            # 地图记载 5 层，但学校 2021 年资料出现 D605；房间编号与地上层数的关系待核实。
            basis += '；学校资料有D605，实际层数及高差待核实'
        if name == '校友广场':
            # 建筑设计论文记载半地下1层、地上2层；高度按层高近似。
            height, basis = 7.2, '建筑论文：地上2层、半地下1层；总高按层高估计'
        if key == 'way/230956722':
            height, basis = 15.0, '折面屋盖按照片估算15米；OSM记录13米，外部高度未核实'
        if key == 'way/230956723':
            height, basis = 15.5, '北夏馆看台曲面顶棚高度按照片估算，未核实'
        if key == 'way/230956727':
            height, basis = 6.8, '高尔夫练习场两层击球棚按照片估算，未核实'
        if key == 'way/319101979':
            height, basis = 15.2, '南区食堂公开照片及报道显示四层；层高、门洞和连桥入口按视觉近似'
        if key in ('way/230935724','way/230935721'):
            height,basis=18.0,'西苑阁区照片显示开放外廊及多层长翼；高度和构件按外观估计'
        if key == 'way/230935723':
            height,basis=kaifeng_config['height_m'],'公开照片和学校资料显示四层；高度、入口、楼梯及局部立面按照片估计'
        if key == 'relation/3110985':
            # 田径场使用独立看台，不按单栋建筑拉伸。
            height, basis = None, '田径场单独建模，看台高度未核实'
        info.append({'region': region, 'osm': key, 'name': name, 'x_m': round(p.x, 1), 'z_m': round(p.y, 1),
                     'height_m': round(height, 2) if height is not None else '', 'height_source': basis})
        if region == '校园':
            obstacle.append(geom)
        if key in SPECIAL:
            continue
        if key == 'way/230935723':
            # 凯风餐厅由独立模块生成，避免与通用楼体重复叠加。
            continue
        if key in ('way/1272873435','way/1272873436','way/1272873437',
                   'way/1272873438','way/1272873439'):
            # 这五个 OSM 命名分部已经包含在 E / 匚 形主体内，不重复叠加外墙。
            continue
        if key in ('way/230935724','way/230935721'):
            xiyuan_dorm(walls['xiyuan'],xiyuan_frame,xiyuan_railing,roofs,geom,altitude,height)
            continue
        if key in road_building_bases:
            # 坡路局部调整后仍保留原估计楼体标高，用独立基础墙接到地面。
            campus_activity_foundation.add_foundation(
                road_foundations[road_building_groups[key]],geom,top=altitude-1.98,
                terrain_height=rendered_terrain_height,polygons=polygons)
        if key in INDEPENDENT_DORMS:
            continue  # 独立楼栋保留索引、植被避让及坡路基座，外观不再画通用拉伸壳体。
        style = 'city' if region == '周边背景' else building_style(key, t, geom)
        if key == 'way/319101979':
            # 照片可见密集白色竖向构件、南侧二层外廊和标牌；开口位置与尺度为估计。
            entrance = LineString(((430.0,233.9),(430.0,242.0)))
            south_entries = [LineString(((x,283.8),(x,274.5))) for x in (410.0,448.0)]
            ground_entry = LineString(((396.0,283.8),(396.0,274.5)))
            walkway_y = surface_height(430.0,234.0)+4.2
            openings=[(entrance,walkway_y)]+[(p,walkway_y) for p in south_entries]+[(ground_entry,altitude+.03)]
            extrude_with_walkway_openings(canteen_glass,roofs,geom,altitude-2,altitude+height,(),
                                         cap=False,leveled_paths=openings)
            roofs.horizontal(geom,walkway_y)
            canteen_roof.horizontal(geom,altitude+height)
            for poly in polygons(geom):
                coords=list(orient(poly,sign=1.0).exterior.coords)
                for a,b in zip(coords[:-1],coords[1:]):
                    side=LineString((a,b))
                    if side.length < .1:
                        continue
                    for dist in np.arange(0.0,side.length,1.05):
                        p=side.interpolate(dist)
                        levels=[(altitude-1,altitude+height)]
                        for path,level in openings:
                            if path.distance(p)<2.8:
                                levels=[span for low,high in levels for span in
                                        ((low,min(high,level-.12)),(max(low,level+3.12),high)) if span[1]>span[0]]
                        for low,high in levels:
                            if high>low:
                                beam(canteen_frame,(p.x,low,p.y),(p.x,high,p.y),.065)
                    for y in (altitude+4.2,altitude+8.0,altitude+11.8,altitude+height):
                        beam(canteen_frame,(a[0],y,a[1]),(b[0],y,b[1]),.12)
            # 屋顶卫星影像中可见较高矩形体量和较小设施，类型、尺寸及位置未测量。
            for shape,h in ((box(382,245,395,259),1.7),(box(405,245,410,249),.8),
                            (box(425,247,430,250),.7),(box(442,254,447,258),.9)):
                pad=clean(shape.intersection(geom.buffer(-1)))
                if not pad.is_empty:
                    extrude(canteen_roof,canteen_roof,pad,altitude+height,altitude+height+h)
            # 2019 年照片中的红色楼层指示牌：文字重绘，牌位按立面比例近似。
            south_a,south_b=np.asarray((356.10,278.65)),np.asarray((473.02,276.14))
            tangent=(south_b-south_a)/np.linalg.norm(south_b-south_a)
            normal=np.asarray((-tangent[1],tangent[0]))
            def south_point(x,y,offset=.02):
                p=south_a+tangent*((x-south_a[0])/tangent[0])+normal*offset
                return (p[0],y,p[1])
            # 实拍玻璃后可见折线形白色楼梯侧带；位置及楼梯真实构造未取得图纸。
            stair_band=((384.0,altitude+14.9),(405.0,altitude+11.8),(411.0,altitude+11.8),
                        (432.0,altitude+8.6),(440.0,altitude+8.6))
            for a,b in zip(stair_band[:-1],stair_band[1:]):
                p,q=south_point(*a),south_point(*b)
                # 白色带位于立面与竖向格栅之间，避免压在格栅前面。
                canteen_frame.quad((p[0],p[1]-.30,p[2]),(q[0],q[1]-.30,q[2]),
                                   (q[0],q[1]+.30,q[2]),(p[0],p[1]+.30,p[2]),
                                   (normal[0],0,normal[1]),((0,0),(1,0),(1,1),(0,1)))
            # 南侧照片中外廊下面为铺装，分段贴合渲染地形而非悬空水平板。
            for x0 in np.arange(400.0,472.0,3.0):
                x1=min(x0+3.0,473.0)
                points=[]
                for x,offset in ((x0,.08),(x1,.08),(x1,8.0),(x0,8.0)):
                    q=south_point(x,0,offset)
                    points.append((q[0],rendered_terrain_height(q[0],q[2])+.07,q[2]))
                roofs.quad(*points,surface_normal(points[0],points[1],points[2]),
                           ((0,0),(1,0),(1,1),(0,1)))
            for x,bottom,v0,v1 in ((396.0,altitude+2.95,.5,1.0),(410.0,walkway_y+2.72,0.0,.5)):
                center=south_a+tangent*((x-south_a[0])/tangent[0])+normal*.25
                left,right=center-tangent*3.0,center+tangent*3.0
                canteen_sign.quad((left[0],bottom,left[1]),(right[0],bottom,right[1]),
                                  (right[0],bottom+.8,right[1]),(left[0],bottom+.8,left[1]),
                                  (normal[0],0,normal[1]),((0,v0),(1,v0),(1,v1),(0,v1)))
            # 照片中的外廊由浅色立柱支撑，柱距、与外墙的距离为近似。
            for x in (402.0,418.0,434.0,450.0,464.0):
                z=south_a[1]+(x-south_a[0])*(south_b[1]-south_a[1])/(south_b[0]-south_a[0])+6.5
                beam(canteen_frame,(x,altitude-.1,z),(x,walkway_y-.25,z),.23)
            continue
        if key == 'way/230847104':
            huiyuan_building(walls['strip'],roofs,huiyuan_shell,huiyuan_ground,huiyuan_sign,
                             huiyuan_motto,huiyuan_paving,huiyuan_planting,huiyuan_entry_roof,geom,altitude,height)
            continue
        if key.startswith('way/') and key[4:] in huizi_ids:
            huizi_building(walls['strip'],roofs,huizi_roof,huizi_shell,huizi_ground,geom,key[4:],height)
            continue
        if key == 'way/231406259':
            # H3 端部是高架入口，照片可见白色厚墩、红色内墙、上部玻璃与细柱。
            # 门洞比例和支柱位置按照片估计；OSM 只给出轮廓及层数，没有结构图。
            points = list(geom.exterior.coords)
            b,c,d,e = (np.asarray(p,dtype=float) for p in points[1:5])
            def slice_portal(start,end):
                return Polygon((e+(b-e)*start,d+(c-d)*start,
                                d+(c-d)*end,e+(b-e)*end))
            portal = slice_portal(0,1).intersection(geom)
            body = clean(geom.difference(portal))
            extrude(walls['huiwen'],roofs,body,altitude-2,altitude+height,6,3.3)
            header_bottom = altitude+height*.56
            extrude(huiwen_detail,roofs,portal,header_bottom,altitude+height)
            roofs.horizontal(portal,header_bottom)
            huiwen_steps.horizontal(portal,altitude+.04)
            glass_accent(glazed,portal,header_bottom+.5,altitude+height-1.2,
                         lambda length:length>25,.76,2)
            extrude(huiwen_detail,roofs,slice_portal(0,.22),altitude-2,altitude+height)
            extrude(huiwen_red,roofs,slice_portal(.22,.36),altitude-2,header_bottom,6,3.3)
            # 单根柱子支撑悬跨端，柱径和位置均为照片推定。
            column = (e+(b-e)*.80+d+(c-d)*.80)*.5
            beam(huiwen_detail,(column[0],altitude-1,column[1]),
                 (column[0],header_bottom,column[1]),.48)
            # 门前宽阶梯及平台；级数、纵深和高差均为照片近似，不是实测地形。
            for i in range(6):
                p0,p1 = huiwen_outward*(i*1.0),huiwen_outward*((i+1)*1.0)
                strip = Polygon((huiwen_c+p0,huiwen_d+p0,huiwen_d+p1,huiwen_c+p1))
                extrude(huiwen_detail,huiwen_steps,strip,altitude-1.3,altitude-i*.20+.03)
            landing = Polygon((huiwen_c+huiwen_outward*6,huiwen_d+huiwen_outward*6,
                               huiwen_d+huiwen_outward*20,huiwen_c+huiwen_outward*20))
            huiwen_steps.horizontal(landing,altitude-1.17)
            huiwen_roof_detail(huiwen_detail,body,altitude+height+.25)
            continue
        if key == 'way/231406104':
            # H4 圆弧形屋顶由航拍可辨的白色同心退台和暖褐色台面组成。
            extrude(walls['huiwen'],roofs,geom,altitude-2,altitude+height,6,3.3)
            center = Point(-247.7,-545.6)
            for radius in (19,14,9):
                ring = center.buffer(radius,resolution=32).difference(center.buffer(radius-2.3,resolution=32))
                ring = clean(ring.intersection(geom).intersection(box(-280,-545.6,-215,-510)))
                if not ring.is_empty:
                    huiwen_terrace.horizontal(ring,altitude+height+.08)
                    curb = clean(ring.boundary.buffer(.18).intersection(geom))
                    extrude(huiwen_detail,roofs,curb,altitude+height+.05,altitude+height+.30)
            continue
        if style == 'huiwen':
            extrude(walls[style],roofs,geom,altitude-2,altitude+height,6,3.3)
            huiwen_roof_detail(huiwen_detail,geom,altitude+height+.25)
            continue
        if key == 'way/231572974':
            # 2026 年照片显示旧石砌建筑为弧形两层，广场一侧有石柱和白色栏杆。
            # 石柱间距、楼层高度和门窗以照片概括；轮廓取 OSM，非施工图。
            inner=geom.buffer(-1.1)
            if inner.is_empty:
                inner=geom
            extrude(walls['fortress'],roofs,inner,stone_level-.2,stone_level+height,4,4)
            glass_accent(glazed,inner,stone_level+.45,stone_level+2.9,lambda length:length>2,.75)
            for level in (3.5,height):
                roofs.horizontal(geom.buffer(.25),stone_level+level)
            coords=list(geom.exterior.coords)
            front=LineString(coords[1:17])
            for distance in np.arange(0,front.length,4.0):
                p=front.interpolate(distance)
                beam(walls['fortress'],(p.x,stone_level,p.y),(p.x,stone_level+height,p.y),.24)
                beam(roofs,(p.x,stone_level+3.5,p.y),(p.x,stone_level+4.4,p.y),.035)
            for a,b in zip(list(front.coords)[:-1],list(front.coords)[1:]):
                for y in (stone_level+3.65,stone_level+4.4):
                    beam(roofs,(a[0],y,a[1]),(b[0],y,b[1]),.035)
            continue
        if key == 'way/231665582':
            # 泳池新照中的白色三层配套楼：首层浅墙、中层玻璃、水平白色挑檐。
            extrude(roofs,roofs,geom,pool_level-.2,pool_level+3.3)
            extrude(glazed,roofs,geom,pool_level+3.3,pool_level+9.7)
            for bottom,top in ((3.3,4.2),(6.7,7.9),(9.7,height)):
                band=geom.buffer(.35)
                extrude(roofs,roofs,band,pool_level+bottom,pool_level+top)
            continue
        if key == 'way/231795526':
            # 学活两翼与带屋盖的开放中庭参照校园实拍和建筑调研；分界、楼高与构件为近似。
            # 把 OSM 总轮廓按建筑长轴切成两翼，避免将有廊桥的中庭画成整块实心楼。
            aligned=affinity.rotate(geom,20,origin=geom.centroid)
            def world_shape(shape): return affinity.rotate(shape,-20,origin=geom.centroid)
            def world_pt(x,z,y):
                p=affinity.rotate(Point(x,z),-20,origin=geom.centroid)
                return (p.x,y,p.y)
            west=world_shape(aligned.intersection(box(60,-10,98,70)))
            east=world_shape(aligned.intersection(box(135,-10,180,70)))
            atrium=world_shape(aligned.intersection(box(98,-10,135,70)))
            extrude(walls['activity'],roofs,west,altitude-4,altitude+height,6.5,3.4)
            extrude(walls['strip'],roofs,east,altitude-4,altitude+height,5.5,3.4)
            # 独立基础墙只放在两翼楼体下方，随相邻坡地接地；中庭保持开放。
            for wing in (west,east):
                campus_activity_foundation.add_foundation(
                    activity_foundation,wing,top=altitude-3.98,
                    terrain_height=rendered_terrain_height,polygons=polygons)
            roofs.horizontal(atrium,altitude+.04)
            # 中庭两层连廊连接红白两翼，带透明感的白色栏杆。
            walk=world_shape(box(98,27.4,135,31.6).intersection(aligned))
            for y in (altitude+3.4,altitude+6.8):
                extrude(roofs,roofs,walk,y-.25,y)
                for z in (27.4,31.6):
                    beam(roofs,world_pt(98,z,y+1),world_pt(135,z,y+1),.045)
                    for x in np.arange(98,135.1,3):
                        beam(roofs,world_pt(x,z,y),world_pt(x,z,y+1),.04)
            # 轻质顶棚用多个浅拱单元表达照片中的白色网架屋盖。
            for i in range(6):
                xa,xb=98+i*37/6,98+(i+1)*37/6
                for j in range(6):
                    za,zb=20+j*18/6,20+(j+1)*18/6
                    ya=altitude+height+.8*math.sin(math.pi*j/6)
                    yb=altitude+height+.8*math.sin(math.pi*(j+1)/6)
                    p0,p1=world_pt(xa,za,ya),world_pt(xb,za,ya)
                    p2,p3=world_pt(xb,zb,yb),world_pt(xa,zb,yb)
                    roofs.quad(p0,p1,p2,p3,surface_normal(p0,p1,p2),((0,0),(1,0),(1,1),(0,1)))
                beam(roofs,world_pt(xa,20,altitude+height),world_pt(xa,38,altitude+height),.055)
            # 一层到二层的短阶梯仅表示位置关系，实际级数和 U 形转折仍待核实。
            for i in range(20):
                xa,xb=99+i*.45,99+(i+1)*.45
                y=altitude+3.4*(i+1)/20
                p0,p1=world_pt(xa,22,y),world_pt(xb,22,y)
                p2,p3=world_pt(xb,24.4,y),world_pt(xa,24.4,y)
                roofs.quad(p0,p1,p2,p3,(0,1,0),((0,0),(1,0),(1,1),(0,1)))
            continue
        if name == '校友广场':
            # 校友广场航拍显示长条上层主体、南侧屋顶花园与外部楼梯。
            # OSM 只有整体轮廓，内部体量比例及各层高程仍为照片估计。
            nw,sw,se,ne = list(geom.exterior.coords)[:4]
            def mix(a,b,t): return (a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
            rear_w,rear_e = mix(nw,sw,.53),mix(ne,se,.53)
            upper = Polygon((nw,ne,rear_e,rear_w))
            terrace = clean(geom.difference(upper))
            extrude(glazed,roofs,geom,altitude-2,altitude+3.6)
            extrude(glazed,roofs,upper,altitude+3.6,altitude+height)
            for shape,y in ((upper,altitude+height),(terrace,altitude+3.6)):
                inset=shape.buffer(-2)
                if not inset.is_empty:
                    alumni_garden.horizontal(inset,y+.06)
            # 大片落地窗之间有白色水平檐口与竖向边框。
            for a,b in zip(list(upper.exterior.coords)[:-1],list(upper.exterior.coords)[1:]):
                for y in (altitude+3.6,altitude+height):
                    beam(roofs,(a[0],y,a[1]),(b[0],y,b[1]),.23)
            # 两侧外露台阶限制在屋顶花园范围内，连接不同层级的平台。
            def footprint_point(u,v): return np.asarray(mix(mix(nw,ne,u),mix(sw,se,u),v))
            for start_u,end_u in ((.08,.27),(.92,.73)):
                groundpoint=footprint_point(start_u,.90)
                landing=footprint_point(end_u,.53)
                direction=landing-groundpoint
                direction/=np.linalg.norm(direction)
                along=np.asarray((-direction[1],direction[0]))
                low,high=altitude+3.66,altitude+height
                steps=max(2,int(math.ceil((high-low)/.18)))
                for i in range(steps):
                    a=groundpoint+(landing-groundpoint)*i/steps
                    b=groundpoint+(landing-groundpoint)*(i+1)/steps
                    y0,y1=low+(high-low)*i/steps,low+(high-low)*(i+1)/steps
                    def pt(p,sign,y): return (p[0]+along[0]*1.5*sign,y,p[1]+along[1]*1.5*sign)
                    roofs.quad(pt(a,-1,y1),pt(b,-1,y1),pt(b,1,y1),pt(a,1,y1),(0,1,0),
                               ((0,0),(1,0),(1,1),(0,1)))
                    roofs.quad(pt(a,-1,y0),pt(a,1,y0),pt(a,1,y1),pt(a,-1,y1),
                               (-direction[0],0,-direction[1]),((0,0),(1,0),(1,1),(0,1)))
                for side in (-1,1):
                    beam(roofs,pt(groundpoint,side,low+.25),pt(landing,side,high+.25),.14)
            continue
        if key == 'way/319101983':
            # 十层主体采用 OSM 独立楼体分部，不再按整体边界臆造内院。
            tower = way_polygon(ways['776358632'])
            podium = clean(geom.difference(tower))
            if not podium.is_empty:
                extrude(walls[style],roofs,podium,altitude-2,altitude+9.9)
            extrude_teng(walls[style],roofs,glazed,tower,altitude,height)
            continue
        if key == 'way/319101984':
            # 致原楼 OSM 精细分部含两座14层塔楼、不同层数裙房与高空连桥。
            for part_id in ('776358624','776358625','776358626','776358627',
                            '776358628','776358629','776358630','776358631'):
                item = ways[part_id]
                part_tags, part = tags(item), way_polygon(item)
                top = float(part_tags.get('height', float(part_tags.get('building:levels','3'))*3.6))
                bottom = float(part_tags.get('min_height',float(part_tags.get('building:min_level','0'))*3.6))
                extrude(walls[style],roofs,part,altitude+bottom-(2 if bottom==0 else 0),altitude+top)
                if bottom > 0:
                    roofs.horizontal(part,altitude+bottom)
            continue
        if key == 'way/231086125':
            # 致信楼航拍显示底部深色基座、浅色主体和局部连续玻璃幕墙。
            extrude(walls['strip'],roofs,geom,altitude-2,altitude+4.5)
            extrude(walls[style],roofs,geom,altitude+4.5,altitude+height)
            glass_accent(glazed,geom,altitude+6,altitude+height-2,lambda length:length>28,.58,3)
            continue
        if key == 'way/984606893':
            # 致知楼由连续玻璃格架和折线外轮廓组成，不使用通用散窗贴图。
            zhizhi_building(zhizhi_shell,zhizhi_glass,zhizhi_frame,roofs,geom,altitude,height)
            continue
        if key == 'way/984606892':
            south_l4_building(l4_glass,l4_frame,l4_garden,l4_equipment,roofs,geom,altitude,height)
            continue
        if key == 'way/231086366':
            zhigong_building(zhigong_shell,zhigong_glass,zhigong_dark,zhigong_perforated,
                             zhigong_louvers,zhigong_rails,zhigong_steps,zhigong_sign,
                             roofs,geom,altitude,height)
            continue
        if key == 'way/629940892':
            south_grid_facade(zhizhen_shell,zhizhen_glass,zhizhen_side,zhizhen_frame,
                              zhizhen_podium,roofs,geom,altitude,height,'776358635',12,12)
            continue
        if key == 'way/722200010':
            studio_facade(studio_shell,studio_glass,studio_sign,studio_columns,roofs,geom,altitude,height)
            continue
        if key in ('way/629940889','way/629940890'):
            zhiyi_facade(zhiyi_shell,zhiyi_glass,zhiyi_vents,zhiyi_rails,roofs,geom,altitude,height,key)
            continue
        if key == 'way/776358621':
            make_zhiyi_podium(zhiyi_podium,roofs,zhiyi_gallery,geom,altitude,height)
            continue
        if key == 'way/654873808':
            # 致理楼 L2 对照照片使用深色玻璃与浅色横框；两条地图连廊进入时留出门洞。
            extrude_with_walkway_openings(zhili_glass,roofs,geom,altitude-2,altitude+height,
                                         ('776358635','987586196'),zhili_frame,zhili_side)
            continue
        if key == 'way/776358622':
            # L1 入口低层与 17 层塔楼分别对应两块地图轮廓。
            zhili_l1_low_building(zhili_a_shell,zhili_a_glass,zhili_a_frame,zhili_a_sign,roofs,geom,altitude,height)
            continue
        if key == 'way/654873809':
            south_grid_facade(zhizhen_shell,zhizhen_glass,zhizhen_side,zhizhen_frame,
                              zhizhen_podium,roofs,geom,altitude,height,None,6.6,15,
                              zhili_high_walkways)
            continue
        if key == 'way/629940891':
            # 工务署竣工照片可见 L3 两翼白色格窗与深灰短侧；楼间支路保留开口。
            south_grid_facade(zhizhen_shell,zhizhen_glass,zhizhen_side,zhizhen_frame,
                              zhizhen_podium,roofs,geom,altitude,height,'1272661381',6.6,10,
                              zhili_high_walkways)
            # OSM 将伸向演播厅的高层悬挑体单列为 building:part，不能忽略。
            zhili_cantilever(zhili_crown,zhili_crown_glass,zhili_crown_frame,
                             way_polygon(ways['776358623']),geom,altitude)
            continue
        if key == 'way/938357240':
            huicui_building(huicui_shell,huicui_glass,huicui_upper,huicui_stage,
                             huicui_frames,huicui_skylights,roofs,geom,altitude,height)
            continue
        if key == 'way/1081134194':
            huiyi_original_building(huicui_shell,huicui_glass,huicui_frames,
                                    walls['strip'],roofs,geom,altitude,height)
            continue
        if key == 'way/1455957244':
            huiyi_teaching_lab(huicui_shell,huicui_glass,huicui_frames,roofs,geom,altitude,height)
            continue
        if key == 'way/1207358023':
            # 与教学实验楼重叠的部分已裁去，此处为规划图中独立的五层扩建部分。
            huicui_neighbor_tower(huicui_shell,huicui_glass,huicui_frames,roofs,geom,altitude,height,5,3.4,True)
            for z in (-673,-655,-635):
                skylight=box(160,z,171,z+2).intersection(geom)
                if not skylight.is_empty:
                    huicui_skylights.horizontal(skylight,altitude+height+.08)
            continue
        if key in ('way/1271852286','way/1271852287'):
            # 吴玉章楼总平面图标记 13F，汇语楼核查表明确 14F；立面根据实拍近似。
            is_huiyu=key=='way/1271852287'
            huicui_neighbor_tower(huicui_shell,huicui_glass,huicui_frames,roofs,geom,altitude,height,
                                  14 if is_huiyu else 13,2.4 if is_huiyu else 3.5,is_huiyu)
            continue
        if key in ('way/1558576826','way/1558576825','way/1558576824'):
            new_zhai_facade(walls[style],zhai_stone,glazed,roofs,zhai_paving,
                            geom,altitude,height,zhai_arcades.get(key))
            continue
        if key == 'way/317029242':
            # 档案馆实景为近乎无窗的浅色盒体，短侧面嵌有大幅印章文字。
            extrude(archive_wall,roofs,geom,altitude-2,altitude+height)
            for poly in polygons(geom):
                coords = list(orient(poly,sign=1.0).exterior.coords)
                for a,b in zip(coords[:-1],coords[1:]):
                    dx,dz = b[0]-a[0],b[1]-a[1]
                    length = math.hypot(dx,dz)
                    if length < .1:
                        continue
                    nx,nz = dz/length,-dx/length
                    if length < 45:
                        # 双侧印章纹样只做抽象的可辨识近似。
                        inset=.8
                        aa=(a[0]+dx*inset/length+nx*.12,a[1]+dz*inset/length+nz*.12)
                        bb=(b[0]-dx*inset/length+nx*.12,b[1]-dz*inset/length+nz*.12)
                        archive_seal.quad((aa[0],altitude+1.2,aa[1]),(bb[0],altitude+1.2,bb[1]),
                                          (bb[0],altitude+height-1.2,bb[1]),(aa[0],altitude+height-1.2,aa[1]),
                                          (nx,0,nz),((1,1),(0,1),(0,0),(1,0)))
                    else:
                        for i in range(1,41):
                            t=i/42
                            x,z=a[0]+dx*t+nx*.13,a[1]+dz*t+nz*.13
                            beam(archive_fins,(x,altitude+1.5,z),(x,altitude+height-.5,z),.06)
                        # 以退后的玻璃入口表达正立面；位置按渲染图近似。
                        aa=(a[0]+dx*.43+nx*.2,a[1]+dz*.43+nz*.2)
                        bb=(a[0]+dx*.57+nx*.2,a[1]+dz*.57+nz*.2)
                        archive_door.quad((aa[0],altitude+.2,aa[1]),(bb[0],altitude+.2,bb[1]),
                                          (bb[0],altitude+3.6,bb[1]),(aa[0],altitude+3.6,aa[1]),
                                          (nx,0,nz),((0,0),(3,0),(3,1),(0,1)))
            continue
        if key == 'way/1455962776':
            yanhui_building(yanhui_roof,yanhui_under,yanhui_steel,yanhui_stone,yanhui_floor,
                            yanhui_seats,yanhui_seat_detail,yanhui_stage,yanhui_screen,yanhui_steps,yanhui_seams,geom,altitude,height)
            continue
        if key == 'way/230956722':
            # 正面、桁架及独立屋盖在 make_yuanping_and_beixia 中绘制。
            continue
        if key == 'way/230956723':
            # 北夏馆为露天池旁的看台与蓝色曲面顶棚，单独制作而非封闭盒子。
            continue
        if key == 'way/230956727':
            # 弧形击球棚按地图轮廓及实拍单独制作，保留上下两层开放击球位。
            continue
        if key == 'way/1085792762':
            # 航拍有视差：以 OSM 北侧边缘结合施工照估计矩形塔楼，不能作测绘轮廓。
            # 广阔裙房与塔楼各自建模；裙房层高及分段尚无竣工图。
            low = min(9.0, height)
            extrude(walls['city'], roofs, geom, altitude - 2, altitude + low)
            tower = Polygon(((508.16,-508.26),(568.36,-524.65),
                             (581.02,-480.67),(520.82,-464.28))).intersection(geom)
            extrude(walls[style], roofs, tower, altitude + low, altitude + height,
                    unit_width=2.5,unit_height=3.6)
        else:
            extrude(walls[style], roofs, geom, altitude - 2, altitude + height)
    # 大道对面的城市轮廓来自单独的 OSM 快照，只作为远景示意。
    # 已有校园/周边建筑不重复；未标注高度时用常见层高估算。
    existing={key for key,_,_ in buildings+surrounding}
    for way in city_ways:
        t=tags(way); p=city_points(way); key='way/'+way.get('id')
        if 'building' not in t or key in existing or len(p)<4 or p[0]!=p[-1]: continue
        geom=clean(Polygon(p))
        if geom.is_empty or geom.area<100 or not city_extent.buffer(-1).covers(geom): continue
        point=geom.representative_point()
        try:
            h=float(t.get('height','').replace('m','').strip())
            if not 3<=h<=400: raise ValueError()
            basis='OSM height'
        except ValueError:
            try:
                levels=float(t.get('building:levels',''))
                if not 1<=levels<=100: raise ValueError()
                h=levels*3.3; basis=f'OSM {levels:g}层，层高估算3.3米'
            except ValueError:
                h=18.0; basis='未见高度，按远景体量估算'
        altitude=city_surface(point.x,point.y)
        extrude(walls['city'],roofs,geom,altitude-2,altitude+h)
        info.append({'region':'北侧远景','osm':key,'name':t.get('name',''),
                     'x_m':round(point.x,1),'z_m':round(point.y,1),
                     'height_m':round(h,2),'height_source':basis})
    huizi_roof_details(huizi_shell,huizi_canopy,roofs,huizi_equipment)
    huizi_entry(huizi_shell,huizi_paving)
    counts = {style: mesh.save(ASSETS / f'campus-wall-{style}.mesh') for style, mesh in walls.items()}
    counts['roof'] = roofs.save(ASSETS / 'campus-roofs.mesh')
    counts['xiyuan-frame'] = xiyuan_frame.save(ASSETS / 'campus-xiyuan-frame.mesh')
    counts['xiyuan-railing'] = xiyuan_railing.save(ASSETS / 'campus-xiyuan-railing.mesh')
    counts['glazed'] = glazed.save(ASSETS / 'campus-landmark-glass.mesh')
    # 与楼侧廊道共用石材，院内地表仍使用原渲染高程。
    drape_on_terrain(zhai_paving,zhai_courtyard,.065,lambda x,z:(x/4.8,z/4.8))
    counts['zhai-stone'] = zhai_stone.save(ASSETS/'campus-zhai-stone.mesh')
    counts['zhai-paving'] = zhai_paving.save(ASSETS/'campus-zhai-paving.mesh')
    counts['activity-foundation'] = activity_foundation.save(ASSETS / 'campus-activity-foundation.mesh')
    for name,mesh in road_foundations.items():
        counts[name+'-foundation'] = mesh.save(ASSETS / f'campus-{name}-foundation.mesh')
    counts['alumni-garden'] = alumni_garden.save(ASSETS / 'campus-alumni-garden.mesh')
    for name,mesh in (('canteen-glass',canteen_glass),('canteen-frame',canteen_frame),
                      ('canteen-roof',canteen_roof),('canteen-sign',canteen_sign)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    for name,mesh in (('huiwen-detail',huiwen_detail),('huiwen-terrace',huiwen_terrace),
                      ('huiwen-red',huiwen_red),('huiwen-steps',huiwen_steps)):
        counts[name] = mesh.save(ASSETS / f'campus-{name}.mesh')
    for name,mesh in (('archive-wall',archive_wall),('archive-seal',archive_seal),
                      ('archive-fins',archive_fins),('archive-door',archive_door)):
        counts[name] = mesh.save(ASSETS / f'campus-{name}.mesh')
    for name,mesh in (('zhiyi-shell',zhiyi_shell),('zhiyi-glass',zhiyi_glass),('zhiyi-vents',zhiyi_vents),
                      ('zhiyi-rails',zhiyi_rails),('zhiyi-podium',zhiyi_podium),('zhiyi-gallery',zhiyi_gallery)):
        counts[name] = mesh.save(ASSETS / f'campus-{name}.mesh')
    for name,mesh in (('zhizhen-shell',zhizhen_shell),('zhizhen-glass',zhizhen_glass),
                      ('zhizhen-side',zhizhen_side),('zhizhen-frame',zhizhen_frame),
                      ('zhizhen-podium',zhizhen_podium),('studio-shell',studio_shell),
                      ('studio-glass',studio_glass),('studio-sign',studio_sign),('studio-columns',studio_columns)):
        counts[name] = mesh.save(ASSETS / f'campus-{name}.mesh')
    for name,mesh in (('zhizhi-shell',zhizhi_shell),('zhizhi-glass',zhizhi_glass),('zhizhi-frame',zhizhi_frame),
                      ('l4-glass',l4_glass),('l4-frame',l4_frame),('l4-garden',l4_garden),('l4-equipment',l4_equipment)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    for name,mesh in (('zhigong-shell',zhigong_shell),('zhigong-glass',zhigong_glass),
                      ('zhigong-dark',zhigong_dark),('zhigong-perforated',zhigong_perforated),
                      ('zhigong-louvers',zhigong_louvers),('zhigong-rails',zhigong_rails),
                      ('zhigong-steps',zhigong_steps),('zhigong-sign',zhigong_sign)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    for name,mesh in (('zhili-a-shell',zhili_a_shell),('zhili-a-glass',zhili_a_glass),
                      ('zhili-a-frame',zhili_a_frame),('zhili-a-sign',zhili_a_sign)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    counts['zhili-glass'] = zhili_glass.save(ASSETS/'campus-zhili-glass.mesh')
    counts['zhili-frame'] = zhili_frame.save(ASSETS/'campus-zhili-frame.mesh')
    counts['zhili-side'] = zhili_side.save(ASSETS/'campus-zhili-side.mesh')
    for name,mesh in (('zhili-crown',zhili_crown),('zhili-crown-glass',zhili_crown_glass),
                      ('zhili-crown-frame',zhili_crown_frame)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    for name,mesh in (('huicui-shell',huicui_shell),('huicui-glass',huicui_glass),
                      ('huicui-upper',huicui_upper),('huicui-stage',huicui_stage),
                      ('huicui-frames',huicui_frames),('huicui-skylights',huicui_skylights)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    for name,mesh in (('huizi-shell',huizi_shell),('huizi-ground',huizi_ground),
                      ('huizi-canopy',huizi_canopy),('huizi-paving',huizi_paving),
                      ('huizi-roof',huizi_roof),('huizi-equipment',huizi_equipment)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    for name,mesh in (('huiyuan-shell',huiyuan_shell),('huiyuan-ground',huiyuan_ground),
                      ('huiyuan-sign',huiyuan_sign),('huiyuan-motto',huiyuan_motto),
                      ('huiyuan-paving',huiyuan_paving),('huiyuan-planting',huiyuan_planting),('huiyuan-entry-roof',huiyuan_entry_roof)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    for name,mesh in (('yanhui-roof',yanhui_roof),('yanhui-under',yanhui_under),('yanhui-steel',yanhui_steel),
                      ('yanhui-stone',yanhui_stone),('yanhui-floor',yanhui_floor),('yanhui-seats',yanhui_seats),('yanhui-seat-detail',yanhui_seat_detail),
                      ('yanhui-stage',yanhui_stage),('yanhui-screen',yanhui_screen),
                      ('yanhui-steps',yanhui_steps),('yanhui-seams',yanhui_seams)):
        counts[name] = mesh.save(ASSETS/f'campus-{name}.mesh')
    with (DATA / 'building-index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['region', 'osm', 'name', 'x_m', 'z_m', 'height_m', 'height_source'])
        writer.writeheader()
        writer.writerows(info)
    return counts, unary_union(obstacle)


def make_water():
    mesh, bank = MeshWriter(), MeshWriter()
    for _, geom, y in lake_heights:
        mesh.horizontal(geom, y + .2)
        # 航拍可见浅色湖岸护边；具体宽度随地段变化，这里按窄边近似。
        edge = geom.buffer(1.1).difference(geom.buffer(-.15))
        bank.horizontal(edge,y+.26)
    return mesh.save(ASSETS / 'campus-water.mesh'),bank.save(ASSETS / 'campus-water-bank.mesh')


def make_wenshan_terraces():
    """按可见的半圆亲水台阶近似两处上文山湖景观；非实测平面。"""
    mesh,detail = MeshWriter(),MeshWriter()

    def half_ellipse(cx,cz,front,side,forward):
        fx,fz=forward
        length=math.hypot(fx,fz)
        fx,fz=fx/length,fz/length
        sx,sz=-fz,fx
        pts=[]
        for i in range(33):
            a=-math.pi/2+math.pi*i/32
            pts.append((cx+front*math.cos(a)*fx+side*math.sin(a)*sx,
                        cz+front*math.cos(a)*fz+side*math.sin(a)*sz))
        return Polygon(pts)

    def rear_apron(cx,cz,half_width,depth,forward):
        fx,fz=forward
        length=math.hypot(fx,fz)
        fx,fz=fx/length,fz/length
        sx,sz=-fz,fx
        return Polygon([(cx+sx*half_width,cz+sz*half_width),
                        (cx-sx*half_width,cz-sz*half_width),
                        (cx-sx*half_width-fx*depth,cz-sz*half_width-fz*depth),
                        (cx+sx*half_width-fx*depth,cz+sz*half_width-fz*depth)])

    # 西岸：朝湖心展开的近半圆台阶及上部亲水平台。
    for i in range(4):
        area=half_ellipse(-377,190,15.0-i*1.35,14.0-i*1.05,(.72,.69))
        extrude(mesh,mesh,area,upper_lake_level+.21,upper_lake_level+.36+i*.18)
    extrude(mesh,mesh,rear_apron(-377,190,10.85,6.5,(.72,.69)),
            upper_lake_level+.21,upper_lake_level+.90)
    # 东南岸：较宽的弧形台阶，沿水边展开并略微朝西北。
    for i in range(4):
        area=half_ellipse(-231,211,9.5-i*1.1,15.5-i*.75,(-.97,-.24))
        extrude(mesh,mesh,area,upper_lake_level+.21,upper_lake_level+.36+i*.18)
    extrude(mesh,mesh,rear_apron(-231,211,13.25,5.0,(-.97,-.24)),
            upper_lake_level+.21,upper_lake_level+.90)
    # 西岸相邻的三瓣休憩铺地，轮廓和其中灌木位置参考 2023 年航拍。
    extrude(mesh,mesh,trefoil_patio,upper_lake_level+.92,upper_lake_level+1.06)
    # 亲水平台绕树的放射形铺装接缝和三瓣小广场的弧形边线。
    stage_top=unary_union((half_ellipse(-377,190,10.95,10.85,(.72,.69)),
                           rear_apron(-377,190,10.85,6.5,(.72,.69))))
    fx,fz=.72/math.hypot(.72,.69),.69/math.hypot(.72,.69)
    sx,sz=-fz,fx
    for i in range(11):
        a=-math.pi/2+math.pi*i/10
        end=(-377+10.65*(math.cos(a)*fx+math.sin(a)*sx),
             190+10.65*(math.cos(a)*fz+math.sin(a)*sz))
        seam=LineString(((-381.5,185.5),end)).buffer(.045,cap_style='flat').intersection(stage_top)
        detail.horizontal(seam,upper_lake_level+.916)
    ring=Point(-381.5,185.5).buffer(1.8).difference(Point(-381.5,185.5).buffer(1.72))
    detail.horizontal(ring.intersection(stage_top),upper_lake_level+.918)
    for x,z in trefoil_centers:
        ring=Point(x,z).buffer(3.45).difference(Point(x,z).buffer(3.36)).intersection(trefoil_patio)
        detail.horizontal(ring,upper_lake_level+1.076)
    return {'terraces':mesh.save(ASSETS/'campus-wenshan-terraces.mesh'),
            'paving_detail':detail.save(ASSETS/'campus-wenshan-paving-detail.mesh')}


def beam(mesh, start, end, radius):
    """以带法线的方形梁连接两点，桥柱、立杆和横梁共用。"""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = b - a
    length = np.linalg.norm(direction)
    if length < .01:
        return
    direction /= length
    ref = np.array((0., 1., 0.)) if abs(direction[1]) < .95 else np.array((1., 0., 0.))
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    corners_a = [a + radius*(su*u + sv*v) for su, sv in ((-1,-1),(1,-1),(1,1),(-1,1))]
    corners_b = [b + radius*(su*u + sv*v) for su, sv in ((-1,-1),(1,-1),(1,1),(-1,1))]
    for i in range(4):
        j = (i+1) % 4
        normal = np.cross(corners_b[i]-corners_a[i], corners_a[j]-corners_a[i])
        normal /= max(np.linalg.norm(normal), .0001)
        mesh.quad(corners_a[i], corners_b[i], corners_b[j], corners_a[j], normal,
                  ((0,0),(length/4,0),(length/4,1),(0,1)))


def round_beam(mesh, start, end, radius, segments=10):
    """用圆截面近似钢管斜撑，实际管径和连接节点未测量。"""
    a,b = np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    direction = b-a
    length = np.linalg.norm(direction)
    if length < .01:
        return
    direction /= length
    ref = np.array((0.,1.,0.)) if abs(direction[1]) < .95 else np.array((1.,0.,0.))
    u = np.cross(direction,ref)
    u /= np.linalg.norm(u)
    v = np.cross(direction,u)
    for i in range(segments):
        t0,t1 = 2*math.pi*i/segments,2*math.pi*(i+1)/segments
        n0,n1 = u*math.cos(t0)+v*math.sin(t0),u*math.cos(t1)+v*math.sin(t1)
        normal = n0+n1
        normal /= np.linalg.norm(normal)
        mesh.quad(a+n0*radius,b+n0*radius,b+n1*radius,a+n1*radius,normal,
                  ((0,0),(length/4,0),(length/4,1),(0,1)))


def make_zhiyi_skywalks():
    """学校工务署照片中的三层露天连桥；精确高程与尺寸未经实测。"""
    deck,rails=MeshWriter(),MeshWriter()
    a=np.array((zhiyi_bridge_ends[0].x,zhiyi_bridge_ends[0].y))
    b=np.array((zhiyi_bridge_ends[1].x,zhiyi_bridge_ends[1].y))
    direction=(b-a)/np.linalg.norm(b-a)
    side=np.array((-direction[1],direction[0]))
    start,end=a-direction*.30,b+direction*.30
    footprint=Polygon((start-side*1.30,end-side*1.30,end+side*1.30,start+side*1.30))
    def edge(p,sign,y):
        return (p[0]+sign*side[0]*1.23,y,p[1]+sign*side[1]*1.23)
    for floor in (4,6,8):
        y=zhiyi_level+floor*3.3
        extrude(deck,deck,footprint,y-.35,y+.03)
        for sign in (-1,1):
            for height in (.25,.52,.79,1.05):
                beam(rails,edge(start,sign,y+height),edge(end,sign,y+height),.025)
            for i in range(6):
                p=start+(end-start)*i/5
                beam(rails,edge(p,sign,y),edge(p,sign,y+1.05),.035)
        # 桥板底部的几道细梁参照照片表达，不代表结构详图。
        for i in range(1,5):
            p=start+(end-start)*i/5
            beam(deck,edge(p,-1,y-.39),edge(p,1,y-.39),.06)
    return {'deck':deck.save(ASSETS/'campus-zhiyi-skywalk.mesh'),
            'rails':rails.save(ASSETS/'campus-zhiyi-skywalk-rails.mesh')}


def make_zhili_high_walkways():
    """L1 与 L3 间地图中的三条人行线：楼层、高程和护栏均为估计。"""
    deck,rails=MeshWriter(),MeshWriter()
    base=surface_height(343,512)
    for path_id,floor in zhili_high_walkways:
        points=way_points(ways[path_id])
        a,b=np.asarray(points[0]),np.asarray(points[-1])
        direction=(b-a)/np.linalg.norm(b-a)
        side=np.asarray((-direction[1],direction[0]))
        start,end=a-direction*.18,b+direction*.18
        footprint=Polygon((start-side*1.4,end-side*1.4,end+side*1.4,start+side*1.4))
        y=base+(floor-1)*3.3
        extrude(deck,deck,footprint,y-.35,y+.03)
        def edge(p,sign,h):
            return (p[0]+sign*side[0]*1.34,h,p[1]+sign*side[1]*1.34)
        for sign in (-1,1):
            n=(sign*side[0],0,sign*side[1])
            deck.quad(edge(start,sign,y),edge(end,sign,y),edge(end,sign,y+.72),
                      edge(start,sign,y+.72),n,((0,0),(1,0),(1,1),(0,1)))
            beam(rails,edge(start,sign,y+1.05),edge(end,sign,y+1.05),.04)
            for i in range(7):
                p=start+(end-start)*i/6
                beam(rails,edge(p,sign,y+.72),edge(p,sign,y+1.05),.035)
    return {'deck':deck.save(ASSETS/'campus-zhili-high-walkways.mesh'),
            'rails':rails.save(ASSETS/'campus-zhili-high-walkway-rails.mesh')}


def make_zhigong_parking():
    """停车位与少量车辆按影像概括，车的数量和停放位置不作为现场记录。"""
    light,dark,glass,wheels,chargers=(MeshWriter() for _ in range(5))
    lot=zhigong_parking.buffer(-1.0)
    count=0
    for row,z in enumerate((651.5,665.5,679.5,693.5)):
        for col,x in enumerate(np.arange(82.3,154,2.6)):
            if (col+row*3)%7 not in (0,2,4):
                continue
            footprint=box(x-.82,z-2.05,x+.82,z+2.05)
            if not lot.covers(footprint):
                continue
            count+=1
            y=surface_height(x,z)
            body=light if (col+row)%4 else dark
            extrude(body,body,footprint,y+.25,y+.80)
            x0,x1=x-.73,x+.73
            z0,z1=z-1.20,z+1.10
            w0,w1=z-.70,z+.65
            lo,hi=y+.80,y+1.40
            uv=((0,0),(1,0),(1,1),(0,1))
            glass.quad((x0,lo,z0),(x0,lo,z1),(x0,hi,w1),(x0,hi,w0),(-1,0,0),uv)
            glass.quad((x1,lo,z1),(x1,lo,z0),(x1,hi,w0),(x1,hi,w1),(1,0,0),uv)
            glass.quad((x1,lo,z0),(x0,lo,z0),(x0,hi,w0),(x1,hi,w0),(0,0,-1),uv)
            glass.quad((x0,lo,z1),(x1,lo,z1),(x1,hi,w1),(x0,hi,w1),(0,0,1),uv)
            body.quad((x0,hi,w0),(x0,hi,w1),(x1,hi,w1),(x1,hi,w0),(0,1,0),uv)
            for dx in (-.88,.88):
                for dz in (-1.25,1.25):
                    extrude(wheels,wheels,box(x+dx-.13,z+dz-.38,x+dx+.13,z+dz+.38),y+.05,y+.52)
    # 校方 2025 年通知在该停车场规划充电设施；示意桩的具体数量与位置未核实。
    for x in np.arange(84.5,153.0,5.2):
        z=655.5
        pad=box(x-.28,z-.24,x+.28,z+.24)
        if not lot.covers(pad):
            continue
        y=surface_height(x,z)
        extrude(chargers,chargers,pad,y,y+1.2)
        glass.quad((x-.20,y+.65,z-.245),(x+.20,y+.65,z-.245),
                   (x+.20,y+1.04,z-.245),(x-.20,y+1.04,z-.245),(0,0,-1),
                   ((0,0),(1,0),(1,1),(0,1)))
    return {'cars':count,
            'light':light.save(ASSETS/'campus-zhigong-cars-light.mesh'),
            'dark':dark.save(ASSETS/'campus-zhigong-cars-dark.mesh'),
            'glass':glass.save(ASSETS/'campus-zhigong-cars-glass.mesh'),
            'wheels':wheels.save(ASSETS/'campus-zhigong-cars-wheels.mesh'),
            'chargers':chargers.save(ASSETS/'campus-zhigong-chargers.mesh')}


def make_huixing_landscape():
    # 利用地图中的台阶线和同一高程网格制作实体踏步；高度与宽度不是现场测量。
    steps,hedges=MeshWriter(),MeshWriter()
    mapped=way_points(ways['237891957'])
    a,b=np.asarray(mapped[0]),np.asarray(mapped[-1])
    direction=(b-a)/np.linalg.norm(b-a)
    lateral=np.asarray((-direction[1],direction[0]))
    count=18
    def corner(p,side,y):
        return (p[0]+side*lateral[0]*4.0,y,p[1]+side*lateral[1]*4.0)
    levels=[]
    for i in range(count):
        p0,p1=a+(b-a)*i/count,a+(b-a)*(i+1)/count
        samples=(p0+lateral*4,p0-lateral*4,p1+lateral*4,p1-lateral*4,(p0+p1)*.5)
        # 四角均高于局部地形，避免倾斜坡面从宽台阶的侧边穿出。
        levels.append(max(surface_height(p[0],p[1]) for p in samples)+.08)
    for i in range(count):
        p0,p1=a+(b-a)*i/count,a+(b-a)*(i+1)/count
        y=levels[i]
        steps.quad(corner(p0,-1,y),corner(p1,-1,y),corner(p1,1,y),corner(p0,1,y),
                   (0,1,0),((0,0),(1,0),(1,1),(0,1)))
        if i+1<count:
            low=min(levels[i],levels[i+1])-.02
            high=max(levels[i],levels[i+1])
            steps.quad(corner(p1,-1,low),corner(p1,1,low),corner(p1,1,high),corner(p1,-1,high),
                       (direction[0],0,direction[1]),((0,0),(1,0),(1,1),(0,1)))
    # 草坪中各排矮篱笆根据航拍概括；每行分段适应粗地形，种类和株数未核实。
    for garden in huixing_gardens:
        xmin,zmin,xmax,zmax=garden.bounds
        for i in range(5):
            x=xmin+(xmax-xmin)*(i+1)/6
            for j in range(3):
                z0=zmin+1+(zmax-zmin-2)*j/3
                z1=zmin+1+(zmax-zmin-2)*(j+1)/3
                beam(hedges,(x,surface_height(x,z0)+.42,z0),
                     (x,surface_height(x,z1)+.42,z1),.38)
    return {'steps':steps.save(ASSETS/'campus-huixing-steps.mesh'),
            'hedges':hedges.save(ASSETS/'campus-huixing-hedges.mesh')}


def drape_on_terrain(mesh, footprint, offset, uv_map):
    """按地形三角形裁分铺装，保证坡路紧贴可见地面而不依赖水平平面。"""
    if footprint.is_empty:
        return
    nz,nx=terrain_grid.shape
    dx,dz=(X1-X0)/(nx-1),(Z1-Z0)/(nz-1)
    minx,minz,maxx,maxz=footprint.bounds
    i0=max(0,int((minx-X0)/dx));i1=min(nx-2,int((maxx-X0)/dx))
    j0=max(0,int((minz-Z0)/dz));j1=min(nz-2,int((maxz-Z0)/dz))
    prepared=prep(footprint)
    for j in range(j0,j1+1):
        for i in range(i0,i1+1):
            for x,z,sx,sz in terrain_cell_parts(i,j):
                corners=((x,z),(x,z+sz),(x+sx,z+sz),(x+sx,z))
                for indexes in ((0,1,2),(0,2,3)):
                    triangle=Polygon([corners[k] for k in indexes])
                    if not prepared.intersects(triangle):
                        continue
                    surface=footprint.intersection(triangle)
                    for poly in polygons(surface):
                        if poly.area<1e-5:
                            continue
                        poly=orient(poly,sign=1.0)
                        rings=[list(poly.exterior.coords)[:-1]]+[list(r.coords)[:-1] for r in poly.interiors]
                        rings=[ring for ring in rings if len(ring)>=3]
                        if not rings:
                            continue
                        vertices=np.asarray([p for ring in rings for p in ring],dtype=np.float64)
                        ends=np.cumsum([len(ring) for ring in rings],dtype=np.uint32)
                        for index in mapbox_earcut.triangulate_float64(vertices,ends):
                            px,pz=vertices[index]
                            mesh.v((px,rendered_terrain_height(px,pz)+offset,pz),rendered_terrain_normal(px,pz),uv_map(px,pz))


def make_road_details():
    """在地图道路边界放置低路缘，并在杜鹃山生成贴合坡地的步道。"""
    road_mesh,marking_mesh,curbs,trail,boardwalk=(MeshWriter() for _ in range(5))
    walking=[g for _,t,g in roadways if t.get('highway') in PEDESTRIAN_ROADS
             and t.get('bridge') in (None,'no') and t.get('tunnel') in (None,'no','building_passage')]
    walkway_openings=unary_union([g.buffer(1.8) for g in walking]
                                 +[p.buffer(2.2) for _,p,_ in mapped_crossings])
    # 广场、湖池和建筑已有独立铺装；只在实际道路外轮廓绘制路缘。
    keep_clear=unary_union([walkway_openings,time_square_paving,huicui_forecourt,south_courtyard,
                            yuanping_paving,zhizhi_courtyard,liyan_entry,zhiyuan_entry]
                           +[g.buffer(.4) for _,_,g in waters]
                           +[g.buffer(.2) for _,_,g in buildings])
    # 局部车行面覆盖粗地图纹理，以细颗粒贴图和真实三角面保留近景路面细节。
    visual_clear=unary_union([time_square_paving,huicui_forecourt,south_courtyard,yuanping_paving,
                              zhizhi_courtyard,liyan_entry,zhiyuan_entry]
                             +[g for _,_,g in waters]+[g for _,_,g in buildings])
    pavement=vehicle_surface.intersection(campus.buffer(3)).difference(visual_clear)
    drape_on_terrain(road_mesh,pavement,.035,lambda x,z:(x/8.0,z/8.0))
    drape_on_terrain(marking_mesh,road_markings.intersection(pavement),.065,lambda x,z:(x,z))
    curb_lines=vehicle_surface.boundary.intersection(campus.buffer(3)).difference(keep_clear)
    uv4=((0,0),(1,0),(1,1),(0,1))
    for line in lines(curb_lines):
        if line.length<.3:
            continue
        distances=list(np.arange(0,line.length,3.0))+[line.length]
        for a,b in zip(distances[:-1],distances[1:]):
            p,q=line.interpolate(a),line.interpolate(b)
            vx,vz=q.x-p.x,q.y-p.y
            length=math.hypot(vx,vz)
            if length<.05:
                continue
            nx,nz=-vz/length,vx/length
            py,qy=rendered_terrain_height(p.x,p.y),rendered_terrain_height(q.x,q.y)
            def edge(point,y,side,up):
                return (point.x+nx*.085*side,y+up,point.y+nz*.085*side)
            al,ar=edge(p,py,-1,.14),edge(p,py,1,.14)
            bl,br=edge(q,qy,-1,.14),edge(q,qy,1,.14)
            curbs.quad(al,bl,br,ar,surface_normal(al,bl,br),uv4)
            for side in (-1,1):
                curbs.quad(edge(p,py,side,.01),edge(q,qy,side,.01),
                           edge(q,qy,side,.14),edge(p,py,side,.14),(nx*side,0,nz*side),uv4)

    # OSM 的山路只记录足迹；学校资料说明此处设有铺装与塑木栈道。
    # 木栈道具体对应哪一条地图线尚未核实，只选择中央一段作宽度近似。
    peak=Point(nodes['2406038732'])
    area=peak.buffer(100)
    footways=[g.intersection(area) for key,t,g in roadways if key.startswith('way/256680')
              and t.get('highway')=='footway' and t.get('bridge') in (None,'no') and g.intersects(area)]
    all_trails=unary_union([g.buffer(.9,cap_style='round',join_style='round') for g in footways])
    wood_axis=LineString(way_points(ways['256680438']))
    wooden=wood_axis.intersection(area).buffer(.65,cap_style='round',join_style='round').difference(vehicle_surface)
    stone=all_trails.difference(wooden).difference(vehicle_surface)
    drape_on_terrain(trail,stone,.055,lambda x,z:(x/3.0,z/3.0))
    def wood_uv(x,z):
        p=Point(x,z)
        distance=wood_axis.project(p)
        center=wood_axis.interpolate(distance)
        a=wood_axis.interpolate(max(0,distance-.4));b=wood_axis.interpolate(min(wood_axis.length,distance+.4))
        vx,vz=b.x-a.x,b.y-a.y
        length=max(.001,math.hypot(vx,vz))
        across=((x-center.x)*(-vz)+(z-center.y)*vx)/length
        return (distance/2.0,.5+across/1.3)
    drape_on_terrain(boardwalk,wooden,.055,wood_uv)
    return {'pavement':road_mesh.save(ASSETS/'campus-road-surface.mesh'),
            'markings':marking_mesh.save(ASSETS/'campus-road-markings.mesh'),
            'curb':curbs.save(ASSETS/'campus-road-curbs.mesh'),
            'dujuan-trail':trail.save(ASSETS/'campus-dujuan-trail.mesh'),
            'dujuan-boardwalk':boardwalk.save(ASSETS/'campus-dujuan-boardwalk.mesh')}


def make_road_environment():
    """汇总其他独立景点的保留区，再生成全校道路的人行与绿化层。"""
    peak=Point(nodes['2406038732'])
    dujuan=unary_union([g.intersection(peak.buffer(100)).buffer(1.5)
                        for key,t,g in roadways if key.startswith('way/256680')
                        and t.get('highway')=='footway' and g.intersects(peak.buffer(100))])
    gate_walk=LineString(LIDE_WALK_POINTS).buffer(3)
    # 保留已单独制作的广场、景园、门区和商铺人行带，避免重复铺装和覆盖专用材质。
    reserved=unary_union((time_square_paving,huicui_forecourt,south_courtyard,yuanping_paving,
                          zhizhi_courtyard,liyan_entry,zhiyuan_entry,zhili_a_entry,zhigong_parking,
                          zhigong_forecourt,zhigong_stair_site,stone_plaza,south_pool,dujuan,gate_walk,ancient_stone_reserve,kaifeng_forecourt,
                          qiaoyuan_walkway.buffer(.15),qiaoyuan_brick.buffer(.15),
                          box(-438,518,-417,598),box(-298,483,-251,513)))
    reserved=reserved.union(zhounan_portal['reserved']).union(zhai_courtyard)
    bridge_space=unary_union([g.buffer((7.5 if key=='way/574956548' else
                                        3.5 if key=='way/776358616' else 2.0)+3.0)
                              for key,t,g in roadways if t.get('bridge') not in (None,'no')])
    counts,walking_surface,candidates,lamps=campus_road_network.build_road_environment(
        campus=campus,vehicle_roads=ground_vehicle_roads,roadways=roadways,vehicle_surface=vehicle_surface,
        buildings=buildings,waters=waters,pitches=pitches,reserved=reserved,bridge_clearance=bridge_space,
        junctions=road_junctions,road_width=road_width,pedestrian_types=PEDESTRIAN_ROADS,
        drape=drape_on_terrain,mesh_type=MeshWriter,height_at=rendered_terrain_height,
        beam=beam,ring_ids=CAMPUS_RING_IDS,assets=ASSETS,data=DATA)
    return counts,walking_surface,candidates,lamps,reserved


def make_lide_walk():
    """把照片启发的步道改为独立、贴地的砖面和浅色边带。"""
    axis=LineString(LIDE_WALK_POINTS)
    booth=Polygon(LIDE_WALK['gate_booth_footprint_m']).buffer(.15)
    blocked=unary_union([vehicle_surface.buffer(.12),booth] + [g.buffer(.15) for _,_,g in buildings])
    candidate=axis.buffer(LIDE_WALK['width_m_estimate']/2,cap_style='round',join_style='round').difference(blocked)
    # 与车行线或门岗相交时仅保留最长连续步道，丢弃裁切产生的孤立碎片。
    paving=max(polygons(candidate),key=lambda part:part.area)
    border=paving.buffer(LIDE_WALK['edge_width_m_estimate']).difference(paving).difference(blocked)
    bricks,edge=MeshWriter(),MeshWriter()
    drape_on_terrain(bricks,paving,.065,lambda x,z:(x/2.4,z/2.4))
    drape_on_terrain(edge,border,.075,lambda x,z:(x/2.4,z/2.4))
    return {'paving':bricks.save(ASSETS/'campus-lide-walk.mesh'),
            'edge':edge.save(ASSETS/'campus-lide-walk-edge.mesh')}


def make_qiaoyuan_garden():
    """根据 2022 年照片近似重建乔相阁前的森林校巴花园。"""
    walk,brick,mulch,frame,words,wood,tops,plants,plants_light,stones=(MeshWriter() for _ in range(10))
    def local_uv(x,z):
        d=np.asarray((x,z))-qiaoyuan_origin
        return (float(np.dot(d,qiaoyuan_u))/4,float(np.dot(d,qiaoyuan_v))/4)
    def brick_uv(x,z):
        d=np.asarray((x,z))-qiaoyuan_origin
        return (float(np.dot(d,qiaoyuan_u))/3.2,float(np.dot(d,qiaoyuan_v))/1.6)
    # 路边红砖与园内浅赭色人字形路网分开绘制，并与地图人行道相交。
    drape_on_terrain(brick,qiaoyuan_brick,.068,brick_uv)
    drape_on_terrain(walk,qiaoyuan_walkway,.083,local_uv)
    beds=unary_union([affinity.scale(Point(*qiaoyuan_xy(u,v)).buffer(1,resolution=16),xfact=rx,yfact=rz)
                      for u,v,rx,rz in ((-13,17,2.3,3.0),(-6,19,2.8,2.0),
                                        (9,20.5,2.4,2.1),(13,12,2.4,1.7))])
    beds=beds.intersection(qiaoyuan_garden_site).difference(qiaoyuan_walkway.buffer(.35)).difference(qiaoyuan_brick)
    drape_on_terrain(mulch,beds,.055,local_uv)

    # 浅蓝绿色管框依据照片中的校巴外轮廓概括；尺寸和精确位置未经实测。
    center=qiaoyuan_xy(-11,9.6)
    tangent=qiaoyuan_u*6.5+qiaoyuan_v*5.7
    tangent/=np.linalg.norm(tangent)
    side=np.asarray((-tangent[1],tangent[0]))
    left,right=center-side*2.35,center+side*2.35
    yl=rendered_terrain_height(*left)
    yr=rendered_terrain_height(*right)
    top=max(yl,yr)+2.9
    def pt(p,y):return (p[0],y,p[1])
    outline=[pt(left,yl+.05),pt(left,top-.22),pt(left+side*.20,top),
             pt(right-side*.48,top),pt(right-side*.12,top-.18),pt(right,top-.42),pt(right,yr+.05)]
    for a,b in zip(outline[:-1],outline[1:]):
        round_beam(frame,a,b,.065)
    # 屋顶上是独立的镂空文字，透明部分不遮挡后方树木。
    a=center-side*2.05
    b=center+side*1.65
    normal=(-tangent[0],0,-tangent[1])
    words.quad(pt(a,top+.03),pt(b,top+.03),pt(b,top+.58),pt(a,top+.58),normal,
               ((0,0),(1,0),(1,1),(0,1)))
    # 小型内部窗框与示意牌只表达照片中能辨认的轮廓。
    panel_center=center+side*1.25+tangent*2.9
    panel_left,panel_right=panel_center-side*.62,panel_center+side*.62
    panel_base=max(rendered_terrain_height(*panel_left),rendered_terrain_height(*panel_right))
    frame_points=[pt(panel_left,panel_base+.12),pt(panel_left,panel_base+1.62),
                  pt(panel_left+side*.12,panel_base+1.78),pt(panel_right-side*.12,panel_base+1.78),
                  pt(panel_right,panel_base+1.62),pt(panel_right,panel_base+.12)]
    for a,b in zip(frame_points[:-1],frame_points[1:]):
        round_beam(frame,a,b,.045)

    # 木桩及木凳按照片布置在园路侧边，逐个贴合局部地面。
    for u,v,r,h in ((-14,13,.32,.38),(-12.7,15.5,.24,.52),(-9,17.4,.37,.42),
                    (-5,20.4,.27,.34),(5.8,20.6,.38,.42),(11,19.4,.25,.55),
                    (14,16.1,.35,.38),(14.7,12.5,.27,.45),(-15,20,.24,.42)):
        p=qiaoyuan_xy(u,v)
        if qiaoyuan_walkway.distance(Point(p))<r+.25:
            continue
        y=rendered_terrain_height(*p)+.04
        round_beam(wood,(p[0],y,p[1]),(p[0],y+h,p[1]),r,12)
        tops.horizontal(Point(*p).buffer(r,resolution=8),y+h+.005)
    for u,v in ((-13.5,19.0),(8.2,11.9),(13.6,22.2)):
        p=qiaoyuan_xy(u,v)
        if qiaoyuan_walkway.distance(Point(p))<1.15:
            continue
        ya=rendered_terrain_height(*p)+.5
        a,b=p-qiaoyuan_u*.82,p+qiaoyuan_u*.82
        beam(wood,(a[0],ya,a[1]),(b[0],ya,b[1]),.10)
        for end in (a,b):
            ground=rendered_terrain_height(*end)
            round_beam(frame,(end[0],ground+.03,end[1]),(end[0],ya-.04,end[1]),.035,8)

    # 低矮叶片、草丛与卵石用轻量几何重建，避免将现场照片贴到地面。
    allowed=(qiaoyuan_garden_site.difference(qiaoyuan_walkway.buffer(.65))
             .difference(qiaoyuan_brick.buffer(.35)).difference(vehicle_surface.buffer(.8))
             .difference(qiaoxiang_tower.buffer(1)))
    allowed=prep(allowed)
    rng=random.Random(20221214)
    for i in range(220):
        u,v=rng.uniform(-17,17),rng.uniform(7.8,24)
        p=qiaoyuan_xy(u,v)
        if not allowed.covers(Point(p)):
            continue
        y=rendered_terrain_height(*p)+.06
        mesh=plants_light if i%3==0 else plants
        length=rng.uniform(.22,.60)
        for k in range(7):
            angle=math.tau*k/7+rng.uniform(-.18,.18)
            direction=np.asarray((math.cos(angle),math.sin(angle)))
            lateral=np.asarray((-direction[1],direction[0]))
            mid=p+direction*length*.48
            tip=p+direction*length
            width=length*.20
            base=(p[0],y,p[1])
            left=(mid[0]+lateral[0]*width,y+length*.30,mid[1]+lateral[1]*width)
            right=(mid[0]-lateral[0]*width,y+length*.30,mid[1]-lateral[1]*width)
            end=(tip[0],y+length*.45,tip[1])
            mesh.triangle(base,left,end,(0,1,0),((0,0),(1,0),(1,1)))
            mesh.triangle(base,end,right,(0,1,0),((0,0),(1,1),(0,1)))
    for i in range(55):
        t=i/54
        u=-15+5.2*t+rng.uniform(-.65,.65)
        v=17+5.3*t+rng.uniform(-.55,.55)
        p=qiaoyuan_xy(u,v)
        if not allowed.covers(Point(p)):
            continue
        y=rendered_terrain_height(*p)+.04
        r=rng.uniform(.07,.18)
        round_beam(stones,(p[0],y,p[1]),(p[0],y+r*.65,p[1]),r,8)
        stones.horizontal(Point(*p).buffer(r,resolution=5),y+r*.66)
    files=((walk,'walk'),(brick,'brick'),(mulch,'mulch'),(frame,'frame'),(words,'words'),
           (wood,'wood'),(tops,'wood-tops'),(plants,'plants'),(plants_light,'plants-light'),(stones,'stones'))
    return {name:mesh.save(ASSETS/f'campus-qiaoyuan-garden-{name}.mesh') for mesh,name in files}


def make_xiyuan_shops():
    """在阁区坡路和乔相阁对面布置近似店面；店铺顺序采用使用者提供的信息。"""
    (glazing,frames,signs,canopy,shell,roof,paving,bicycle,opposite_paving,
     membrane,awning_frame,base_dark,base_brick,red_paving,plant_bed,plant_leaves,
     manner_wall) = (MeshWriter() for _ in range(17))
    manifest=json.loads((DATA/'qiaoyuan-shops.json').read_text(encoding='utf-8'))
    indices={name:i for i,name in enumerate(('manner','youjian','fruit','seven','tea','print',
                                             'repair','sf','lawson','barber','glasses','szu'))}

    def storefront(a,b,outward,shop_id):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        outward=np.asarray(outward,dtype=float)
        outward/=np.linalg.norm(outward)
        direction=b-a
        length=float(np.linalg.norm(direction))
        direction/=length
        # 店门落在中部地面，立面维持水平；两端随坡地以短基座收口。
        ground_a=rendered_terrain_height(*a)
        ground_b=rendered_terrain_height(*b)
        middle=(a+b)*.5
        y=rendered_terrain_height(*middle)+.06
        def p(t,h,offset=.10):
            q=a+direction*t+outward*offset
            return (float(q[0]),float(y+h),float(q[1]))
        n=(float(outward[0]),0,float(outward[1]))
        pa=a+outward*.12
        pb=b+outward*.12
        shell.quad((pa[0],min(ground_a-.08,y+.08),pa[1]),(pb[0],min(ground_b-.08,y+.08),pb[1]),
                   (pb[0],y+.09,pb[1]),(pa[0],y+.09,pa[1]),n,((0,0),(length/4,0),(length/4,1),(0,1)))
        # 统一图集的每一行分别绘制品牌文字，不将未知店名冒充为已核实信息。
        index=indices[shop_id]
        v0=1-(index+1)/12+.002
        v1=1-index/12-.002
        screen_right=np.asarray((outward[1],-outward[0]))
        u0,u1=((.002,.998) if np.dot(direction,screen_right)>0 else (.998,.002))
        glazing.quad(p(.12,.08),p(length-.12,.08),p(length-.12,2.42),p(.12,2.42),n,
                     ((u0,v0),(u1,v0),(u1,v1),(u0,v1)))
        signs.quad(p(0,2.48,.16),p(length,2.48,.16),p(length,3.30,.16),p(0,3.30,.16),n,
                   ((u0,v0),(u1,v0),(u1,v1),(u0,v1)))
        if shop_id=='manner':
            manner_wall.quad(p(0,3.30,.08),p(length,3.30,.08),p(length,8.0,.08),p(0,8.0,.08),n,
                             ((0,0),(2,0),(2,1),(0,1)))
        # 实景照片中 7-11 与果果切的落地窗下部为实体墙，门口则落到地面。
        if shop_id in ('manner','fruit','seven'):
            wall=base_brick if shop_id=='fruit' else base_dark
            for l,r in ((.12,length*.40),(length*.60,length-.12)):
                wall.quad(p(l,.08,.23),p(r,.08,.23),p(r,.52,.23),p(l,.52,.23),n,
                          ((0,0),((r-l)/2,0),((r-l)/2,.5),(0,.5)))
        for t in np.linspace(0,length,max(3,round(length/2.1)+1)):
            beam(frames,p(t,.03,.20),p(t,2.43,.20),.035)
        for t in (length*.47,length*.53):
            beam(frames,p(t,.85,.25),p(t,1.35,.25),.025)
        for h in (.08,2.40,2.48,3.30):
            beam(frames,p(0,h,.20),p(length,h,.20),.045)
        # 711 一侧由下方连续膜棚覆盖；其余店面只示意招牌外挑的檐口。
        if shop_id not in ('youjian','fruit','seven','tea'):
            canopy.quad(p(0,2.46,.10),p(length,2.46,.10),p(length,2.46,.80),p(0,2.46,.80),
                        (0,1,0),((0,0),(1,0),(1,1),(0,1)))
            beam(frames,p(0,2.45,.80),p(length,2.45,.80),.045)
        return y

    # 7-11 的 OSM 点位在文杏阁与丁香阁之间；其余店面间距据顺序近似。
    for shop in manifest['road_side']:
        z0,z1=shop['start_z'],shop['end_z']
        def front_x(z): return -433.1375+(z-519.8867)*.00328+.15
        storefront((front_x(z0),z0),(front_x(z1),z1),(1,0),shop['id'])
    # 接近修车铺的低层附属部分补足街墙，范围为近似而非地图测绘。
    repair_base=rendered_terrain_height(-433,585)
    extrude(shell,roof,box(-437.0,582.6,-432.92,594.8),repair_base-2,repair_base+3.55)

    sf=manifest['across_from_repair']
    def sf_x(z): return -419.6+(z-587.8)*.064-.13
    storefront((sf_x(sf['start_z']),sf['start_z']),(sf_x(sf['end_z']),sf['end_z']),(-1,0),sf['id'])

    # 乔相阁对面四家店排在海志楼临路侧，左右顺序没有现场证据，暂作占位。
    a=np.asarray((-293.71,508.53)); b=np.asarray((-253.68,498.88))
    tangent=(b-a)/np.linalg.norm(b-a)
    normal=np.asarray((.234,.972))
    for i,shop in enumerate(manifest['opposite_qiaoxiang']):
        p0=a+tangent*(.7+i*9.9)
        p1=a+tangent*(.7+(i+1)*9.9-.15)
        storefront(p0,p1,normal,shop['id'])

    # 两侧贴地砖带和车道相交处裁切，避免把路口封死。
    opposite_walk=Polygon((a,b,b+normal*3.8,a+normal*3.8))
    strips=unary_union((box(-432.9,520,-428.55,596),box(-423.2,587.7,-419.45,597)))
    strips=clean(strips.difference(vehicle_surface.buffer(.08)))
    drape_on_terrain(paving,strips,.07,lambda x,z:(x/2,z/2))
    opposite_walk=clean(opposite_walk.difference(vehicle_surface.buffer(.08)))
    drape_on_terrain(opposite_paving,opposite_walk,.07,lambda x,z:(x/4,z/4))
    # Manner 前的暖红色地面与浅色花槽可从 2026 年现场照片辨认。
    mat=clean(box(-432.7,522.5,-429.65,531.4).intersection(strips))
    drape_on_terrain(red_paving,mat,.087,lambda x,z:(x/3,z/1.5))

    # 白色连续张拉膜及分叉钢管：轮廓参考 2023、2026 年照片，尺寸为估计。
    start_z,end_z=532.2,575.0
    bays=7
    spacing=(end_z-start_z)/bays
    inner_x,outer_x=-432.85,-429.38
    def canopy_point(bay,u,v):
        x=inner_x+(outer_x-inner_x)*u
        z=start_z+spacing*(bay+v)
        ground=rendered_terrain_height(x,z)
        height=3.55+.12*(1-u)+.48*math.sin(math.pi*v)*u-.20*math.sin(math.pi*u)
        return (x,ground+height,z)
    for bay in range(bays):
        for ix in range(4):
            for iz in range(8):
                u0,u1=ix/4,(ix+1)/4
                v0,v1=iz/8,(iz+1)/8
                p0,p1=canopy_point(bay,u0,v0),canopy_point(bay,u1,v0)
                p2,p3=canopy_point(bay,u1,v1),canopy_point(bay,u0,v1)
                membrane.quad(p0,p1,p2,p3,surface_normal(p0,p1,p2),((u0,v0),(u1,v0),(u1,v1),(u0,v1)))
        for u in (0,1):
            for step in range(8):
                round_beam(awning_frame,canopy_point(bay,u,step/8),canopy_point(bay,u,(step+1)/8),.028,6)
    for index in range(bays+1):
        z=start_z+spacing*index
        base=rendered_terrain_height(outer_x,z)
        joint=(outer_x,base+3.55,z)
        round_beam(awning_frame,(outer_x,base+.05,z),joint,.075,8)
        if index>0:
            round_beam(awning_frame,joint,canopy_point(index-1,1,.5),.035,8)
        if index<bays:
            round_beam(awning_frame,joint,canopy_point(index,1,.5),.035,8)
        inner=(inner_x,rendered_terrain_height(inner_x,z)+3.67,z)
        round_beam(awning_frame,joint,inner,.035,8)

    # 海志楼店前有低矮绿植带；留出通向各店门口的缺口。
    beds=[]
    for left,right in ((.5,5.5),(9.5,15.3),(19.5,25.2),(29.4,35.3),(39.2,40.7)):
        corners=[a+tangent*t+normal*s for t,s in ((left,2.75),(right,2.75),(right,3.55),(left,3.55))]
        beds.append(Polygon(corners))
    beds=clean(unary_union(beds).intersection(opposite_walk))
    drape_on_terrain(plant_bed,beds,.085,lambda x,z:(x,z))
    rng=random.Random(6601)
    for i in range(850):
        along=rng.uniform(.6,40.6)
        across=rng.uniform(2.83,3.47)
        q=a+tangent*along+normal*across
        if not beds.covers(Point(*q)):
            continue
        y=rendered_terrain_height(*q)+.095
        length=rng.uniform(.23,.48)
        for angle in (0,math.pi/3,2*math.pi/3):
            dx,dz=math.cos(angle)*length,math.sin(angle)*length
            plant_leaves.triangle((q[0]-dx*.18,y,q[1]-dz*.18),
                                  (q[0]+dx,y+length*.65,q[1]+dz),
                                  (q[0]+dz*.14,y,q[1]-dx*.14),(0,1,0),((0,0),(1,1),(0,1)))
    # 单车是修车铺位置的示意陈设，放在人行带内，不阻断车道。
    bike_x,bike_z=-431.45,589.0
    bike_ground=rendered_terrain_height(bike_x,bike_z)+.04
    centers=[(bike_x,bike_ground+.34,bike_z+offset) for offset in (-.65,.65)]
    for cx,cy,cz in centers:
        for i in range(18):
            a0,a1=math.tau*i/18,math.tau*(i+1)/18
            round_beam(bicycle,(cx,cy+.32*math.sin(a0),cz+.32*math.cos(a0)),
                       (cx,cy+.32*math.sin(a1),cz+.32*math.cos(a1)),.018,6)
    hub=(bike_x,bike_ground+.42,bike_z)
    seat=(bike_x,bike_ground+.98,bike_z-.18)
    handle=(bike_x,bike_ground+1.06,bike_z+.48)
    for a0,b0 in ((centers[0],hub),(centers[1],hub),(centers[0],seat),(seat,hub),
                  (seat,handle),(handle,centers[1])):
        round_beam(bicycle,a0,b0,.025,6)
    beam(bicycle,(bike_x-.13,seat[1]+.03,seat[2]),(bike_x+.13,seat[1]+.03,seat[2]),.03)
    beam(bicycle,(bike_x-.20,handle[1],handle[2]),(bike_x+.20,handle[1],handle[2]),.025)
    items=(('glass',glazing),('frames',frames),('signs',signs),('canopy',canopy),
           ('shell',shell),('roof',roof),('paving',paving),('bicycle',bicycle),
           ('opposite-paving',opposite_paving),('membrane',membrane),('awning-frame',awning_frame),
           ('base-dark',base_dark),('base-brick',base_brick),('red-paving',red_paving),
           ('plant-bed',plant_bed),('plant-leaves',plant_leaves),('manner-wall',manner_wall))
    return {name:mesh.save(ASSETS/f'campus-xiyuan-shops-{name}.mesh') for name,mesh in items}


def make_bridges(walking_surface):
    # 两桥的中心线由 OSM 提供。高度未见可公开的竣工图：用连续纵坡及公路净空近似。
    # 召南桥参考实拍的白色顶棚；周南桥按学校记载区分较宽的通道和混凝土桥身。
    deck, rails, roof, support, roadway, markings, lower, truss, glass = (MeshWriter() for _ in range(9))
    zhounan_yellow,zhounan_pots,zhounan_plants,zhounan_ceiling = (MeshWriter() for _ in range(4))
    portal_treads,portal_walls = MeshWriter(),MeshWriter()
    baishi = [g for _, t, g in roadways if t.get('name') == '白石路' and t.get('bridge') in (None, 'no')]
    major_road = prep(unary_union(baishi).buffer(13))
    # 桥柱应避开地面车道和人行带；优先沿原桥轴线移到附近的可用位置。
    support_exclusion=prep(unary_union([vehicle_surface.buffer(.9),walking_surface.buffer(.9)]
                                        +[g.buffer(.2) for _,_,g in buildings]))
    support_positions=[]

    def support_location(items,index,minimum_height,local=False):
        for delta in (0,-1,1,-2,2,-3,3):
            j=index+delta
            if not 0<=j<len(items):
                continue
            item=items[j]
            if local and item[6]:
                continue
            x,z,y=item[1],item[2],item[3]
            p=Point(x,z)
            if (major_road.covers(p) or support_exclusion.covers(p)
                    or any(math.hypot(x-a,z-b)<3.0 for a,b in support_positions)):
                continue
            ground=rendered_terrain_height(x,z)
            if y-ground<=minimum_height:
                continue
            support_positions.append((x,z))
            return item,ground
        return None

    profiles = []
    main_sections = []
    lower_profile = None
    for osm_id, width in (('776358616', 7.0), ('574956548', 15.0)):
        center = LineString(way_points(ways[osm_id]))
        length = center.length
        crosses = [p for road in baishi for p in [center.intersection(road)] if p.geom_type == 'Point']
        crossing = float(np.mean([center.project(p) for p in crosses])) if crosses else length*.5
        distances = list(np.arange(0.0, length, 4.0)) + [length]
        ends = [surface_height(*center.coords[0])+.35, surface_height(*center.coords[-1])+.35]
        if osm_id == '574956548':
            ends[-1] = zhounan_portal['upper_ground']+.35
        road_height = max([surface_height(p.x,p.y) for p in crosses] or [min(ends)])
        crest = max(road_height+6.2, max(ends)-1.5)
        if osm_id == '574956548':
            # 按使用者对现场桥高的反馈抬高 1 号桥；并非实测高程。
            # 两端仍使用同一连续纵坡回落，双层结构及桥墩随之整体调整。
            crest += 2.5
        section = []
        for dist in distances:
            p = center.interpolate(dist)
            before = center.interpolate(max(0, dist-.6))
            after = center.interpolate(min(length, dist+.6))
            tangent = np.array((after.x-before.x, after.y-before.y))
            tangent /= max(np.linalg.norm(tangent), .0001)
            normal = np.array((-tangent[1], tangent[0]))
            # 核心跨路段保持平稳，两头使用平滑曲线与各自地形衔接。
            ramp_end = max(crossing-26.0, 1.0)
            ramp_start = min(crossing+26.0, length-1.0)
            if dist < ramp_end:
                t = dist / ramp_end
                y = ends[0] + (crest-ends[0]) * t*t*(3-2*t)
            elif dist > ramp_start:
                t = (dist-ramp_start) / (length-ramp_start)
                y = crest + (ends[1]-crest) * t*t*(3-2*t)
            else:
                y = crest
            section.append((dist, p.x, p.y, y, normal[0], normal[1]))

        def edge(item, offset, up=0.0):
            _, x, z, y, nx, nz = item
            return (x+nx*offset, y+up, z+nz*offset)

        # OSM 同一位置另有 layer=1 的人行路径；与 layer=2 的车行桥分开。
        # 学校记载 1 号桥为钢桁腹组合梁，实拍可见上下板与斜向钢管。
        # 上下板间距、渐变长度和管径均为照片约束下的估计。
        lower_path = LineString(way_points(ways['776358646']))
        # 楼梯口和南端人行线都会从桥板侧面进入；护栏要给两条路线留开口。
        lower_stair_opening = unary_union((LineString(way_points(ways['776358647'])).buffer(2.1),
                                          lower_path.buffer(1.8)))
        if osm_id == '574956548':
            lower_stair_opening = lower_stair_opening.union(zhounan_portal['terrace'].buffer(.25))
        def lower_depth(item):
            dist,x,z,y,_,_ = item
            if osm_id != '574956548' or dist < 180 or lower_path.distance(Point(x,z)) > 9:
                return 0.0
            if dist >= zhounan_portal['transition']:
                # 斋区端人行层到达侧开口后结束，上层沿用原有高程。
                return y-zhounan_portal['floor'](dist) if dist<=zhounan_portal['opening_end'] else 0.0
            # 桥下人行层在匝道旁直接进入独立的下层结构；不能先与车道合并，
            # 否则会把相邻楼梯入口错误抬到车行层。离开高架或遇到地形时再收合。
            return max(0.0,min(3.4,y-surface_height(x,z)-.35))

        def planter(position):
            # 桥下近照可见白色花盆和丛生绿叶；尺寸、株数按照片近似。
            x,y,z=position
            segments=10
            for k in range(segments):
                a,b=2*math.pi*k/segments,2*math.pi*(k+1)/segments
                ca,sa,cb,sb=math.cos(a),math.sin(a),math.cos(b),math.sin(b)
                p=(x+.20*ca,y,z+.20*sa); q=(x+.20*cb,y,z+.20*sb)
                r=(x+.27*cb,y+.48,z+.27*sb); s=(x+.27*ca,y+.48,z+.27*sa)
                n=(math.cos((a+b)*.5),.14,math.sin((a+b)*.5))
                zhounan_pots.quad(p,q,r,s,n,((0,0),(1,0),(1,1),(0,1)))
            for k in range(9):
                a=2*math.pi*k/9
                radial=np.asarray((math.cos(a),0,math.sin(a)))
                side=np.asarray((-math.sin(a),0,math.cos(a)))
                base=np.asarray((x,y+.44,z))+radial*.03
                tip=np.asarray((x,y+.97+.11*math.sin(k*2.3),z))+radial*(.25+.06*math.cos(k))
                left=base-side*.065; right=base+side*.065
                zhounan_plants.triangle(left,right,tip,surface_normal(left,right,tip),((0,0),(1,0),(.5,1)))

        for a, b in zip(section[:-1], section[1:]):
            segment = b[0]-a[0]
            al, ar = edge(a,width*.5), edge(a,-width*.5)
            bl, br = edge(b,width*.5), edge(b,-width*.5)
            deck.quad(al,bl,br,ar,(0,1,0),((0,0),(segment/4,0),(segment/4,1),(0,1)))
            if osm_id == '574956548':
                # 车行桥板底面和横梁应能从下层看见；上层原模型只有单面地板。
                zhounan_ceiling.quad(edge(a,-width*.5,-.8),edge(b,-width*.5,-.8),
                                     edge(b,width*.5,-.8),edge(a,width*.5,-.8),(0,-1,0),
                                     ((0,0),(segment/4,0),(segment/4,1),(0,1)))
            for side in (-1,1):
                normal = (a[4]*side, 0, a[5]*side)
                deck.quad(edge(a,side*width*.5,-.8),edge(b,side*width*.5,-.8),
                          edge(b,side*width*.5),edge(a,side*width*.5),normal,((0,0),(1,0),(1,1),(0,1)))
                if osm_id == '776358616':
                    for level in (.22, 1.05):
                        beam(rails,edge(a,side*(width*.5-.12),level),edge(b,side*(width*.5-.12),level),.045)
                    # 照片显示桥侧为玻璃板和金属扶手；透明板在主渲染结束后单独绘制。
                    glass.quad(edge(a,side*(width*.5-.10),.18),edge(b,side*(width*.5-.10),.18),
                               edge(b,side*(width*.5-.10),1.02),edge(a,side*(width*.5-.10),1.02),normal,
                               ((0,0),(1,0),(1,1),(0,1)))
                else:
                    # 2024 年照片中的上层道路有实心护墙及黄扶手，非两条悬空白杆。
                    outer,inner=side*width*.5,side*(width*.5-.42)
                    deck.quad(edge(a,inner,0),edge(b,inner,0),edge(b,inner,.70),edge(a,inner,.70),
                              (-normal[0],0,-normal[2]),((0,0),(1,0),(1,1),(0,1)))
                    deck.quad(edge(a,outer,0),edge(b,outer,0),edge(b,outer,.70),edge(a,outer,.70),
                              normal,((0,0),(1,0),(1,1),(0,1)))
                    deck.quad(edge(a,outer,.70),edge(b,outer,.70),edge(b,inner,.70),edge(a,inner,.70),
                              (0,1,0),((0,0),(1,0),(1,1),(0,1)))
                    beam(zhounan_yellow,edge(a,side*(width*.5-.19),1.05),
                         edge(b,side*(width*.5-.19),1.05),.045)
            if osm_id == '574956548':
                # 上层按照片表现双向车道、白边线和双黄中心线，现时通车状态未核实。
                half = width*.5-.45
                roadway.quad(edge(a,half,.035),edge(b,half,.035),edge(b,-half,.035),edge(a,-half,.035),
                             (0,1,0),((0,0),(1,0),(1,1),(0,1)))
                for side in (-1,1):
                    outer=side*(half-.16)
                    markings.quad(edge(a,outer-.07,.055),edge(b,outer-.07,.055),
                                  edge(b,outer+.07,.055),edge(a,outer+.07,.055),
                                  (0,1,0),((0,0),(1,0),(1,1),(0,1)))
                    center_offset=side*.25
                    zhounan_yellow.quad(edge(a,center_offset-.06,.06),edge(b,center_offset-.06,.06),
                                        edge(b,center_offset+.06,.06),edge(a,center_offset+.06,.06),
                                        (0,1,0),((0,0),(1,0),(1,1),(0,1)))
                da,db = lower_depth(a),lower_depth(b)
                if min(da,db) > .20:
                    wa = 3.5+4.0*min(1,da/2.4)
                    wb = 3.5+4.0*min(1,db/2.4)
                    lower.quad(edge(a,wa,-da),edge(b,wb,-db),edge(b,-wb,-db),edge(a,-wa,-da),
                               (0,1,0),((0,0),(1,0),(1,1),(0,1)))
                    lower.quad(edge(a,-wa,-da-.42),edge(b,-wb,-db-.42),edge(b,wb,-db-.42),edge(a,wa,-da-.42),
                               (0,-1,0),((0,0),(1,0),(1,1),(0,1)))
                    for side in (-1,1):
                        normal = (a[4]*side,0,a[5]*side)
                        lower.quad(edge(a,side*wa,-da-.42),edge(b,side*wb,-db-.42),
                                   edge(b,side*wb,-db),edge(a,side*wa,-da),normal,
                                   ((0,0),(1,0),(1,1),(0,1)))
                        ra,rb=edge(a,side*(wa-.15),-da+1.05),edge(b,side*(wb-.15),-db+1.05)
                        if not lower_stair_opening.intersects(LineString(((ra[0],ra[2]),(rb[0],rb[2])))):
                            beam(rails,ra,rb,.045)
                        # 不锈钢竖杆比原先每 8 米一根立柱更接近照片；楼梯支路处保留开口。
                        for j in range(max(1,int(segment/.36))):
                            t=(j+.5)/max(1,int(segment/.36))
                            p=np.asarray(edge(a,side*(wa-.15),-da))*(1-t)+np.asarray(edge(b,side*(wb-.15),-db))*t
                            if not lower_stair_opening.covers(Point(p[0],p[2])):
                                beam(rails,p,p+np.asarray((0,1.02,0)),.018)
            elif crossing-96 <= (a[0]+b[0])*.5 <= crossing+96:
                # 召南桥照片中的浅弧形白色雨棚，分条近似曲面，而非人字形屋顶。
                for k in range(-4,4):
                    u,v = (width*.5+.15)*k/4,(width*.5+.15)*(k+1)/4
                    hu,hv = 3.32+.18*(1-(k/4)**2),3.32+.18*(1-((k+1)/4)**2)
                    roof.quad(edge(a,u,hu),edge(b,u,hu),edge(b,v,hv),edge(a,v,hv),
                              (0,1,0),((0,0),(1,0),(1,1),(0,1)))

        for i, item in enumerate(section):
            dist, x, z, y, nx, nz = item
            if i % 2 == 0 and osm_id == '776358616':
                for side in (-1,1):
                    offset = side*(width*.5-.12)
                    beam(rails,edge(item,offset,.10),edge(item,offset,1.10),.055)
            if osm_id == '776358616' and crossing-96 <= dist <= crossing+96:
                for side in (-1,1):
                    offset = side*(width*.5-.12)
                    beam(rails,edge(item,offset,0),edge(item,offset,3.32),.09)
                if i % 2 == 0:
                    beam(rails,edge(item,-width*.5,3.32),edge(item,0,3.50),.055)
                    beam(rails,edge(item,0,3.50),edge(item,width*.5,3.32),.055)
            if osm_id == '574956548' and i+2 < len(section) and i % 2 == 0:
                mid,last = section[i+1],section[i+2]
                da,dm,db = lower_depth(item),lower_depth(mid),lower_depth(last)
                if da > .20:
                    half = 3.5+4.0*min(1,da/2.4)
                    for side in (-1,1):
                        p=edge(item,side*(half-.15),-da)
                        if not lower_stair_opening.covers(Point(p[0],p[2])):
                            beam(rails,p,edge(item,side*(half-.15),-da+1.05),.045)
                if min(da,dm,db) >= 2.7:
                    # 下层顶部间隔布置横梁及两排盆栽，避免遮挡中央通道。
                    for cross_section in (item,mid):
                        beam(zhounan_ceiling,edge(cross_section,-5.35,-.92),edge(cross_section,5.35,-.92),.12)
                    for side in (-1,1):
                        offset = side*5.5
                        round_beam(truss,edge(item,offset,-da),edge(mid,offset,-.8),.19)
                        round_beam(truss,edge(mid,offset,-.8),edge(last,offset,-db),.19)
                        pot=edge(item,side*4.85,-da+.03)
                        if not lower_stair_opening.covers(Point(pot[0],pot[2])):
                            planter(pot)
            if osm_id == '574956548' and i % 2 == 0:
                for side in (-1,1):
                    offset=side*(width*.5-.19)
                    beam(zhounan_yellow,edge(item,offset,.70),edge(item,offset,1.05),.045)
            # 桥墩避开车道和步行带，落到同一地形网格上。
            if i % 8 == 4:
                candidate=support_location(section,i,2.2)
                if candidate:
                    pillar,ground=candidate
                    _,px,pz,py,_,_=pillar
                    depth = lower_depth(pillar)
                    if osm_id == '574956548' and depth > 2.7:
                        for side in (-1,1):
                            beam(support,(px,ground-.2,pz),edge(pillar,side*4.1,-depth-.42),.54)
                    else:
                        beam(support,(px,ground-.2,pz),(px,py-.75,pz),.50 if width < 10 else .82)
        profiles.append({'osm': osm_id, 'length_m': round(length,1), 'width_m': width,
                         'crossing_m': round(crossing,1), 'road_height_estimate_m': round(road_height,1),
                         'deck_height_estimate_m': round(crest,1), 'ends_m': [round(v,1) for v in ends],
                         'min_height_above_terrain_m': round(min(v[3]-surface_height(v[1],v[2]) for v in section),2),
                         'max_grade': round(max(abs(a[3]-b[3])/(b[0]-a[0]) for a,b in zip(section[:-1],section[1:])),3)})
        if osm_id == '574956548':
            lower_profile = ([item[0] for item in section],[lower_depth(item) for item in section])
            profiles[-1]['lower_path_osm'] = '776358646'
            profiles[-1]['max_deck_separation_estimate_m'] = round(max(lower_profile[1]),2)
        main_sections.append((center, section))

    # 地图中的南侧接线与周南桥共用端点；其是否对车辆开放尚未核实。
    # 在模型中补齐连续路面，端点高度与现有桥面、地面路网分别对齐。
    ramp=LineString(way_points(ways['611264728']))
    main_start=main_sections[1][1][0]
    ramp_sections=[]
    ramp_distances=list(np.arange(0.0,ramp.length,4.0))+[ramp.length]
    for distance in ramp_distances:
        p=ramp.interpolate(distance)
        before=ramp.interpolate(max(0,distance-.5))
        after=ramp.interpolate(min(ramp.length,distance+.5))
        tx,tz=after.x-before.x,after.y-before.y
        norm=max(.001,math.hypot(tx,tz))
        tangent_normal=np.asarray((-tz/norm,tx/norm))
        t=distance/ramp.length
        # 尾部法线使用主桥首截面，避免两段路边界出现细缝。
        n=(1-t)*tangent_normal+t*np.asarray(main_start[4:6])
        n/=max(.001,np.linalg.norm(n))
        y=rendered_terrain_height(p.x,p.y)+.035
        y+=(main_start[3]-rendered_terrain_height(*ramp.coords[-1])-.035)*t
        ramp_sections.append((distance,p.x,p.y,y,n[0],n[1]))
    for a,b in zip(ramp_sections[:-1],ramp_sections[1:]):
        wa=3.3+4.2*(a[0]/ramp.length)
        wb=3.3+4.2*(b[0]/ramp.length)
        segment=b[0]-a[0]
        deck.quad(edge(a,wa),edge(b,wb),edge(b,-wb),edge(a,-wa),
                  (0,1,0),((0,0),(segment/4,0),(segment/4,1),(0,1)))
        roadway.quad(edge(a,wa-.25,.025),edge(b,wb-.25,.025),
                     edge(b,-wb+.25,.025),edge(a,-wa+.25,.025),
                     (0,1,0),((0,0),(1,0),(1,1),(0,1)))
        for side in (-1,1):
            # 线型按已有桥面做连续衔接，不声称与现场标线完全一致。
            markings.quad(edge(a,side*(wa-.45)-.06,.045),edge(b,side*(wb-.45)-.06,.045),
                          edge(b,side*(wb-.45)+.06,.045),edge(a,side*(wa-.45)+.06,.045),
                          (0,1,0),((0,0),(1,0),(1,1),(0,1)))
            zhounan_yellow.quad(edge(a,side*.25-.06,.05),edge(b,side*.25-.06,.05),
                                edge(b,side*.25+.06,.05),edge(a,side*.25+.06,.05),
                                (0,1,0),((0,0),(1,0),(1,1),(0,1)))
    profiles[1]['south_ground_ramp_osm']='611264728'
    profiles[1]['south_ground_ramp_length_m']=round(ramp.length,1)
    profiles[1]['zhai_pedestrian_portal']=campus_zhounan_portal.build(
        zhounan_portal,portal_treads,portal_walls,rails,beam,rendered_terrain_height)
    campus_zhounan_portal.save_tile_texture(ASSETS/'zhounan-zhai-tile.ppm')

    # 召南桥接致艺楼的地图线是穿楼通道；桥尾到通道保持同一地面高度。
    # 具体室内布局和高程未知，通道和后续缓坡是基于地图的连续性估计。
    zhaonan_passage=LineString(way_points(ways['776358640']))
    zhaonan_passage_y=main_sections[0][1][-1][3]
    deck.horizontal(zhaonan_passage.buffer(1.6,cap_style=2,join_style=1),zhaonan_passage_y)
    profiles[0]['building_passage_osm']='776358640'

    # 南区还存在一组贯穿建筑的高架人行连廊；初版遗漏了这些 OSM 桥梁路径。
    # 无竣工标高时按二层连廊估算；靠近跨路主桥处平滑接到主桥高度。
    local_widths = {'776358635':2.25,'1272661381':2.25,'1273971271':2.25,
                    '987586196':1.6,'1272661382':1.15,'1273971272':1.7,
                    '776358646':1.6,'984608623':1.6,'1297166011':1.6,'776358637':1.6,
                    '776358639':1.6,'776358641':1.6}
    local_paths = {i:LineString(way_points(ways[i])) for i in local_widths}
    local_nodes = {i:{n.get('ref') for n in ways[i].findall('nd')} for i in local_widths}
    junctions = {}
    for osm_id in local_widths:
        openings=[]
        for other_id,other_path in local_paths.items():
            if osm_id == other_id:
                continue
            for node_id in local_nodes[osm_id].intersection(local_nodes[other_id]):
                # 仅在地图路径真实共用的节点附近切开护墙，保留其余路段的护栏。
                openings.append(other_path.buffer(local_widths[other_id]+.2).intersection(Point(nodes[node_id]).buffer(6)))
        if osm_id == '984608623':
            # 饭堂二层和桥侧步道间的门洞与短平台是位置近似，护栏需在此留出口。
            openings.append(LineString(((430.0,233.9),(430.0,242.0))).buffer(2.8))
        junctions[osm_id] = prep(unary_union(openings))
    walkway_count = 0
    walkway_sections = []
    for key, t, path in roadways:
        if (t.get('bridge') in (None, 'no') or t.get('highway') != 'footway'
                or t.get('layer') != '1' or path.centroid.x < 220 or path.centroid.y < -100):
            continue
        for line in lines(path):
            if line.length < 5:
                continue
            if key == 'way/776358646':
                # 下层主桥板和粉砖平台已接管北端，裁去此前贴近上层的重复支线。
                line=substring(line,0,line.length-29)
            walkway_count += 1
            distances = list(np.arange(0.0,line.length,4.0)) + [line.length]
            items = []
            if key == 'way/776358646':
                # 照片中的南端步道先接入下层，而非突然爬到上层车道。
                # 从地图路线起点平滑接至桥下独立桥板；高度仍为近似。
                anchor=line.interpolate(50.0)
                center,section=main_sections[1]
                anchor_chain=center.project(anchor)
                anchor_top=float(np.interp(anchor_chain,[v[0] for v in section],[v[3] for v in section]))
                anchor_depth=float(np.interp(anchor_chain,lower_profile[0],lower_profile[1]))
                anchor_y=anchor_top-anchor_depth+.04
                first=line.interpolate(0)
                start_y=surface_height(first.x,first.y)+4.2
            for dist in distances:
                p = line.interpolate(dist)
                before, after = line.interpolate(max(0,dist-.5)), line.interpolate(min(line.length,dist+.5))
                tangent = np.array((after.x-before.x,after.y-before.y))
                tangent /= max(np.linalg.norm(tangent),.0001)
                normal = np.array((-tangent[1],tangent[0]))
                y = surface_height(p.x,p.y)+4.2
                if key == 'way/776358646':
                    center,section=main_sections[1]
                    projected=center.project(p)
                    top=float(np.interp(projected,[v[0] for v in section],[v[3] for v in section]))
                    depth=float(np.interp(projected,lower_profile[0],lower_profile[1]))
                    if dist < 50.0:
                        t=dist/50.0
                        # 同时抬高主桥后，适度减少 S 形曲线中部的最大坡度。
                        blend=.75*t+.25*t*t*(3-2*t)
                        y=start_y+(anchor_y-start_y)*blend
                    else:
                        y=max(surface_height(p.x,p.y)+.35,top-depth)+.04
                    under_main=dist>=50.0 and center.distance(p)<7.0
                    items.append((dist,p.x,p.y,y,normal[0],normal[1],under_main))
                    continue
                if key == 'way/776358641':
                    # 连接穿楼通道和南区二层外廊；地图未提供高程，按连续坡道估计。
                    end=line.interpolate(line.length)
                    end_y=surface_height(end.x,end.y)+4.2
                    y=zhaonan_passage_y+(end_y-zhaonan_passage_y)*(dist/line.length)
                    items.append((dist,p.x,p.y,y,normal[0],normal[1],False))
                    continue
                nearest, chosen = 16.0, None
                for center, section in main_sections:
                    separation = center.distance(p)
                    if separation < nearest:
                        nearest, chosen = separation, (center,section)
                if chosen is not None:
                    center, section = chosen
                    projected = center.project(p)
                    bridge_y = float(np.interp(projected,[v[0] for v in section],[v[3] for v in section]))
                    blend = max(0.,1.-nearest/16.)
                    y = y*(1.-blend) + bridge_y*blend
                items.append((dist,p.x,p.y,y,normal[0],normal[1],False))
            def side(item, offset, up=0.):
                _, x, z, y, nx, nz, _ = item
                return (x+nx*offset,y+up,z+nz*offset)
            # 致真楼与演播厅旁的这一段在实景中是露天平台；其余路段仍保留原先的近似雨棚。
            open_plaza = key in ('way/776358635','way/1272661381','way/1273971271')
            canteen_balcony = key == 'way/776358637'
            half_width = 2.25 if open_plaza else 1.6
            junction = junctions.get(key.removeprefix('way/'))
            for a,b in zip(items[:-1],items[1:]):
                under_main = a[6] and b[6]
                middle = Point((a[1]+b[1])*.5,(a[2]+b[2])*.5)
                indoors = any(way_polygon(ways[i]).covers(middle) for i in
                              ('629940892','654873808','629940891','629940889','629940890','776358621'))
                # 主桥下层板已覆盖人行线，避免再叠一层几乎共面的步道造成闪烁。
                if not under_main:
                    deck.quad(side(a,half_width),side(b,half_width),side(b,-half_width),side(a,-half_width),
                              (0,1,0),((0,0),(1,0),(1,1),(0,1)))
                for sign in (-1,1):
                    if not under_main:
                        ea,eb=side(a,sign*half_width),side(b,sign*half_width)
                        opening = (junction is not None and junction.intersects(LineString(((ea[0],ea[2]),(eb[0],eb[2])))))
                        if key == 'way/776358646' and (a[6] or b[6]):
                            opening = True
                        if canteen_balcony and sign == -1:
                            opening = True  # 朝饭堂外墙的一侧是开放外廊，不设阻拦栏杆。
                        if not opening:
                            deck.quad(side(a,sign*half_width,-.42),side(b,sign*half_width,-.42),
                                      side(b,sign*half_width,.78 if open_plaza else 0),
                                      side(a,sign*half_width,.78 if open_plaza else 0),(a[4]*sign,0,a[5]*sign),
                                      ((0,0),(1,0),(1,1),(0,1)))
                            rail_offset = sign*(half_width-.10)
                            beam(rails,side(a,rail_offset,1.05),side(b,rail_offset,1.05),.045)
                            if canteen_balcony and sign == 1:
                                # 南侧外廊照片有密集竖杆，栏杆间距仍为视觉估计。
                                for j in range(1,max(2,int((b[0]-a[0])/1.0))):
                                    t=j/max(2,int((b[0]-a[0])/1.0))
                                    p=np.asarray(side(a,rail_offset))*(1-t)+np.asarray(side(b,rail_offset))*t
                                    beam(rails,p,p+np.asarray((0,1.05,0)),.025)
                        if not open_plaza and not indoors and key != 'way/776358646' and not canteen_balcony:
                            roof.quad(side(a,0,3.2),side(b,0,3.2),side(b,sign*1.75,2.95),side(a,sign*1.75,2.95),
                                      (0,1,0),((0,0),(1,0),(1,1),(0,1)))
            for i, item in enumerate(items):
                dist,x,z,y,_,_,under_main = item
                indoors = any(way_polygon(ways[j]).covers(Point(x,z)) for j in
                              ('629940892','654873808','629940891','629940889','629940890','776358621'))
                if i % 2 == 0 and not under_main:
                    for sign in (-1,1):
                        offset = sign*(half_width-.10)
                        edge_point=side(item,offset)
                        opening=junction is not None and junction.covers(Point(edge_point[0],edge_point[2]))
                        if canteen_balcony and sign == -1:
                            opening=True
                        if not opening:
                            beam(rails,side(item,offset,.05),side(item,offset,1.05 if open_plaza or indoors or canteen_balcony or key == 'way/776358646' else 2.95),.06)
                        if not open_plaza and not indoors and key != 'way/776358646' and not canteen_balcony:
                            beam(rails,side(item,offset,2.95),side(item,0,3.2),.045)
                if i % 6 == 3 and not under_main and not canteen_balcony and not indoors:
                    candidate=support_location(items,i,2.0,local=True)
                    if candidate:
                        pillar,ground=candidate
                        _,px,pz,py,_,_,_=pillar
                        beam(support,(px,ground-.1,pz),(px,py-.4,pz),.32)
            walkway_sections.append((line,items))

    # 地图步道与饭堂外墙略有间距，补上二层入口短平台；只填空隙，避免重叠闪烁。
    canteen=way_polygon(ways['319101979'])
    access=LineString(((430.0,233.9),(430.0,242.0)))
    existing=local_paths['984608623'].buffer(1.6,cap_style=2)
    landing=clean(access.buffer(2.25,cap_style=2).difference(existing).difference(canteen))
    if not landing.is_empty:
        deck.horizontal(landing,surface_height(430.0,234.0)+4.21)

    # 南侧照片有二层外廊；填补建筑与地图步道之间的窄带。
    south_path=local_paths['776358637']
    balcony=Polygon(((399.4,277.72),(466.7,276.25),(466.7,282.05),(399.4,282.55)))
    balcony=clean(balcony.difference(canteen).difference(south_path.buffer(1.6,cap_style=2)))
    if not balcony.is_empty:
        deck.horizontal(balcony,surface_height(430.0,234.0)+4.21)
    # 地图记录的桥边楼梯：第一端接高架，另一端逐级落至地面。
    # 未取得施工图，踏步尺寸和扶手为近似；折线路径保留地图原样。
    stair_profiles = []
    for osm_id in ('237757618','776358647','987586194','1272661382','1273971272'):
        line = LineString(way_points(ways[osm_id]))
        start, end = Point(line.coords[0]), Point(line.coords[-1])
        candidates = [(path.distance(start),path,items) for path,items in main_sections+walkway_sections]
        distance, path, items = min(candidates,key=lambda candidate:candidate[0])
        if distance > 8:
            continue
        high = float(np.interp(path.project(start),[v[0] for v in items],[v[3] for v in items]))
        low = surface_height(end.x,end.y)+.12
        if high <= low+.3:
            continue
        count = max(2,int(math.ceil(line.length/.85)))
        # 演播厅旁第一段靠近楼墙，保持 2.3 米以免踏步伸进建筑；另一段按照片略加宽。
        local_stair = osm_id in ('1272661382','1273971272')
        half_width = 1.7 if osm_id == '1273971272' else 1.15
        handrail_offset = half_width-.10
        stair_profiles.append({'osm':osm_id,'width_estimate_m':round(2*half_width,2),
                               'high_estimate_m':round(high,2),'low_estimate_m':round(low,2),
                               'length_m':round(line.length,2),'steps_estimate':count})
        treads = []
        for i in range(count+1):
            dist = line.length*i/count
            p = line.interpolate(dist)
            a, b = line.interpolate(max(0,dist-.4)), line.interpolate(min(line.length,dist+.4))
            tangent = np.array((b.x-a.x,b.y-a.y))
            tangent /= max(np.linalg.norm(tangent),.0001)
            normal = np.array((-tangent[1],tangent[0]))
            treads.append((dist,p.x,p.y,high+(low-high)*i/count,normal[0],normal[1]))
        def stair_edge(item, offset, y):
            return (item[1]+item[4]*offset,y,item[2]+item[5]*offset)
        for a,b in zip(treads[:-1],treads[1:]):
            deck.quad(stair_edge(a,half_width,a[3]),stair_edge(b,half_width,a[3]),
                      stair_edge(b,-half_width,a[3]),stair_edge(a,-half_width,a[3]),
                      (0,1,0),((0,0),(1,0),(1,1),(0,1)))
            dx,dz=b[1]-a[1],b[2]-a[2]
            length=max(math.hypot(dx,dz),.0001)
            deck.quad(stair_edge(b,half_width,b[3]),stair_edge(b,-half_width,b[3]),
                      stair_edge(b,-half_width,a[3]),stair_edge(b,half_width,a[3]),
                      (dx/length,0,dz/length),((0,0),(1,0),(1,1),(0,1)))
            for sign in (-1,1):
                # 两处楼梯高端与步道转弯相接；入口侧栏杆留空，避免挡住主通道。
                if osm_id in ('1272661382','776358647') and sign == -1 and a[0] < 3.2:
                    continue
                if local_stair:
                    # 照片中的白色斜向护墙随踏步下降；厚度、高度按照片估计。
                    deck.quad(stair_edge(a,sign*half_width,a[3]-.40),stair_edge(b,sign*half_width,b[3]-.40),
                              stair_edge(b,sign*half_width,b[3]+.72),stair_edge(a,sign*half_width,a[3]+.72),
                              (a[4]*sign,0,a[5]*sign),((0,0),(1,0),(1,1),(0,1)))
                beam(rails,stair_edge(a,sign*handrail_offset,a[3]+1.05),
                     stair_edge(b,sign*handrail_offset,b[3]+1.05),.035)
        for item in treads[::5]:
            for sign in (-1,1):
                if osm_id in ('1272661382','776358647') and sign == -1 and item[0] < 3.2:
                    continue
                beam(rails,stair_edge(item,sign*handrail_offset,item[3]),
                     stair_edge(item,sign*handrail_offset,item[3]+1.05),.045)
        if local_stair:
            for item in treads[8:-4:10]:
                x,z,y=item[1],item[2],item[3]
                ground=rendered_terrain_height(x,z)
                if y-ground>1.5 and not support_exclusion.covers(Point(x,z)):
                    beam(support,(x,ground-.1,z),(x,y-.4,z),.25)

    # 两条北馆连廊沿 OSM 的精确线位跨过道路，接到二层平台；高程和宽度按照片近似。
    library_level = surface_height(-126.4,-102.9)+4.35
    library_link_profiles = []
    for osm_id,line in zip(('1274485928','231782436'),north_library_links):
        a,b=np.asarray(line.coords[0]),np.asarray(line.coords[-1])
        direction=(b-a)/np.linalg.norm(b-a)
        normal=np.asarray((-direction[1],direction[0]))
        half_width=2.25
        # 地图端点与简化建筑外墙有小量配准误差，桥板向两端延伸以避免视觉断缝。
        start,end=a-direction*1.3,b+direction*1.3
        footprint=LineString((start,end)).buffer(half_width,cap_style=2)
        extrude(deck,deck,footprint,library_level-.38,library_level)
        for sign in (-1,1):
            p,q=start+sign*normal*half_width,end+sign*normal*half_width
            roof.quad((p[0],library_level,p[1]),(q[0],library_level,q[1]),
                      (q[0],library_level+.95,q[1]),(p[0],library_level+.95,p[1]),
                      (sign*normal[0],0,sign*normal[1]),((0,0),(1,0),(1,1),(0,1)))
            beam(rails,(p[0],library_level+.98,p[1]),(q[0],library_level+.98,q[1]),.045)
        # 靠两端设支柱，避开中间穿行道路；柱位与尺寸为估计。
        for t in (.14,.86):
            p=a*(1-t)+b*t
            ground=surface_height(*p)
            if library_level-ground>1.3:
                for sign in (-1,1):
                    c=p+sign*normal*(half_width-.30)
                    if not support_exclusion.covers(Point(*c)):
                        ground=rendered_terrain_height(*c)
                        beam(support,(c[0],ground-.2,c[1]),(c[0],library_level-.38,c[1]),.22)
        library_link_profiles.append({'osm':osm_id,'length_m':round(line.length,2),
                                      'width_estimate_m':4.5,'deck_height_estimate_m':round(library_level,2)})

    # 北馆正门两段台阶已拆为独立入口资源，避免与新踏面重复或穿插。

    # 上下文山湖及北馆附近另有几处较短的 OSM 人行桥，按地形生成低矮桥面。
    lake_bridges = 0
    water_near = prep(unary_union([geom for _,geom,_ in lake_heights]).buffer(8))
    for key,t,path in roadways:
        if (t.get('bridge') in (None,'no') or t.get('highway') != 'footway'
                or path.centroid.x > 0 or not water_near.intersects(path)):
            continue
        for line in lines(path):
            if line.length < 4 or line.length > 75:
                continue
            lake_bridges += 1
            positions=[]
            for dist in list(np.arange(0.,line.length,3.0))+[line.length]:
                p=line.interpolate(dist)
                before,after=line.interpolate(max(0,dist-.5)),line.interpolate(min(line.length,dist+.5))
                direction=np.array((after.x-before.x,after.y-before.y))
                direction/=max(np.linalg.norm(direction),.0001)
                positions.append((dist,p.x,p.y,surface_height(p.x,p.y)+.5,-direction[1],direction[0]))
            def lake_edge(item,offset,up=0.):
                return (item[1]+item[4]*offset,item[3]+up,item[2]+item[5]*offset)
            for a,b in zip(positions[:-1],positions[1:]):
                deck.quad(lake_edge(a,1.2),lake_edge(b,1.2),lake_edge(b,-1.2),lake_edge(a,-1.2),
                          (0,1,0),((0,0),(1,0),(1,1),(0,1)))
                for sign in (-1,1):
                    beam(rails,lake_edge(a,sign*1.1,.9),lake_edge(b,sign*1.1,.9),.045)
            for item in positions[::2]:
                for sign in (-1,1):
                    beam(rails,lake_edge(item,sign*1.1),lake_edge(item,sign*1.1,.9),.045)
    profiles.append({'south_district_mapped_elevated_walkways': walkway_count,
                     'north_district_mapped_lake_bridges': lake_bridges,
                     'north_library_walkways': library_link_profiles,
                     'mapped_stairs': stair_profiles,
                     'note': '线位来自 OSM，桥面高程、上下层间距、雨棚和支撑细部为照片约束下的近似'})
    files = ((deck,'campus-bridge.mesh'), (rails,'campus-bridge-rails.mesh'),
             (roof,'campus-bridge-roof.mesh'), (support,'campus-bridge-support.mesh'),
             (roadway,'campus-bridge-road.mesh'), (markings,'campus-bridge-markings.mesh'),
             (lower,'campus-zhounan-lower.mesh'),(truss,'campus-zhounan-truss.mesh'),
             (glass,'campus-zhaonan-glass.mesh'),(zhounan_yellow,'campus-zhounan-yellow.mesh'),
             (zhounan_pots,'campus-zhounan-pots.mesh'),(zhounan_plants,'campus-zhounan-plants.mesh'),
             (zhounan_ceiling,'campus-zhounan-ceiling.mesh'),
             (portal_treads,'campus-zhounan-zhai-treads.mesh'),(portal_walls,'campus-zhounan-zhai-walls.mesh'))
    counts = {name: mesh.save(ASSETS / name) for mesh, name in files}
    (DATA / 'bridge-profile.json').write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf8')
    return counts


def make_stadium():
    # 外圈看台与照片中的浅色通道、分区座椅；位置和排数均为近似。
    red,blue,orange,yellow,seating,base,canopy,masts,lamps = (MeshWriter() for _ in range(9))
    surface,goals = (MeshWriter() for _ in range(2))
    geom,ground = stadium_footprint,flat_sports[0][1]
    cx,cz = stadium_center.x,stadium_center.y
    origin = (cx,cz)
    rotate = lambda g: affinity.rotate(g,stadium_angle,origin=origin)
    def point3(x,y,z):
        q=rotate(Point(x,z))
        return (q.x,y,q.y)
    # 整场使用独立高分辨率纹理和 mipmap，避免细白线俯视时变成闪烁的点。
    x0,z0,x1,z1=geom.bounds
    # 先以两倍尺寸绘制，再缩成 2048×3072，细线边缘可平滑过渡。
    texture_size=(4096,6144)
    arena=Image.new('RGB',texture_size,(103,144,122))
    paint=ImageDraw.Draw(arena)
    def texpoint(p):
        return (round((p[0]-x0)/(x1-x0)*(texture_size[0]-1)),
                round((p[1]-z0)/(z1-z0)*(texture_size[1]-1)))
    def paint_polygon(shape,color):
        for polygon in polygons(shape):
            paint.polygon([texpoint(p) for p in polygon.exterior.coords],fill=color)
    def paint_line(shape,color=(240,237,223),width=.16):
        pixels=max(1,round(width*texture_size[0]/(x1-x0)))
        for line in lines(shape):
            points=[texpoint(p) for p in line.coords]
            if len(points)>1: paint.line(points,fill=color,width=pixels,joint='curve')

    # 跑道由圆弧与直道组成，西侧还留出照片可见的直道延伸和两条跑跳助跑带。
    outer_r = stadium_inner_radius+stadium_lanes*stadium_lane_width
    extension = rotate(box(cx-outer_r,cz-stadium_half_straight-24,
                           cx-stadium_inner_radius,cz-stadium_half_straight))
    runways = [rotate(box(cx-outer_r-offset-1.35,cz-39,cx-outer_r-offset,cz+17)) for offset in (4.0,8.0)]
    paint_polygon(unary_union((stadium_track_outer,extension,*runways)),(177,77,62))
    field_surface=stadium_field.intersection(stadium_infield)
    paint_polygon(field_surface,(142,166,91))
    # 草地色差由固定种子生成，表达天然草坪的细小斑驳，不复制航拍像素。
    turf_rng=random.Random(230848519)
    inside_field=prep(field_surface.buffer(-.3))
    for _ in range(3400):
        x=turf_rng.uniform(field_surface.bounds[0],field_surface.bounds[2])
        z=turf_rng.uniform(field_surface.bounds[1],field_surface.bounds[3])
        if not inside_field.covers(Point(x,z)): continue
        px,pz=texpoint((x,z))
        radius=turf_rng.randint(1,5)
        color=turf_rng.choice(((136,160,86),(147,169,96),(139,163,87),(148,168,96)))
        paint.ellipse((px-radius,pz-radius,px+radius,pz+radius),fill=color)
    for lane in range(stadium_lanes+1):
        radius = stadium_inner_radius+lane*stadium_lane_width
        ring = stadium_axis.buffer(radius,quad_segs=48).boundary
        paint_line(ring)
        line = rotate(LineString(((cx-radius,cz-stadium_half_straight-24),
                                  (cx-radius,cz-stadium_half_straight))))
        paint_line(line)
    finish=rotate(LineString(((cx-outer_r,cz-stadium_half_straight-23.0),
                              (cx-stadium_inner_radius,cz-stadium_half_straight-23.0))))
    paint_line(finish,width=.20)

    # 球场边线、中线、禁区、球门区与球门按常见足球场比例示意，不当作现场测量。
    field_lines = [box(cx-34,cz-52.5,cx+34,cz+52.5).boundary,
                   LineString(((cx-34,cz),(cx+34,cz))),Point(cx,cz).buffer(9.15,quad_segs=32).boundary]
    for sign in (-1,1):
        edge=cz+sign*52.5
        for width,depth in ((40.32,16.5),(18.32,5.5)):
            field_lines.append(LineString(((cx-width/2,edge),(cx-width/2,edge-sign*depth),
                                           (cx+width/2,edge-sign*depth),(cx+width/2,edge))))
        field_lines.append(Point(cx,edge-sign*11).buffer(.13,quad_segs=8).boundary)
        # 前框和后框连成轻量球门，面朝球场中心。
        zfront=edge; zback=edge+sign*2.0
        for x in (cx-3.66,cx+3.66):
            beam(goals,point3(x,ground+.1,zfront),point3(x,ground+2.54,zfront),.055)
            beam(goals,point3(x,ground+2.54,zfront),point3(x,ground+2.3,zback),.045)
            beam(goals,point3(x,ground+.1,zback),point3(x,ground+2.3,zback),.045)
        beam(goals,point3(cx-3.66,ground+2.54,zfront),point3(cx+3.66,ground+2.54,zfront),.055)
        beam(goals,point3(cx-3.66,ground+2.3,zback),point3(cx+3.66,ground+2.3,zback),.045)
    # 旋转使标线与 OSM 球场的轻微方向一致。
    for line in field_lines:
        paint_line(rotate(line))
    paint_polygon(Point(cx,cz).buffer(.13,quad_segs=8),(240,237,223))
    arena.resize((2048,3072),Image.Resampling.LANCZOS).save(ASSETS/'stadium-surface.ppm')
    surface.horizontal_mapped(geom.buffer(-10.8),ground+.05,
                              lambda x,z:((x-x0)/(x1-x0),(z-z0)/(z1-z0)))

    for poly in polygons(geom):
        coords = list(orient(poly,sign=1.0).exterior.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            dx,dz = b[0]-a[0],b[1]-a[1]
            length = math.hypot(dx,dz)
            if length<.1: continue
            nx,nz = -dz/length,dx/length
            def pair(t):
                x,z=a[0]+dx*t,a[1]+dz*t
                return (x,ground+7,z),(x+nx*11,ground+.7,z+nz*11)
            outer_a,inner_a=pair(0);outer_b,inner_b=pair(1)
            normal=surface_normal(outer_a,outer_b,inner_b)
            base.quad(outer_a,outer_b,inner_b,inner_a,normal,((0,0),(1,0),(1,1),(0,1)))
            beam(seating,outer_a,outer_b,.045)
            beam(seating,inner_a,inner_b,.045)
            # 航拍可见座位被浅色楼梯分区；具体座位数量未核实。
            pieces=max(1,math.ceil(length/11.0))
            gap=min(.55/length,.18/pieces)
            for j in range(pieces):
                lo,hi=j/pieces+gap,(j+1)/pieces-gap
                if hi<=lo: continue
                oa,ia=pair(lo);ob,ib=pair(hi)
                midx,midz=a[0]+dx*(lo+hi)/2,a[1]+dz*(lo+hi)/2
                if midx>486 and -475<midz<-333:
                    target=yellow if -425<midz<-386 else orange
                elif midx<355 and -465<midz<-340:
                    target=red
                else:
                    target=blue if (midx<400 or midx>455) else orange
                lift=np.asarray(normal)*.035
                oa,ob,ib,ia=[tuple(np.asarray(p)+lift) for p in (oa,ob,ib,ia)]
                repeat=length*(hi-lo)/.53
                target.quad(oa,ob,ib,ia,normal,((0,0),(repeat,0),(repeat,14),(0,14)))

    # 主看台的八片白色膜面用曲面近似，立杆和索具单独成网格。
    for i in range(8):
        za,zb=-449+i*10.8,-449+(i+1)*10.8
        def membrane(u,v):
            x=342+12*u-1.1*u*math.sin(math.pi*v)
            z=za+(zb-za)*v
            tension=max(0.0,math.sin(math.pi*u)*math.sin(math.pi*v))
            y=ground+9.5-.4*u+4.2*tension**1.6
            return (x,y,z)
        for iu in range(18):
            for iv in range(20):
                a,b=membrane(iu/18,iv/20),membrane((iu+1)/18,iv/20)
                c,d=membrane((iu+1)/18,(iv+1)/20),membrane(iu/18,(iv+1)/20)
                canopy.quad(a,b,c,d,surface_normal(a,b,c),((0,0),(1,0),(1,1),(0,1)))
        beam(masts,(342,ground+7,za),(342,ground+15,za),.11)
        beam(masts,(342,ground+14,za),membrane(.5,.5),.035)

    # 现场照片中看台外缘可见高杆照明；杆位、数量与灯具阵列按照片概括。
    for x,z in ((351,-473),(351,-338),(490,-474),(490,-335)):
        beam(masts,(x,ground+6.5,z),(x,ground+31,z),.16)
        side=1 if x<cx else -1
        for row in range(4):
            for col in range(7):
                z0=z+(col-3)*.56; y0=ground+28.8+row*.52
                x0=x+side*.18
                corners=((x0,y0,z0-.18),(x0,y0,z0+.18),
                         (x0,y0+.34,z0+.18),(x0,y0+.34,z0-.18))
                lamps.quad(*corners,(side,0,0),((0,0),(1,0),(1,1),(0,1)))
    files = ((red,'campus-stands-red.mesh'),(blue,'campus-stands-blue.mesh'),
             (orange,'campus-stands-orange.mesh'),(yellow,'campus-stands-yellow.mesh'),
             (base,'campus-stands-base.mesh'),(seating,'campus-stands-lines.mesh'),
             (canopy,'campus-stadium-canopy.mesh'),(masts,'campus-stadium-masts.mesh'),
             (lamps,'campus-stadium-lamps.mesh'),(surface,'campus-stadium-surface.mesh'),
             (goals,'campus-stadium-goals.mesh'))
    result = {name: mesh.save(ASSETS / name) for mesh, name in files}
    # 校方航拍可辨西北主看台与弯道看台间隙；只局部切开并接合已有路面/场面。
    import campus_stadium_entry
    result.update(campus_stadium_entry.patch_assets(ASSETS, ASSETS, height_at=rendered_terrain_height))
    return result


def make_rooftop_sports():
    # 运动广场有架空运动层与天台球场；场地位置按 OSM，平台轮廓和高度按航拍近似。
    basketball_ids = ('231692661', '231692664', '231692665')
    tennis_ids = ('1081134204', '1081134205', '1081134206')
    basketball = [way_polygon(ways[osm_id]) for osm_id in basketball_ids]
    platform = unary_union(basketball).minimum_rotated_rectangle.buffer(4, join_style='mitre')
    center = platform.representative_point()
    ground = surface_height(center.x,center.y)
    roof_level = ground + 5.5
    shell, roof, court, tennis, rail, fence, boards, rims = (MeshWriter() for _ in range(8))
    stair = LineString(way_points(ways['1282922807']))
    # OSM 在运动广场西侧标有一段台阶；入口及终点高程按平台估计。
    stair_opening = stair.buffer(1.25)
    extrude(shell,roof,clean(platform.difference(stair_opening)),roof_level-.7,roof_level)
    roof.horizontal(platform,ground+.1)

    def boundary_frame(poly, base, top, columns=False):
        for part in polygons(poly):
            for a,b in zip(list(part.exterior.coords)[:-1],list(part.exterior.coords)[1:]):
                length = math.hypot(b[0]-a[0],b[1]-a[1])
                pieces = max(1, int(math.ceil(length/7.0)))
                for j in range(pieces+1):
                    u = j/pieces
                    x,z = a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u
                    if columns and stair.buffer(2.3).covers(Point(x,z)):
                        continue
                    if columns:
                        beam(shell,(x,ground-.2,z),(x,base-.5,z),.32)
                    beam(rail,(x,base,z),(x,top,z),.045)
                for j in range(pieces):
                    u,v=j/pieces,(j+1)/pieces
                    start=(a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u)
                    end=(a[0]+(b[0]-a[0])*v,a[1]+(b[1]-a[1])*v)
                    if columns and stair.buffer(2.3).intersects(LineString((start,end))):
                        continue
                    for y in (base+.2,top):
                        beam(rail,(start[0],y,start[1]),(end[0],y,end[1]),.035)

    boundary_frame(platform,roof_level,roof_level+2.4,True)

    # 台阶沿原有折线逐级上升；具体踏步尺寸仍属示意。
    first,last = Point(stair.coords[0]),Point(stair.coords[-1])
    low = surface_height(first.x,first.y)+.1
    count = max(2,int(math.ceil((roof_level-low)/.18)))
    steps = []
    for j in range(count+1):
        dist = stair.length*j/count
        p = stair.interpolate(dist)
        a,b = stair.interpolate(max(0,dist-.15)),stair.interpolate(min(stair.length,dist+.15))
        v = np.asarray((b.x-a.x,b.y-a.y)); v /= max(np.linalg.norm(v),.0001)
        steps.append((p.x,p.y,low+(roof_level-low)*j/count,-v[1],v[0]))
    def step_edge(item,side,y):
        return (item[0]+side*1.1*item[3],y,item[1]+side*1.1*item[4])
    for a,b in zip(steps[:-1],steps[1:]):
        roof.quad(step_edge(a,-1,a[2]),step_edge(b,-1,a[2]),step_edge(b,1,a[2]),step_edge(a,1,a[2]),
                  (0,1,0),((0,0),(1,0),(1,1),(0,1)))
        shell.quad(step_edge(b,-1,a[2]),step_edge(b,1,a[2]),step_edge(b,1,b[2]),step_edge(b,-1,b[2]),
                   (b[4],0,-b[3]),((0,0),(1,0),(1,1),(0,1)))
        for side in (-1,1):
            beam(rail,step_edge(a,side,a[2]+1),step_edge(b,side,b[2]+1),.035)
    for item in steps[::5]:
        for side in (-1,1):
            beam(rail,step_edge(item,side,item[2]),step_edge(item,side,item[2]+1),.04)

    def court_surface(mesh, poly, y):
        # 地图球场是矩形：沿短边和长边映射整张示意材质。
        p = list(poly.exterior.coords)[:-1]
        if len(p) != 4:
            raise ValueError('Expected a four-corner mapped sports court')
        mesh.quad((p[0][0],y,p[0][1]),(p[1][0],y,p[1][1]),
                  (p[2][0],y,p[2][1]),(p[3][0],y,p[3][1]),
                  (0,1,0),((0,0),(1,0),(1,1),(0,1)))

    def court_axes(poly):
        # 当前六个 OSM 场地的 0→1 边为长轴；生成前校验，避免静默旋转球架。
        p = np.asarray(list(poly.exterior.coords)[:-1],dtype=float)
        long_edge, short_edge = p[1]-p[0], p[3]-p[0]
        length,width = np.linalg.norm(long_edge),np.linalg.norm(short_edge)
        if len(p) != 4 or length <= width:
            raise ValueError('Sports court polygon does not have the expected long-axis ordering')
        return p,long_edge/length,short_edge/width,length,width

    def point3(xz,y):
        return (float(xz[0]),float(y),float(xz[1]))

    def net_panel(a,b,bottom,top):
        # 由裁剪纹理画出通透围网；位置和围网密度按照片近似。
        delta = b-a
        length = np.linalg.norm(delta)
        normal = (float(delta[1]/length),0,float(-delta[0]/length))
        fence.quad(point3(a,bottom),point3(b,bottom),point3(b,top),point3(a,top),normal,
                   ((0,0),(length/1.6,0),(length/1.6,(top-bottom)/1.6),(0,(top-bottom)/1.6)))

    def make_hoop(p,long_axis,short_axis,length,width,at_far_end):
        # 单独绘制移动式支架、篮板和篮圈，参考南体实景；未获场地施工尺寸。
        start = (p[1]+p[2])*.5 if at_far_end else (p[0]+p[3])*.5
        inward = -long_axis if at_far_end else long_axis
        board = start+inward*(length*.105)
        rear = start+inward*.6
        y = roof_level
        for side in (-1,1):
            a = rear+short_axis*(side*.65)
            b = board+short_axis*(side*.55)
            beam(rail,point3(a,y+.15),point3(b,y+3.72),.055)
            beam(rail,point3(a,y+.15),point3(a+inward*.85,y+.15),.055)
            beam(rail,point3(a+inward*.85,y+.15),point3(b,y+3.12),.045)
        beam(rail,point3(rear-short_axis*.65,y+.18),point3(rear+short_axis*.65,y+.18),.055)
        left,right = board-short_axis*.9,board+short_axis*.9
        normal = (float(inward[0]),0,float(inward[1]))
        boards.quad(point3(left,y+2.9),point3(right,y+2.9),point3(right,y+3.95),point3(left,y+3.95),
                    normal,((0,0),(1,0),(1,1),(0,1)))
        for h in (2.9,3.95):
            beam(rail,point3(left,y+h),point3(right,y+h),.025)
        for end in (left,right):
            beam(rail,point3(end,y+2.9),point3(end,y+3.95),.025)
        center = board+inward*.35
        ring_points = [center+.23*(math.cos(i*math.tau/16)*long_axis+math.sin(i*math.tau/16)*short_axis)
                       for i in range(17)]
        for a,b in zip(ring_points[:-1],ring_points[1:]):
            beam(rims,point3(a,y+3.05),point3(b,y+3.05),.022)
        # 细线只用于提示篮网轮廓，避免用大量三角形影响全校漫游。
        for i in (0,4,8,12):
            lower = center+.13*(math.cos(i*math.tau/16)*long_axis+math.sin(i*math.tau/16)*short_axis)
            beam(rail,point3(ring_points[i],y+3.05),point3(lower,y+2.62),.012)

    for poly in basketball:
        court_surface(court,poly,roof_level+.04)
        p,long_axis,short_axis,length,width = court_axes(poly)
        make_hoop(p,long_axis,short_axis,length,width,False)
        make_hoop(p,long_axis,short_axis,length,width,True)
        # 各场地独立围网及场地照明；具体门洞、灯位按照片概括。
        for a,b in zip(p,np.roll(p,-1,axis=0)):
            net_panel(a,b,roof_level+.1,roof_level+3.0)
            delta=b-a
            pieces=max(1,math.ceil(np.linalg.norm(delta)/6))
            for i in range(pieces+1):
                q=a+delta*i/pieces
                beam(rail,point3(q,roof_level),point3(q,roof_level+3),.035)
            beam(rail,point3(a,roof_level+3),point3(b,roof_level+3),.035)
        for q in (p[0],p[1]):
            beam(rail,point3(q,roof_level+3),point3(q,roof_level+7.5),.055)
            beam(rail,point3(q-short_axis*.25,roof_level+7.5),point3(q+short_axis*.25,roof_level+7.5),.10)

    # 至柔馆运动场层高按学生指南近似；校方另有四楼记录，结构尚待核实。
    gym = way_polygon(ways['230957349'])
    p = gym.representative_point()
    gym_roof = surface_height(p.x,p.y) + HEIGHTS['way/230957349'][0]
    for osm_id in tennis_ids:
        poly = way_polygon(ways[osm_id])
        court_surface(tennis,poly,gym_roof+.05)
        p,long_axis,short_axis,length,width = court_axes(poly)
        a,b = (p[0]+p[1])*.5,(p[2]+p[3])*.5
        net_panel(a,b,gym_roof+.08,gym_roof+1.02)
        beam(rail,point3(a,gym_roof+1.02),point3(b,gym_roof+1.02),.025)
        for end in (a,b):
            beam(rail,point3(end,gym_roof),point3(end,gym_roof+1.08),.045)
    boundary_frame(gym,gym_roof,gym_roof+2.8)

    files = ((shell,'campus-sports-shell.mesh'),(roof,'campus-sports-roof.mesh'),
             (court,'campus-basketball.mesh'),(tennis,'campus-tennis.mesh'),
             (rail,'campus-sports-rails.mesh'),(fence,'campus-sports-netting.mesh'),
             (boards,'campus-basketball-boards.mesh'),(rims,'campus-basketball-rims.mesh'))
    return {name: mesh.save(ASSETS/name) for mesh,name in files}, platform


def make_sundial():
    # 日晷位置使用 OSM node/11833634683，盘径、坡度及指针长度按校方照片估计。
    x,z = -243.2,-48.6
    ground = surface_height(x,z)
    radius, slope = 5.3, math.radians(23)
    rise = radius*math.sin(slope)+.35
    center = np.asarray((x,ground+rise,z))
    normal = np.asarray((-math.sin(slope),math.cos(slope),0.))
    face, stone, pointer = MeshWriter(),MeshWriter(),MeshWriter()
    pieces = 64
    for i in range(pieces):
        a,b = 2*math.pi*i/pieces,2*math.pi*(i+1)/pieces
        def rim(angle):
            return (x+radius*math.cos(angle)*math.cos(slope),
                    ground+rise+radius*math.cos(angle)*math.sin(slope),z+radius*math.sin(angle))
        p,q = rim(a),rim(b)
        uv = lambda angle:(.5+.49*math.cos(angle),.5+.49*math.sin(angle))
        face.triangle(center,p,q,normal,((.5,.5),uv(a),uv(b)))
        outward = (math.cos((a+b)*.5),0,math.sin((a+b)*.5))
        stone.quad((p[0],ground-.15,p[2]),(q[0],ground-.15,q[2]),q,p,outward,
                   ((i/12,0),((i+1)/12,0),((i+1)/12,(q[1]-ground)/2.2),(i/12,(p[1]-ground)/2.2)))
    beam(pointer,center+normal*.05,center+normal*4.6,.065)
    files = ((face,'campus-sundial-face.mesh'),(stone,'campus-sundial-stone.mesh'),
             (pointer,'campus-sundial-pointer.mesh'))
    return {name:mesh.save(ASSETS/name) for mesh,name in files}


def make_time_square():
    return campus_time_square.make_time_square(
        square=time_square,paving_area=time_square_paving,lawn=time_square_lawn,
        entry_area=huiyuan_entry_area,surface_height=surface_height,drape=drape_on_terrain,
        mesh_type=MeshWriter,beam=beam,clean=clean,lines=lines,assets=ASSETS)


def make_stone_plaza_and_pool():
    # 2026 年学校照片显示：橙白放射铺装、弧形带棚看台、圆顶舞台与较低的泳池。
    # 平面边界使用 OSM；各部件精确位置、高差、半径和阶梯仍是照片约束下的近似。
    shell, flat, plaza, water, canopy, rails = (MeshWriter() for _ in range(6))
    extrude(shell,flat,stone_plaza,pool_level-.4,stone_level-.04)
    plaza.horizontal_mapped(stone_plaza,stone_level+.015,lambda x,z:((x-79)/72,(z-79)/72))

    # 舞台临泳池一侧，低台上有白色圆顶与柱子，顶下补出径向支架。
    stage_center=np.asarray((130.0,101.5))
    stage=affinity.scale(Point(*stage_center).buffer(1,resolution=24),xfact=6.0,yfact=4.5)
    stage_roof=affinity.scale(Point(*stage_center).buffer(1,resolution=32),xfact=7.1,yfact=5.4)
    extrude(shell,flat,stage,stone_level,stone_level+.55)
    canopy.horizontal(stage_roof,stone_level+5.2)
    for dx,dz in ((-4.5,-2.5),(-4.5,2.5),(4.5,-2.5),(4.5,2.5)):
        beam(rails,(stage_center[0]+dx,stone_level+.45,stage_center[1]+dz),
                   (stage_center[0]+dx,stone_level+5.2,stage_center[1]+dz),.11)
    ring=[(stage_center[0]+7.1*math.cos(i*math.tau/32),stone_level+5.18,
           stage_center[1]+5.4*math.sin(i*math.tau/32)) for i in range(33)]
    for a,b in zip(ring[:-1],ring[1:]):
        beam(rails,a,b,.08)
    for i in range(0,32,4):
        beam(rails,(stage_center[0],stone_level+5.15,stage_center[1]),ring[i],.035)

    # 弧形看台沿广场西边抬升；遮棚从内向外覆盖座位。
    center=np.asarray((115.0,116.0))
    angles=np.linspace(1.95,4.34,32)
    def arc(radius,theta):
        return center+radius*np.asarray((math.cos(theta),math.sin(theta)))
    for i in range(4):
        inner,outer=18.0+i*.75,18.75+i*.75
        shape=Polygon([tuple(arc(outer,t)) for t in angles]+[tuple(arc(inner,t)) for t in reversed(angles)])
        shape=clean(shape.intersection(stone_plaza.buffer(-.1)))
        if not shape.is_empty:
            extrude(shell,flat,shape,stone_level-.05,stone_level+.31*(i+1))
    for a,b in zip(angles[:-1],angles[1:]):
        a0,a1=arc(19.1,a),arc(22.1,a)
        b0,b1=arc(19.1,b),arc(22.1,b)
        ya=stone_level+4.5+.45*math.cos((a-angles[0])/(angles[-1]-angles[0])*math.pi)
        yb=stone_level+4.5+.45*math.cos((b-angles[0])/(angles[-1]-angles[0])*math.pi)
        pa,pb,pc,pd=(a0[0],ya,a0[1]),(b0[0],yb,b0[1]),(b1[0],yb,b1[1]),(a1[0],ya,a1[1])
        canopy.quad(pa,pb,pc,pd,surface_normal(pa,pb,pc),((0,0),(1,0),(1,1),(0,1)))
    for theta in angles[::5]:
        p=arc(21.0,theta)
        y=stone_level+4.5+.45*math.cos((theta-angles[0])/(angles[-1]-angles[0])*math.pi)
        beam(rails,(p[0],stone_level+.95,p[1]),(p[0],y,p[1]),.11)

    # 校园现有道路接近广场南侧；按 2026 年照片补出入口梯段，确切级数待测量。
    approach=LineString(way_points(ways['582303818']))
    a,b=np.asarray(approach.coords[0]),np.asarray(approach.coords[-1])
    direction=b-a
    direction/=np.linalg.norm(direction)
    lateral=np.asarray((-direction[1],direction[0]))
    steps=14
    for i in range(steps):
        p0=a+(b-a)*i/steps
        p1=a+(b-a)*(i+1)/steps
        y0=pool_level+(stone_level-pool_level)*i/steps
        y1=pool_level+(stone_level-pool_level)*(i+1)/steps
        def edge(q,side,y): return (q[0]+side*lateral[0]*2,y,q[1]+side*lateral[1]*2)
        flat.quad(edge(p0,-1,y1),edge(p1,-1,y1),edge(p1,1,y1),edge(p0,1,y1),
                  (0,1,0),((0,0),(1,0),(1,1),(0,1)))
        shell.quad(edge(p0,-1,y0),edge(p0,1,y0),edge(p0,1,y1),edge(p0,-1,y1),
                   (-direction[0],0,-direction[1]),((0,0),(1,0),(1,1),(0,1)))
    for side in (-1,1):
        beam(rails,edge(a,side,pool_level+1),edge(b,side,stone_level+1),.04)
        for i in range(0,steps+1,3):
            p=a+(b-a)*i/steps
            y=pool_level+(stone_level-pool_level)*i/steps
            beam(rails,edge(p,side,y),edge(p,side,y+1),.035)

    # 泳池轮廓直接使用 OSM；水面与地砖独立，以照片中的深色泳道线表达尺度方向。
    deck=south_pool.buffer(3.2,join_style='mitre')
    flat.horizontal(deck,pool_level+.04)
    coords=np.asarray(list(south_pool.exterior.coords)[:-1],dtype=float)
    axis=coords[1]-coords[0];axis/=np.linalg.norm(axis)
    across=np.asarray((-axis[1],axis[0]))
    along_values=coords@axis;across_values=coords@across
    min_a,max_a=min(along_values),max(along_values)
    min_b,max_b=min(across_values),max(across_values)
    water.horizontal_mapped(south_pool,pool_level+.09,
        lambda x,z:((np.dot((x,z),axis)-min_a)/(max_a-min_a),
                    (np.dot((x,z),across)-min_b)/(max_b-min_b)))
    # 泳池靠石头坞一侧的阶梯坐席；长度、台阶和栏杆位置为实景近似。
    corners=np.asarray(list(south_pool.minimum_rotated_rectangle.exterior.coords)[:-1])
    edges=[]
    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
        length=np.linalg.norm(b-a)
        if length>35:
            edges.append((Point(*((a+b)*.5)).distance(stone_plaza),a,b))
    _,a,b=min(edges,key=lambda item:item[0])
    outward=(a+b)*.5-np.asarray((south_pool.centroid.x,south_pool.centroid.y))
    outward/=np.linalg.norm(outward)
    start=a+(b-a)*.07;end=b-(b-a)*.07
    for i in range(5):
        offset=3.1+i*.62
        p0,p1=start+outward*offset,end+outward*offset
        p2,p3=p1+outward*.62,p0+outward*.62
        shape=Polygon((p0,p1,p2,p3))
        extrude(shell,flat,shape,pool_level-.1,pool_level+.29*(i+1))
    back_start=start+outward*6.3
    back_end=end+outward*6.3
    for i in range(11):
        p=back_start+(back_end-back_start)*i/10
        beam(rails,(p[0],pool_level+1.45,p[1]),(p[0],pool_level+2.65,p[1]),.04)
    beam(rails,(back_start[0],pool_level+2.65,back_start[1]),
               (back_end[0],pool_level+2.65,back_end[1]),.04)

    files=((shell,'campus-stone-shell.mesh'),(flat,'campus-stone-flat.mesh'),
           (plaza,'campus-stone-plaza.mesh'),(water,'campus-south-pool.mesh'),
           (canopy,'campus-stone-canopies.mesh'),(rails,'campus-stone-rails.mesh'))
    return {name:mesh.save(ASSETS/name) for mesh,name in files}


def make_yuanping_and_beixia():
    """按公开航拍和正面照片重建体育馆、北区双泳池与看台。"""
    gym_shell,gym_base,gym_roof,gym_truss,gym_relief,gym_seal,gym_glass,gym_steps,gym_name,gym_garden = (MeshWriter() for _ in range(10))
    gym_foundation=MeshWriter()
    pool_shell,pool_roof,pool_frame,pool_seats,pool_deck,pool_lanes,pool_practice,pool_coping,pool_under = (MeshWriter() for _ in range(9))
    center = yuanping_building.representative_point()
    level = surface_height(center.x,center.y)
    # 体育馆楼体高度不变；独立基础沿近似地形接地，实际结构未经测量。
    campus_activity_foundation.add_foundation(gym_foundation,yuanping_building,top=level-2,
                                               terrain_height=rendered_terrain_height,polygons=polygons)

    def yp(along,depth,height):
        point = yuanping_mid + yuanping_tangent*along + yuanping_outward*depth
        return (point[0],level+height,point[1])

    def front_quad(mesh,left,right,depth,bottom,top,uv4=((0,0),(1,0),(1,1),(0,1))):
        mesh.quad(yp(left,depth,bottom),yp(right,depth,bottom),yp(right,depth,top),yp(left,depth,top),
                  (yuanping_outward[0],0,yuanping_outward[1]),uv4)

    # 底层保留中央开口并用退后的深色玻璃表示入口，防止外墙封死门廊。
    front_len = np.linalg.norm(yuanping_b-yuanping_a)
    ring = list(orient(yuanping_building,sign=1.0).exterior.coords)
    for a,b in zip(ring[:-1],ring[1:]):
        a,b=np.asarray(a),np.asarray(b)
        is_front = ((np.linalg.norm(a-yuanping_a)<.1 and np.linalg.norm(b-yuanping_b)<.1)
                    or (np.linalg.norm(a-yuanping_b)<.1 and np.linalg.norm(b-yuanping_a)<.1))
        if is_front:
            continue
        dx,dz=b-a
        length=math.hypot(dx,dz)
        gym_base.quad((a[0],level-2,a[1]),(b[0],level-2,b[1]),(b[0],level+4,b[1]),(a[0],level+4,a[1]),
                      (dz/length,0,-dx/length),((0,0),(length/4,0),(length/4,1.8),(0,1.8)))
    for left,right in ((-front_len/2,-6.4),(6.4,front_len/2)):
        front_quad(gym_base,left,right,0,-2,4,((0,0),((right-left)/4,0),((right-left)/4,1.8),(0,1.8)))
    front_quad(gym_glass,-6.4,6.4,-1.1,-.15,3.8)
    for x in np.arange(-6,6.1,2):
        beam(gym_truss,yp(x,-1.06,-.1),yp(x,-1.06,3.8),.045)
    for y in (1.8,3.7):
        beam(gym_truss,yp(-6.3,-1.06,y),yp(6.3,-1.06,y),.05)

    extrude(gym_shell,gym_shell,yuanping_building,level+4,level+9.6)
    # 灰褐色方形运动浮雕位于白色上墙，人物图案是依照片重画的近似。
    # PPM 行序与 OpenGL V 坐标方向相反，正面字样和人物需要翻转 V。
    upright=((0,1),(1,1),(1,0),(0,0))
    front_quad(gym_relief,-24,24,.06,5.2,8.9,upright)

    # 方形折面屋盖向外挑出，白色檐边下可见金属三角桁架。
    roof_outline=yuanping_building.minimum_rotated_rectangle.buffer(1.4,join_style=2)
    corners=list(orient(roof_outline,sign=1.0).exterior.coords)[:-1]
    c=roof_outline.centroid
    for a,b in zip(corners,corners[1:]+corners[:1]):
        pa,pb,peak=(a[0],level+12.6,a[1]),(b[0],level+12.6,b[1]),(c.x,level+15,c.y)
        gym_roof.triangle(pa,pb,peak,surface_normal(pa,pb,peak),((0,0),(1,0),(.5,.5)))
    for y in (9.8,12.35):
        beam(gym_truss,yp(-31,1.1,y),yp(31,1.1,y),.11)
    for i in range(13):
        x=-30+i*5
        beam(gym_truss,yp(x,1.1,9.8),yp(x,1.1,12.35),.07)
        if i<12:
            beam(gym_truss,yp(x,1.1,9.8),yp(x+2.5,1.1,12.35),.075)
            beam(gym_truss,yp(x+2.5,1.1,12.35),yp(x+5,1.1,9.8),.075)

    # 门廊、尖拱形开口及左右二层平台由实际照片近似；台阶高差为推定。
    for x in (-6,6):
        beam(gym_shell,yp(x,2,-1.1),yp(x,2,6.4),.70)
    beam(gym_shell,yp(-6,2,6.2),yp(6,2,6.2),.75)
    beam(gym_shell,yp(-5.3,2,4.5),yp(0,2,5.65),.30)
    beam(gym_shell,yp(0,2,5.65),yp(5.3,2,4.5),.30)
    front_quad(gym_seal,-1.7,1.7,2.77,5.1,8.5,upright)
    for x0,x1 in ((-28,-19),(19,28)):
        platform=Polygon((yp(x0,.05,0)[::2],yp(x1,.05,0)[::2],yp(x1,3,0)[::2],yp(x0,3,0)[::2]))
        extrude(gym_shell,gym_shell,platform,level+2.45,level+3.15)
        beam(gym_truss,yp(x0,3,3.9),yp(x1,3,3.9),.06)
        for x in np.arange(x0,x1+.1,1.5):
            beam(gym_truss,yp(x,3,3.15),yp(x,3,3.9),.035)
    front_quad(gym_name,21.5,26.5,3.04,2.55,3.1,upright)
    for left,right in ((-10,10),(-29,-23),(23,29)):
        for k in range(7):
            inner,outer=3.0+k*.54,3.54+k*.54
            top=-.1-k*.15
            a,b,c,d=yp(left,inner,top),yp(right,inner,top),yp(right,outer,top),yp(left,outer,top)
            gym_steps.quad(a,b,c,d,(0,1,0),((0,0),(1,0),(1,1),(0,1)))
            front_quad(gym_steps,left,right,outer,top-.15,top)
    for left,right in ((-22,-12),(12,22)):
        planter=Polygon((yp(left,3.2,0)[::2],yp(right,3.2,0)[::2],
                         yp(right,6.4,0)[::2],yp(left,6.4,0)[::2]))
        extrude(gym_shell,gym_shell,planter,level-1.25,level-.15)
        extrude(gym_garden,gym_garden,planter.buffer(-.3),level-.14,level+.25)

    # 两池轮廓来自 OSM；场边平台、看台台阶及屋盖形状按照片估计。
    pool_deck.horizontal(beixia_deck,beixia_level+.02)
    for i,geom in enumerate(beixia_pools):
        xmin,zmin,xmax,zmax=geom.bounds
        target=pool_practice if i==0 else pool_lanes
        target.horizontal_mapped(geom,beixia_level+.09,
                                 lambda x,z:((x-xmin)/(xmax-xmin),(z-zmin)/(zmax-zmin)))
        pool_coping.horizontal(geom.buffer(.45).difference(geom),beixia_level+.11)
    xmin,zmin,xmax,zmax=beixia_stand.bounds
    # 图片中后墙和两翼包住蓝色阶梯座席，中间看向露天泳池。
    for block in (box(xmin,zmin,xmin+5,zmax),box(xmax-5,zmin,xmax,zmax),
                  box(xmin+5,zmax-3,xmax-5,zmax)):
        extrude(pool_shell,pool_shell,block,beixia_level-1,beixia_level+8.3)
    seat_left,seat_right=xmin+5.4,xmax-5.4
    for row in range(14):
        za=zmin+.5+row*1.50
        y=beixia_level+.8+row*.36
        a,b,c,d=(seat_left,y,za),(seat_right,y,za),(seat_right,y,za+1.15),(seat_left,y,za+1.15)
        pool_seats.quad(a,b,c,d,(0,1,0),((0,0),(1,0),(1,1),(0,1)))
        pool_shell.quad((seat_left,y-.36,za),(seat_right,y-.36,za),b,a,(0,0,-1),
                        ((0,0),(1,0),(1,1),(0,1)))
        for x in np.arange(seat_left+.4,seat_right,.9):
            beam(pool_seats,(x,y+.03,za+.2),(x,y+.03,za+.75),.13)
    # 罩棚横向跨越看台，蓝色弧面及下方桁架均为重绘构造。
    xleft,xright=xmin-2,xmax+2
    zfront,zback=zmin-3,zmax+2
    def canopy_point(t,z):
        return (xleft+(xright-xleft)*t,beixia_level+10.0+5.5*math.sin(math.pi*t)**.8,z)
    for i in range(24):
        a,b=i/24,(i+1)/24
        pa,pb,pc,pd=canopy_point(a,zfront),canopy_point(b,zfront),canopy_point(b,zback),canopy_point(a,zback)
        pool_roof.quad(pa,pb,pc,pd,surface_normal(pa,pb,pc),((a,0),(b,0),(b,1),(a,1)))
        below=lambda p:(p[0],p[1]-.16,p[2])
        qa,qb,qc,qd=map(below,(pa,pb,pc,pd))
        normal=-surface_normal(pa,pb,pc)
        pool_under.quad(qd,qc,qb,qa,normal,((a,0),(b,0),(b,1),(a,1)))
    for z in np.linspace(zfront,zback,7):
        for i in range(24):
            beam(pool_frame,below(canopy_point(i/24,z)),below(canopy_point((i+1)/24,z)),.075)
    for t in np.linspace(.04,.96,12):
        beam(pool_frame,below(canopy_point(t,zfront)),below(canopy_point(t,zback)),.055)
    # 泳池两端的出发台及扶手为照片近似，数量和位置不是实测。
    xmin,zmin,xmax,zmax=beixia_pools[1].bounds
    for z in np.linspace(zmin+2,zmax-2,8):
        extrude(pool_coping,pool_coping,box(xmin-1.5,z-.34,xmin-.55,z+.34),beixia_level+.05,beixia_level+.65)
        beam(pool_seats,(xmin-1.5,beixia_level+.67,z),(xmin-.55,beixia_level+.67,z),.18)

    files=((gym_shell,'yuanping-shell'),(gym_base,'yuanping-base'),(gym_foundation,'yuanping-foundation'),(gym_roof,'yuanping-roof'),
           (gym_truss,'yuanping-truss'),(gym_relief,'yuanping-relief'),(gym_seal,'yuanping-seal'),
           (gym_glass,'yuanping-glass'),(gym_steps,'yuanping-steps'),
           (gym_name,'yuanping-name'),(gym_garden,'yuanping-garden'),
           (pool_shell,'beixia-shell'),(pool_roof,'beixia-roof'),(pool_frame,'beixia-frame'),
           (pool_seats,'beixia-seats'),(pool_deck,'beixia-deck'),(pool_lanes,'beixia-lanes'),
           (pool_practice,'beixia-practice'),(pool_coping,'beixia-coping'),(pool_under,'beixia-under'))
    result = {name:mesh.save(ASSETS/f'campus-{name}.mesh') for mesh,name in files}
    # 外梯、入口平台和独立行走面在本馆旧几何保存后局部更新，不改北夏馆。
    import campus_yuanping_entry
    entry_vertices = campus_yuanping_entry.patch_assets(
        ASSETS, ASSETS, height_at=rendered_terrain_height,
        frame={'mid':yuanping_mid,'tangent':yuanping_tangent,'outward':yuanping_outward,'level':level})
    result.update({'yuanping-'+name:count for name,count in entry_vertices.items()})
    return result


def make_golf():
    """按地图曲线和实景重建高尔夫练习场的开放击球棚与标牌。"""
    floor,roof,frame,mats,signs = (MeshWriter() for _ in range(5))
    anchor=golf_tee.representative_point()
    level=surface_height(anchor.x,anchor.y)
    floor.horizontal(golf_tee,level+.55)
    floor.horizontal(golf_tee,level+3.8)
    extrude(roof,roof,golf_tee.buffer(.7,join_style=2),level+7.05,level+7.30)

    ring=list(golf_tee.exterior.coords)
    back=LineString(ring[:13])
    front=LineString(ring[13:-1])
    for line in (back,front):
        for distance in np.arange(0,line.length+.1,4.6):
            q=line.interpolate(min(distance,line.length))
            beam(frame,(q.x,level+.4,q.y),(q.x,level+7.1,q.y),.11)
        for y in (3.85,4.85):
            for a,b in zip(line.coords[:-1],line.coords[1:]):
                beam(frame,(a[0],level+y,a[1]),(b[0],level+y,b[1]),.075)

    # 每个击球垫沿弧形棚的内侧切线放置，面朝草坪；间距来自照片的视觉近似。
    for distance in np.arange(1.5,front.length-1.5,3.7):
        q=front.interpolate(distance)
        p0,p1=front.interpolate(max(0,distance-.5)),front.interpolate(min(front.length,distance+.5))
        tangent=np.asarray((p1.x-p0.x,p1.y-p0.y),dtype=float)
        tangent/=np.linalg.norm(tangent)
        outward=np.asarray((-tangent[1],tangent[0]))
        if np.dot(outward,np.asarray((golf_course.centroid.x-q.x,golf_course.centroid.y-q.y)))<0:
            outward=-outward
        origin=np.asarray((q.x,q.y))
        mat=Polygon((origin-tangent*1.35+outward*.3,origin+tangent*1.35+outward*.3,
                     origin+tangent*1.35+outward*3.1,origin-tangent*1.35+outward*3.1))
        y=rendered_terrain_height(q.x+outward[0]*1.6,q.y+outward[1]*1.6)+.08
        mats.horizontal(mat,y)
        edge=origin+tangent*1.55+outward*.25
        end=edge+outward*3.0
        beam(frame,(edge[0],y+.08,edge[1]),(edge[0],y+.8,edge[1]),.045)
        beam(frame,(edge[0],y+.8,edge[1]),(end[0],y+.55,end[1]),.045)

    start=np.asarray((anchor.x,anchor.y))
    toward=np.asarray((golf_course.centroid.x-anchor.x,golf_course.centroid.y-anchor.y))
    toward/=np.linalg.norm(toward)
    across=np.asarray((-toward[1],toward[0]))
    for index,distance in enumerate((50,100,150,200)):
        p=start+toward*distance
        y=rendered_terrain_height(*p)
        left,right=p-across*.9,p+across*.9
        for post in (left,right):
            beam(frame,(post[0],y,post[1]),(post[0],y+1.4,post[1]),.035)
        a,b=(left[0],y+.55,left[1]),(right[0],y+.55,right[1])
        c,d=(right[0],y+1.4,right[1]),(left[0],y+1.4,left[1])
        u0,u1=index/4+.003,(index+1)/4-.003
        signs.quad(a,b,c,d,(-toward[0],0,-toward[1]),((u0,1),(u1,1),(u1,0),(u0,0)))

    files=((floor,'golf-floor'),(roof,'golf-roof'),(frame,'golf-frame'),(mats,'golf-mats'),(signs,'golf-signs'))
    return {name:mesh.save(ASSETS/f'campus-{name}.mesh') for mesh,name in files}


def make_trees(building_footprints,walking_surface,roadside_candidates,road_lamps,gate_reserved):
    # 地图未逐棵记录树木。用有固定随机种子的点位近似植被，避开建筑、水体与道路。
    obstacle_parts = [building_footprints.buffer(5)]
    obstacle_parts.extend(g.buffer(4) for _, _, g in waters)
    obstacle_parts.extend(g.buffer(7) for _, _, g in pitches)
    obstacle_parts.append(zhai_courtyard)
    obstacle_parts.append(wenshan_sites.buffer(5))
    # 2023 年景观改造航拍显示天人广场中央为开敞草坪，避免随机树木遮挡轴线。
    obstacle_parts.append(affinity.scale(Point(-108,36).buffer(1),xfact=47,yfact=62))
    obstacle_parts.append(box(-130,-46,-53,80))
    # 时光广场日晷周围是开放草坪；其范围使用 OSM 的广场轮廓。
    obstacle_parts.append(time_square_paving)
    obstacle_parts.append(Point(291,-400).buffer(12))
    for _, t, geom in roadways:
        if t.get('bridge') in (None, 'no'):
            obstacle_parts.append(geom.buffer(4.2 if t.get('highway') in ('path', 'footway', 'steps') else 7.0))
    forbidden = prep(unary_union(obstacle_parts))
    # 桥面和架空连廊也要避开树干及树冠，否则会出现树木从桥面穿出的穿模。
    # 在原有抽样后再筛除，保留不受影响区域既有的随机树木分布。
    bridge_clearance = prep(unary_union([
        geom.buffer((7.5 if key == 'way/574956548' else 3.5 if key == 'way/776358616' else 2.0) + 5.8)
        for key,t,geom in roadways if t.get('bridge') not in (None,'no')
    ]))
    # 广场与泳池均为人工铺装；不在其中或紧邻边缘随机种树。
    stone_clearance=prep(unary_union((stone_plaza,south_pool.buffer(3.2))).buffer(3))
    ancient_stone_clearance=prep(ancient_stone_site.buffer(8.5))
    # 照片中入口正前方为开敞草坪与铺装，不应被程序随机树冠封死。
    huiwen_clearance=prep(huiwen_approach.buffer(20))
    huixing_clearance=prep(unary_union(huixing_gardens).buffer(2))
    # 南馆前是成列棕榈，替换该带内的随机宽叶树；棕榈单独按照片近似布置。
    huizhi_palm_clearance=prep(LineString(((-110,117),(-25,100))).buffer(6.5))
    north_palm_clearance=prep(unary_union([Point(x,z).buffer(5.5) for x,z in
                                           ((-142,-62),(-146,-56),(-97,-71),(-92,-66))]))
    huizi_court_clearance=prep(unary_union(huizi_courtyards).buffer(2))
    huizi_entry_clearance=prep(huizi_approach.buffer(5))
    huizi_palm_clearance=prep(unary_union([Point(x,z).buffer(5) for x,z in
                                           ((-25,45),(-27,47),(-23,48))]))
    huiyuan_entry_clearance=prep(huiyuan_rect(-3.8,10.8,0,9.2).buffer(5))
    # 2022 年立德门航拍中，通向时光广场的长草坪没有高树。
    # 用 OSM 草地轮廓限制随机树冠；在抽样后筛除，不改变其他区域的随机点位。
    gate_lawn_clearance=prep(way_polygon(ways['1273530488']).buffer(5.0))
    # 中庭航拍显示为开敞铺装和低矮草坪，剔除落在其中的程序随机树。
    south_courtyard_clearance=prep(south_courtyard.buffer(1))
    huicui_plaza_clearance=prep(huicui_forecourt.buffer(2))
    sports_site_clearance=prep(unary_union((beixia_deck.buffer(4),yuanping_paving.buffer(5))))
    golf_clearance=prep(golf_clear)
    legong_palm_clearance=prep(unary_union([Point(x,z).buffer(5.5) for x,z in legong_palms]))
    # 入口岗亭及闸机留出近景空间，避免随机树干穿过设施。
    legong_entry_clearance=prep(Point(300.8,-678).buffer(8))
    all_gate_clearance=prep(gate_reserved.buffer(2))
    # 保留森林氛围，但不让随机树干穿过新园路、入口框架和宿舍通道。
    qiaoyuan_path_clearance=prep(qiaoyuan_walkway.buffer(2.8))
    sidewalk_clearance=prep(walking_surface.buffer(.9))
    lamp_clearance=prep(unary_union([Point(x,z).buffer(4.0) for x,z in road_lamps]))
    # 南区大草坪的开阔范围以 OSM 多边形及校方照片核对，树木保留在外围。
    south_gate_clearance=prep(unary_union((south_gate_lawn.buffer(-1),liyan_entry.buffer(3),
                                           zhiyuan_entry.buffer(3),zhizhi_courtyard.buffer(2),
                                           zhili_a_entry.buffer(5),zhigong_parking.buffer(2),
                                           zhigong_forecourt.buffer(5),zhigong_stair_site.buffer(2))))
    inside = prep(campus.buffer(-5))
    woods = prep(unary_union([g for _, t, g in parks if t.get('natural') == 'wood' or t.get('landuse') == 'forest']))
    lakeside = prep(unary_union([g for _, _, g in waters]).buffer(55))
    rng = random.Random(2024270223)
    trees = []
    # 荔园和文山湖附近树冠相互接近；点位仍为近似，而非逐棵实测。
    step = 7
    for x in range(math.floor(X0), math.ceil(X1), step):
        for z in range(math.floor(Z0), math.ceil(Z1), step):
            xx, zz = x + rng.uniform(-5, 5), z + rng.uniform(-5, 5)
            p = Point(xx, zz)
            # 荔枝林与文山湖边植被较密；树种和逐棵坐标仍属于近似。
            density = .88 if woods.covers(p) or lakeside.covers(p) or (xx < -350 and zz < 150) else .52
            if inside.covers(p) and not forbidden.intersects(p) and rng.random() < density:
                # 树根应落在渲染用的三角地面上，避免粗网格坡面上的双线性插值误差。
                tree = (xx, rendered_terrain_height(xx, zz), zz, rng.uniform(1.00, 1.62))
                if (not bridge_clearance.intersects(p) and not stone_clearance.intersects(p)
                        and not ancient_stone_clearance.intersects(p)
                        and not huiwen_clearance.intersects(p) and not huixing_clearance.intersects(p)
                        and not huizhi_palm_clearance.intersects(p)
                        and not north_palm_clearance.intersects(p)
                        and not huizi_court_clearance.intersects(p)
                        and not huizi_entry_clearance.intersects(p)
                        and not huizi_palm_clearance.intersects(p)
                        and not huiyuan_entry_clearance.intersects(p)
                        and not gate_lawn_clearance.intersects(p)
                        and not south_courtyard_clearance.intersects(p)
                        and not huicui_plaza_clearance.intersects(p)
                        and not sports_site_clearance.intersects(p)
                        and not legong_palm_clearance.intersects(p)
                        and not legong_entry_clearance.intersects(p)
                        and not qiaoyuan_path_clearance.intersects(p)
                        and not sidewalk_clearance.intersects(p)
                        and not lamp_clearance.intersects(p)
                        and not south_gate_clearance.intersects(p)
                        and not golf_clearance.intersects(p)):
                    trees.append(tree)
    # 在有余地的道路外侧补充行道树；沿用同一实例网格，并与既有树和步道保持间距。
    bins=collections.defaultdict(list)
    spacing=8.0
    for x,_,z,_ in trees:
        bins[(math.floor(x/spacing),math.floor(z/spacing))].append((x,z))
    roadside_count=0
    roadside_rng=random.Random(20261103)
    special=(bridge_clearance,stone_clearance,ancient_stone_clearance,huiwen_clearance,huixing_clearance,huizhi_palm_clearance,
             north_palm_clearance,huizi_court_clearance,huizi_entry_clearance,huizi_palm_clearance,
             huiyuan_entry_clearance,gate_lawn_clearance,south_courtyard_clearance,huicui_plaza_clearance,
             sports_site_clearance,legong_palm_clearance,legong_entry_clearance,qiaoyuan_path_clearance,
             sidewalk_clearance,lamp_clearance,south_gate_clearance,golf_clearance)
    for x,z in roadside_candidates:
        point=Point(x,z)
        if not inside.covers(point) or forbidden.intersects(point) or any(g.intersects(point) for g in special):
            continue
        gx,gz=math.floor(x/spacing),math.floor(z/spacing)
        if any(math.hypot(x-a,z-b)<spacing for i in range(gx-1,gx+2)
               for j in range(gz-1,gz+2) for a,b in bins[(i,j)]):
            continue
        trees.append((x,rendered_terrain_height(x,z),z,roadside_rng.uniform(1.00,1.42)))
        bins[(gx,gz)].append((x,z))
        roadside_count+=1
    original_count=len(trees)-roadside_count
    retained=[(index,tree) for index,tree in enumerate(trees)
              if not all_gate_clearance.intersects(Point(tree[0],tree[2]))]
    roadside_count=sum(index>=original_count for index,_ in retained)
    trees=[tree for _,tree in retained]
    with (ASSETS / 'campus-trees.bin').open('wb') as f:
        f.write(struct.pack('<I', len(trees)))
        for t in trees:
            f.write(struct.pack('<4f', *t))
    return len(trees),roadside_count


def make_annotated_planting(height_at=rendered_terrain_height):
    """从带置信度的标注生成独立灌木与可选树木实例，不修改随机树网格。"""
    config=json.loads((DATA/'planting-annotations.json').read_text(encoding='utf8'))
    bridges=unary_union([g.buffer((7.5 if key=='way/574956548' else
                                    3.5 if key=='way/776358616' else 2.0)+5.8)
                         for key,t,g in roadways if t.get('bridge') not in (None,'no')])
    return campus_planting.generate(config=config,assets=ASSETS,campus=campus,
        vehicle_surface=vehicle_surface,buildings=buildings,waters=waters,pitches=pitches,
        booth=Polygon(LIDE_WALK['gate_booth_footprint_m']),bridge_clearance=bridges,
        height_at=height_at,mesh_type=MeshWriter)


def main():
    facade_texture('facade-cream', (218, 218, 207), (87, 114, 119), (187, 191, 185))
    # 老斋/西苑实景以白色与浅暖色为主；文件名沿用旧版以保持渲染器兼容。
    facade_texture('facade-red', (222, 220, 210), (91, 112, 111), (194, 177, 158))
    facade_texture('facade-city', (164, 183, 190), (76, 104, 118), (152, 174, 181))
    landmark_facades()
    sports_textures()
    stone_plaza_textures()
    yuanping_textures()
    golf_texture()
    sundial_textures()
    qiaoyuan_textures()
    gate_textures()
    zhiyuan_sign_textures()
    canopy_texture()
    huizhi_textures()
    huizi_roof_textures()
    zhizhen_textures()
    zhigong_textures()
    canteen_textures()
    dujuan_boardwalk_texture()
    campus_asphalt_texture()
    qiaoyuan_garden_textures()
    xiyuan_shop_textures()
    campus_road_network.sidewalk_texture(ASSETS)
    terrain_vertices = make_terrain()
    make_ground_texture()
    gates,gate_reserved=campus_gates.make_gates(mesh_type=MeshWriter,height_at=rendered_terrain_height,
        beam=beam,extrude=extrude,assets=ASSETS,data=DATA,vehicle_surface=vehicle_surface)
    road_details = make_road_details()
    road_environment,walking_surface,roadside_candidates,road_lamps,road_reserved=make_road_environment()
    lide_walk=make_lide_walk()
    road_stairs=campus_stairways.make_stairways(
        campus=campus,roadways=roadways,vehicle_surface=vehicle_surface,buildings=buildings,waters=waters,
        pitches=pitches,reserved=road_reserved,height_at=rendered_terrain_height,mesh_type=MeshWriter,
        assets=ASSETS,data=DATA)
    shuttle=campus_shuttle.make_shuttle(mesh_type=MeshWriter,way_points=way_points,ways=ways,assets=ASSETS)
    ancient_stone=campus_ancient_stone.make_ancient_stone(
        mesh_type=MeshWriter,height_at=rendered_terrain_height,assets=ASSETS,data=DATA)
    swans=campus_wenshan_swans.make_swans(mesh_type=MeshWriter,upper_lake=upper_lake,
                                          water_level=upper_lake_level+.2,assets=ASSETS,data=DATA)
    qiaoyuan_garden = make_qiaoyuan_garden()
    xiyuan_shops = make_xiyuan_shops()
    north_city_vertices = make_north_city_ground()
    huixing_landscape = make_huixing_landscape()
    mesh_counts, footprint = make_buildings()
    kaifeng_geom=next(g for key,_,g in buildings if key=='way/'+kaifeng_config['osm_way'])
    kp=kaifeng_geom.representative_point()
    kaifeng=campus_kaifeng.make_kaifeng(
        config=kaifeng_config,geom=kaifeng_geom,points=kaifeng_points,altitude=surface_height(kp.x,kp.y),
        mesh_type=MeshWriter,extrude=extrude,extrude_openings=extrude_with_walkway_openings,
        beam=beam,normal=surface_normal,height_at=rendered_terrain_height,assets=ASSETS)
    yuanping_beixia_vertices = make_yuanping_and_beixia()
    golf_vertices = make_golf()
    water_vertices = make_water()
    wenshan_terraces = make_wenshan_terraces()
    bridge_vertices = make_bridges(walking_surface)
    # 北馆入口及U形侧梯使用同一地形基准；独立生成，保留其它桥和道路。
    import campus_north_library_entry
    import campus_north_library_upper
    north_library_entry = campus_north_library_entry.make_assets(ASSETS, ASSETS, height_at=rendered_terrain_height)
    north_library_upper = campus_north_library_upper.make_assets(ASSETS, height_at=rendered_terrain_height)
    with (ASSETS/'campus-bridge.mesh').open('rb') as stream:
        bridge_vertices['campus-bridge.mesh'] = struct.unpack('<I',stream.read(8)[4:])[0]
    south_spine=make_south_courtyard(assets=ASSETS,terrain=ASSETS/'campus-terrain.mesh',
                                    height_at=rendered_terrain_height)
    # 独立宿舍读取本轮桥面高度，切通实际接入点；干净生成时也不依赖旧桥资源。
    dorm_models=make_dorm_models()
    zhiyi_skywalk = make_zhiyi_skywalks()
    zhili_high_skywalk = make_zhili_high_walkways()
    zhigong_parking_vertices = make_zhigong_parking()
    stadium_vertices = make_stadium()
    sports_vertices, sports_platform = make_rooftop_sports()
    sundial_vertices = make_sundial()
    time_square_vertices = make_time_square()
    yanhui_pine_vertices = make_yanhui_pines()
    stone_vertices = make_stone_plaza_and_pool()
    tree_count,roadside_tree_count = make_trees(
        unary_union((footprint,sports_platform,kaifeng_forecourt,Point(-243.2,-48.6).buffer(8))),
        walking_surface,roadside_candidates,road_lamps,gate_reserved)
    # 程序推断的道路环境已生成，样板只裁掉冲突的局部铺装和装饰。
    corridor=campus_lide_corridor.build_layers(
        config=DATA/'lide-corridor.json', assets=ASSETS, output=ASSETS,
        road=vehicle_surface, plaza=time_square_paving, buildings=[g for _,_,g in buildings],
        booth=Polygon(LIDE_WALK['gate_booth_footprint_m']))
    roadside=DATA/'roadside-environment.json'
    lamp_count,verge_area=campus_lide_corridor.refresh_roadside_data(roadside,roadside,ASSETS)
    for key, file in (('inferred_sidewalk','sidewalk'), ('mapped_paths','paths'), ('verges','verges'),
                      ('verge_grass','verge-grass'), ('shrubs','shrubs'), ('road_lamps','lamps'),
                      ('road_lamp_heads','lamp-heads')):
        with (ASSETS/f'campus-network-{file}.mesh').open('rb') as mesh_file:
            road_environment[key]=struct.unpack('<I',mesh_file.read(8)[4:])[0]
    road_environment.update(lamp_count_approximate=lamp_count,verge_area_estimate_m2=verge_area,
                            verge_grass_clumps=road_environment['verge_grass']//9,
                            shrub_count_approximate=road_environment['shrubs']//18)
    planting=make_annotated_planting()
    places=campus_places.make_places(buildings=buildings,waters=waters,nodes=nodes,
        height_at=rendered_terrain_height,height_for=height_for,gates=gates,assets=ASSETS,data=DATA)
    info = {'source': 'OpenStreetMap contributors; Mapzen Terrain Tiles / USGS',
            'campus_buildings': len(buildings), 'surrounding_buildings': len(surrounding),
            'waters': len(waters), 'paths_and_roads': len(roadways),
            'trees_approximate': tree_count, 'terrain_vertices': terrain_vertices,
            'annotated_planting': {'shrubs':len(planting['shrub_points']),
                                   'tree_instances':planting['tree_instances'],
                                   'rejected':len(planting['rejected'])},
            'lide_corridor': {key:value for key,value in corridor.items() if not isinstance(value,list)},
            'south_spine': south_spine,
            'road_details': road_details,
            'road_surface_overrides':road_override_report,
            'road_environment': road_environment, 'roadside_trees_approximate': roadside_tree_count,
            'lide_walk': lide_walk,
            'road_stairs': road_stairs,
            'gates':len(gates),'place_labels':places,
            'shuttle': shuttle,
            'ancient_stone': ancient_stone,
            'wenshan_swans': swans,
            'wenshan_shore_terrain': wenshan_shore_terrain,
            'kaifeng_canteen': kaifeng,
            'qiaoyuan_garden': qiaoyuan_garden,
            'north_city_ground_vertices': north_city_vertices,
            'water_vertices': water_vertices, 'wenshan_terrace_vertices': wenshan_terraces,
            'bridge_vertices': bridge_vertices, 'zhiyi_skywalk_vertices': zhiyi_skywalk,
            'north_library_entry': north_library_entry, 'north_library_upper': north_library_upper,
            'zhili_high_skywalk_vertices': zhili_high_skywalk,
            'zhigong_parking': zhigong_parking_vertices,
            'stadium_vertices': stadium_vertices, 'sports_vertices': sports_vertices,
            'sundial_vertices': sundial_vertices, 'time_square_vertices': time_square_vertices,
            'yanhui_pine_vertices': yanhui_pine_vertices,
            'stone_plaza_vertices': stone_vertices,
            'huixing_landscape_vertices': huixing_landscape,
            'yuanping_beixia_vertices': yuanping_beixia_vertices,
            'golf_vertices': golf_vertices,
            'building_mesh_vertices': mesh_counts}
    info['dorm_models']=dorm_models
    (ASSETS / 'campus-data.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
