"""对可重复生成的校园数据做少量几何与资源一致性检查。无需图形设备。"""
import collections
import csv
import json
import math
import struct
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
ASSETS, DATA = ROOT/'assets', ROOT/'data'


@lru_cache(maxsize=1)
def terrain_data():
    with (ASSETS/'campus-heights.bin').open('rb') as f:
        nx,nz,x0,x1,z0,z1 = struct.unpack('<II4f',f.read(24))
        heights = struct.unpack('<' + str(nx*nz) + 'f',f.read())
    return nx,nz,x0,x1,z0,z1,heights


@lru_cache(maxsize=1)
def terrain_patches():
    with (ASSETS/'campus-terrain-patches.bin').open('rb') as f:
        if f.read(4)!=b'SZP1':
            raise ValueError('Invalid terrain patch data')
        count,=struct.unpack('<I',f.read(4))
        patches=[]
        for _ in range(count):
            nx,nz,x0,x1,z0,z1=struct.unpack('<II4f',f.read(24))
            values=struct.unpack('<'+str(nx*nz)+'f',f.read(nx*nz*4))
            patches.append((nx,nz,x0,x1,z0,z1,values))
        if f.read(1):
            raise ValueError('Unexpected data after terrain patches')
    return patches


def terrain_patch_at(x,z):
    for nx,nz,x0,x1,z0,z1,values in terrain_patches():
        if not (x0<=x<=x1 and z0<=z<=z1):
            continue
        u=max(0,min(nx-1,(x-x0)/(x1-x0)*(nx-1)))
        v=max(0,min(nz-1,(z-z0)/(z1-z0)*(nz-1)))
        i,j=min(int(u),nx-2),min(int(v),nz-2)
        a,b=u-i,v-j
        p,q,r,s=values[j*nx+i],values[j*nx+i+1],values[(j+1)*nx+i],values[(j+1)*nx+i+1]
        return (1-b)*p+(b-a)*r+a*s if b>=a else (1-a)*p+b*s+(a-b)*q
    return None


def terrain_at(x, z):
    local=terrain_patch_at(x,z)
    if local is not None:
        return local
    nx,nz,x0,x1,z0,z1,heights=terrain_data()
    u = max(0,min(nx-1,(x-x0)/(x1-x0)*(nx-1)))
    v = max(0,min(nz-1,(z-z0)/(z1-z0)*(nz-1)))
    i,j = min(int(u),nx-2),min(int(v),nz-2)
    a,b = u-i,v-j
    return ((1-b)*((1-a)*heights[j*nx+i]+a*heights[j*nx+i+1])
            + b*((1-a)*heights[(j+1)*nx+i]+a*heights[(j+1)*nx+i+1]))


def terrain_triangle_at(x,z):
    # 渲染网格使用两片三角面而非双线性曲面，检查踏步时应使用同一插值。
    local=terrain_patch_at(x,z)
    if local is not None:
        return local
    return coarse_terrain_triangle_at(x,z)


def coarse_terrain_triangle_at(x,z):
    nx,nz,x0,x1,z0,z1,heights=terrain_data()
    u=max(0,min(nx-1,(x-x0)/(x1-x0)*(nx-1)))
    v=max(0,min(nz-1,(z-z0)/(z1-z0)*(nz-1)))
    i,j=min(int(u),nx-2),min(int(v),nz-2)
    a,b=u-i,v-j
    p,q,r,s=heights[j*nx+i],heights[j*nx+i+1],heights[(j+1)*nx+i],heights[(j+1)*nx+i+1]
    return (1-b)*p+(b-a)*r+a*s if b>=a else (1-a)*p+b*s+(a-b)*q


def mesh_vertices(filename):
    with (ASSETS/filename).open('rb') as f:
        if f.read(4) != b'SZM1':
            raise ValueError(f'Invalid mesh: {filename}')
        count, = struct.unpack('<I',f.read(4))
        data = f.read()
    if len(data) != count*32 or count < 3:
        raise ValueError(f'Truncated mesh: {filename}')
    return list(struct.iter_unpack('<8f',data))


deck = mesh_vertices('campus-bridge.mesh')
for name,x,z in (('召南桥',186.5,339.0),('周南桥',412.0,139.0)):
    nearby = [v[1] for v in deck if math.hypot(v[0]-x,v[2]-z) < 16]
    if not nearby or max(nearby)-terrain_at(x,z) < 4.5:
        raise ValueError(f'{name}: no bridge deck with expected road clearance')
library_steps = [v for v in deck if -120 < v[0] < -104 and -50 < v[2] < -30]
if (len(library_steps) < 30 or max(v[1] for v in library_steps)-min(v[1] for v in library_steps) < .3
        or any(v[1]-terrain_at(v[0],v[2]) < -.15 for v in library_steps)):
    raise ValueError('North-library approach steps are missing, flat or buried in the terrain')

profiles = json.loads((DATA/'bridge-profile.json').read_text(encoding='utf-8'))
for item in profiles[:2]:
    if item['min_height_above_terrain_m'] < 0 or item['max_grade'] > .12:
        raise ValueError(f'Bridge profile intersects terrain or has a discontinuous ramp: {item}')
if profiles[2]['south_district_mapped_elevated_walkways'] < 10:
    raise ValueError('Expected mapped south-district walkways are missing')

# 校核整块踏面而非只校核楼梯中心，防止临路一侧被边坡穿入。
portal_treads=mesh_vertices('campus-zhounan-zhai-treads.mesh')
for i in range(0,len(portal_treads),3):
    a,b,c=portal_treads[i:i+3]
    for j in range(11):
        for k in range(11-j):
            u,w=j/10,k/10
            x,y,z=(u*a[axis]+w*b[axis]+(1-u-w)*c[axis] for axis in range(3))
            if terrain_triangle_at(x,z)>y+.04:
                raise ValueError(f'ZhouNan portal tread is blocked by terrain at {(x,z)}')

def vertical_surface_hits(vertices,x,z):
    # 在水平位置对上表面作竖向采样，可检测不同资源生成的桥板是否重合或错层。
    found=[]
    for i in range(0,len(vertices),3):
        a,b,c=vertices[i:i+3]
        if min(a[4],b[4],c[4]) < .7:
            continue
        denominator=(b[2]-c[2])*(a[0]-c[0])+(c[0]-b[0])*(a[2]-c[2])
        if abs(denominator) < 1e-7:
            continue
        u=((b[2]-c[2])*(x-c[0])+(c[0]-b[0])*(z-c[2]))/denominator
        w=((c[2]-a[2])*(x-c[0])+(a[0]-c[0])*(z-c[2]))/denominator
        t=1-u-w
        if min(u,w,t) > -1e-4:
            found.append(u*a[1]+w*b[1]+t*c[1])
    return found

lower_deck=mesh_vertices('campus-zhounan-lower.mesh')
truss=mesh_vertices('campus-zhounan-truss.mesh')
bridge_glass=mesh_vertices('campus-zhaonan-glass.mesh')
if profiles[1].get('lower_path_osm') != '776358646' or len(truss) < 1000 or len(bridge_glass) < 100:
    raise ValueError('ZhouNan lower path / truss or Zhaonan glass railing is missing')
# 在下层 OSM 路线的五个位置核对上下桥面分离，且下桥板覆盖步行线路。
for x,z in ((409.4,138.0),(392.5,104.2),(365.7,50.5),(338.8,-3.2),(326.0,-26.0)):
    upper_and_walk=vertical_surface_hits(deck,x,z)
    lower_surfaces=vertical_surface_hits(lower_deck,x,z)
    if not upper_and_walk or not lower_surfaces:
        raise ValueError(f'ZhouNan bridge surface is missing at {(x,z)}')
    low=max(lower_surfaces)
    if (not 3.1 < max(upper_and_walk)-low < 3.8
            or low < terrain_at(x,z)):
        raise ValueError(f'ZhouNan pedestrian path does not align with the lower deck at {(x,z)}')
# 白石路正上方的下层净空需反映本次依用户反馈进行的抬高，不以此冒充实测值。
if max(vertical_surface_hits(lower_deck,409.4,138.0))-terrain_at(409.4,138.0) < 10.0:
    raise ValueError('ZhouNan bridge has not retained its raised road clearance')
if not any(math.hypot(v[0]-411,v[2]-137)<20 and v[1]-terrain_at(v[0],v[2])>5 for v in truss[::60]):
    raise ValueError('ZhouNan steel tubes are missing over the mapped road crossing')

# 致艺楼两塔及三层裙房相连，地形采样不能再把它们抬成互不相连的山包。
site_heights=[terrain_at(x,z) for x,z in ((274.1,376.7),(320.4,325.2),(335.3,360.1))]
if max(site_heights)-min(site_heights)>.3:
    raise ValueError('ZhiYi towers and podium do not share the estimated base platform')
zhiyi_deck=mesh_vertices('campus-zhiyi-skywalk.mesh')
zhiyi_rails=mesh_vertices('campus-zhiyi-skywalk-rails.mesh')
zhiyi_walls=mesh_vertices('campus-zhiyi-shell.mesh')
zhiyi_glass=mesh_vertices('campus-zhiyi-glass.mesh')
if len(zhiyi_rails)<100 or len(zhiyi_walls)<10000 or len(zhiyi_glass)<1000:
    raise ValueError('ZhiYi railings, recessed walls or glazing are missing')
for floor in (4,6,8):
    expected=4+floor*3.3+.03
    levels=vertical_surface_hits(zhiyi_deck,304.9,343.9)
    if not any(abs(y-expected)<.06 for y in levels):
        raise ValueError(f'ZhiYi skywalk deck is missing at estimated floor {floor}')

def segment_hits_triangles(start,end,vertices):
    # 对桥轴线作有限长度射线检测：连接处不应被塔楼实墙或玻璃封住。
    def sub(a,b): return tuple(x-y for x,y in zip(a,b))
    def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dot(a,b): return sum(x*y for x,y in zip(a,b))
    direction=sub(end,start)
    for i in range(0,len(vertices),3):
        a,b,c=(v[:3] for v in vertices[i:i+3])
        if any(max(a[j],b[j],c[j])<min(start[j],end[j])-.01 or
               min(a[j],b[j],c[j])>max(start[j],end[j])+.01 for j in range(3)):
            continue
        edge1,edge2=sub(b,a),sub(c,a)
        h=cross(direction,edge2)
        determinant=dot(edge1,h)
        if abs(determinant)<1e-9:
            continue
        inv=1.0/determinant
        s=sub(start,a)
        u=dot(s,h)*inv
        q=cross(s,edge1)
        v=dot(direction,q)*inv
        t=dot(edge2,q)*inv
        if -.0001<=u<=1.0001 and v>=-.0001 and u+v<=1.0001 and 0<=t<=1:
            return True
    return False

# 在每条连桥的通行高度检查两端门洞；点位按生成网格所在的地图墙线近似。
a,b=(303.0,347.5),(306.8,340.3)
length=math.hypot(b[0]-a[0],b[1]-a[1])
dx,dz=(b[0]-a[0])/length,(b[1]-a[1])/length
for floor in (4,6,8):
    y=4+floor*3.3+1.2
    start=(a[0]-dx*.5,y,a[1]-dz*.5)
    end=(b[0]+dx*.5,y,b[1]+dz*.5)
    if segment_hits_triangles(start,end,zhiyi_walls) or segment_hits_triangles(start,end,zhiyi_glass):
        raise ValueError(f'ZhiYi skywalk at floor {floor} is blocked by a wall or window')

# 两座主桥的桥面范围不能出现程序生成的树干；宽树冠另留出 5.8 米近似间距。
osm = ET.parse(DATA/'yuehai.osm').getroot()
node_xy = {n.get('id'):((float(n.get('lon'))-113.9318)*102800,
                        (22.5355-float(n.get('lat')))*111320) for n in osm.findall('node')}

# 致真楼的地图高架线从裙房两面墙进出；按通行高度检测墙、格架及桥面。
osm_ways = {w.get('id'):w for w in osm.findall('way')}
def mapped_points(osm_id):
    return [node_xy[n.get('ref')] for n in osm_ways[osm_id].findall('nd')]
def cross2(a,b):
    return a[0]*b[1]-a[1]*b[0]

# 2024 年照片中的周南桥下层入口应平顺接入人行层，而不是跃升至车行层。
def point_along(points,distance):
    for a,b in zip(points,points[1:]):
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if distance<=length:
            t=distance/length
            return (a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
        distance-=length
    return points[-1]

lower_route=mapped_points('776358646')
samples=[]
for distance in range(0,81,5):
    x,z=point_along(lower_route,distance)
    ground=terrain_at(x,z)
    hits=[y for y in vertical_surface_hits(deck,x,z)+vertical_surface_hits(lower_deck,x,z) if y>ground+.1]
    if not hits:
        raise ValueError(f'ZhouNan lower approach has a gap at {distance} m')
    # 下方新增地面接线后，竖直射线可能同时穿过不同层；跟踪连续的人行层。
    samples.append(min(hits,key=lambda y:abs(y-samples[-1])) if samples else min(hits))
if any(abs(b-a)>.72 for a,b in zip(samples,samples[1:])):
    raise ValueError('ZhouNan lower approach has an abrupt height change')
stair_profile=next(s for s in profiles[2]['mapped_stairs'] if s['osm']=='776358647')
stair_points=mapped_points('776358647')
sx,sz=stair_points[0]
floors=vertical_surface_hits(lower_deck,sx,sz)
if not floors or min(abs(y-stair_profile['high_estimate_m']) for y in floors)>.12:
    raise ValueError('ZhouNan stair entry is not connected to the lower deck')
end_x,end_z=point_along(stair_points,8.0)
start=(sx,stair_profile['high_estimate_m']+1.0,sz)
end=(end_x,stair_profile['high_estimate_m']+1.0-8.0*(stair_profile['high_estimate_m']-
     stair_profile['low_estimate_m'])/stair_profile['length_m'],end_z)
lower_rails=mesh_vertices('campus-bridge-rails.mesh')
if segment_hits_triangles(start,end,lower_rails) or segment_hits_triangles(start,end,truss):
    raise ValueError('ZhouNan stair entry is blocked by a railing or brace')
# 人行路线先转弯再进入桥下，逐段检测栏杆、斜撑及花盆，不能用直线穿越支撑代替路线。
pots=mesh_vertices('campus-zhounan-pots.mesh')
def lower_route_eye(distance):
    x,z=point_along(lower_route,distance)
    ground=terrain_at(x,z)
    hits=[y for y in vertical_surface_hits(deck,x,z)+vertical_surface_hits(lower_deck,x,z) if y>ground+.1]
    if not hits:
        raise ValueError('ZhouNan lower route is missing its walking surface')
    i=min(int(distance//5),len(samples)-2)
    t=(distance-i*5)/5
    expected=samples[i]*(1-t)+samples[i+1]*t
    return (x,min(hits,key=lambda y:abs(y-expected))+1.0,z)
for distance in range(36,65,2):
    start,end=lower_route_eye(distance),lower_route_eye(distance+2)
    if any(segment_hits_triangles(start,end,mesh) for mesh in (lower_rails,truss,pots)):
        raise ValueError(f'ZhouNan lower route is blocked at {distance} m')
    if 44<=distance<=54:
        # 入口转弯处还要留出行人头部高度，不只检查膝部附近。
        head_a=(start[0],start[1]+.7,start[2])
        head_b=(end[0],end[1]+.7,end[2])
        if any(segment_hits_triangles(head_a,head_b,mesh) for mesh in (lower_rails,truss)):
            raise ValueError(f'ZhouNan lower route has a head-height obstruction at {distance} m')

# 饭堂与步道间补齐的平台、外墙门洞及两条地图人行线交汇处应可连续通行。
canteen_floor=terrain_at(430,234)+4.2
canteen_roofs=mesh_vertices('campus-roofs.mesh')
canteen_glass=mesh_vertices('campus-canteen-glass.mesh')
canteen_frame=mesh_vertices('campus-canteen-frame.mesh')
for z in (233.9,235.3,236.0,237.0,239.0):
    levels=vertical_surface_hits(deck,430,z)+vertical_surface_hits(canteen_roofs,430,z)
    if not any(abs(y-canteen_floor)<.08 for y in levels):
        raise ValueError(f'South canteen connection has a floor gap at z={z}')
for offset in (.8,1.6):
    start,end=(430,canteen_floor+offset,233.9),(430,canteen_floor+offset,240.0)
    if any(segment_hits_triangles(start,end,mesh) for mesh in (deck,lower_rails,canteen_roofs,canteen_glass,canteen_frame)):
        raise ValueError('South canteen entrance is blocked by a railing or facade')
fork=mapped_points('984608623')
join=fork[0]
for offset in (.8,1.6):
    a=(join[0]-3.0,canteen_floor+offset,join[1])
    b=(join[0],canteen_floor+offset,join[1])
    x,z=point_along(lower_route,3.0)
    c=(x,lower_route_eye(3.0)[1]-1.0+offset,z)
    if any(segment_hits_triangles(p,q,mesh) for p,q in ((a,b),(b,c)) for mesh in (deck,lower_rails,truss)):
        raise ValueError('South canteen footway is blocked at the bridge junction')

# 饭堂南侧外廊、楼层入口和东侧折角的桥面应连续，不能被竖向格栅或护栏封住。
for z in (275.5,277.2,278.0,280.0,281.5,282.5):
    levels=vertical_surface_hits(deck,410,z)+vertical_surface_hits(canteen_roofs,410,z)
    if not any(abs(y-canteen_floor)<.08 for y in levels):
        raise ValueError(f'South canteen balcony has a floor gap at z={z}')
for offset in (.8,1.6):
    start,end=(410,canteen_floor+offset,282.4),(410,canteen_floor+offset,275.5)
    if any(segment_hits_triangles(start,end,mesh) for mesh in (deck,lower_rails,canteen_glass,canteen_frame)):
        raise ValueError('South canteen balcony entrance is blocked')
    ground=terrain_at(396,279)+.03
    start,end=(396,ground+offset,282.0),(396,ground+offset,275.0)
    if any(segment_hits_triangles(start,end,mesh) for mesh in (canteen_glass,canteen_frame)):
        raise ValueError('South canteen ground-floor entrance is blocked')
    start=(465,canteen_floor+offset,282.05)
    turn=(478.48,canteen_floor+offset,281.78)
    end=(478.46,canteen_floor+offset,277.78)
    if any(segment_hits_triangles(a,b,mesh) for a,b in ((start,turn),(turn,end)) for mesh in (deck,lower_rails,canteen_frame)):
        raise ValueError('South canteen balcony corner is blocked')
for x in (468,472,476,478.4):
    if not any(abs(y-canteen_floor)<.08 for y in vertical_surface_hits(deck,x,282.0)):
        raise ValueError(f'South canteen balcony corner is missing a deck at x={x}')
roof_detail=mesh_vertices('campus-canteen-roof.mesh')
if max(vertical_surface_hits(roof_detail,390,252))-terrain_at(390,252)<16.5:
    raise ValueError('South canteen roof housing is missing')
for x in (410,430,450):
    pavement=vertical_surface_hits(canteen_roofs,x,284)
    ground=terrain_triangle_at(x,284)
    if not any(.02 < y-ground < .18 for y in pavement):
        raise ValueError('South canteen entrance paving is missing or buried')

# 北馆两侧实际地图连廊须接通相邻建筑，桥板沿全线连续且高于下方道路。
library_links=profiles[2].get('north_library_walkways',[])
if {v['osm'] for v in library_links}!={'1274485928','231782436'}:
    raise ValueError('The two mapped north-library walkways are missing')
strip_walls=mesh_vertices('campus-wall-strip.mesh')
for link in library_links:
    a,b=mapped_points(link['osm'])[0],mapped_points(link['osm'])[-1]
    dx,dz=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dz)
    tx,tz=dx/length,dz/length
    level=link['deck_height_estimate_m']
    for t in (-.05,0,.25,.5,.75,1,1.05):
        x,z=a[0]+dx*t,a[1]+dz*t
        if not any(abs(y-level)<.08 for y in vertical_surface_hits(deck,x,z)):
            raise ValueError(f'North-library walkway {link["osm"]} has a gap near {t:.2f}')
    if level-terrain_at((a[0]+b[0])/2,(a[1]+b[1])/2)<3:
        raise ValueError(f'North-library walkway {link["osm"]} intersects the road below')
    endpoint,into=(a,(-tx,-tz)) if link['osm']=='1274485928' else (b,(tx,tz))
    start=(endpoint[0]-into[0]*1.3,level+1.5,endpoint[1]-into[1]*1.3)
    end=(endpoint[0]+into[0]*1.3,level+1.5,endpoint[1]+into[1]*1.3)
    if segment_hits_triangles(start,end,strip_walls):
        raise ValueError(f'North-library walkway {link["osm"]} is blocked at its adjacent building')
    if link['osm']=='231782436' and any(segment_hits_triangles(start,end,mesh_vertices(name))
            for name in ('campus-huizi-shell.mesh','campus-huizi-ground.mesh')):
        raise ValueError('The HuiZi arcade or roof details block the north-library walkway')
    if link['osm']=='1274485928' and any(segment_hits_triangles(start,end,mesh_vertices(name))
            for name in ('campus-huiyuan-shell.mesh','campus-huiyuan-ground.mesh')):
        raise ValueError('The HuiYuan arcade or roof details block the north-library walkway')

# 汇元楼门廊对齐地图中的人行路线；沿通道中心和两侧检测无遮挡且铺装贴地。
entry_a,entry_b=mapped_points('237893526')
wall_a,wall_b=mapped_points('230847104')[12:14]
ex,ez=entry_b[0]-entry_a[0],entry_b[1]-entry_a[1]
wx,wz=wall_b[0]-wall_a[0],wall_b[1]-wall_a[1]
fraction=cross2((wall_a[0]-entry_a[0],wall_a[1]-entry_a[1]),(wx,wz))/cross2((ex,ez),(wx,wz))
entry=(entry_a[0]+ex*fraction,entry_a[1]+ez*fraction)
length=math.hypot(ex,ez)
tx,tz=ex/length,ez/length
base=terrain_at(-200.4,-23.35)
office_parts=[mesh_vertices(name) for name in ('campus-huiyuan-shell.mesh','campus-huiyuan-ground.mesh',
               'campus-huiyuan-sign.mesh','campus-huiyuan-motto.mesh')]
for side in (-1,0,1):
    x,z=entry[0]-tz*side,entry[1]+tx*side
    start=(x+tx*7.1,base+1.5,z+tz*7.1)
    end=(x-tx*7,base+1.5,z-tz*7)
    if any(segment_hits_triangles(start,end,mesh) for mesh in [strip_walls]+office_parts):
        raise ValueError('HuiYuan entrance is blocked along the mapped footpath')
office_paving=mesh_vertices('campus-huiyuan-paving.mesh')
for distance in (0,2,4,6,7):
    x,z=entry[0]+tx*distance,entry[1]+tz*distance
    levels=vertical_surface_hits(office_paving,x,z)
    if not levels or not .02 < max(levels)-terrain_at(x,z) < .35:
        raise ValueError('HuiYuan entrance paving is buried or floating above the terrain')

# 汇紫楼三段主屋面按用户模型的相对关系依次抬升；数值为暂定估计。
huizi_shell=mesh_vertices('campus-huizi-shell.mesh')
huizi_ground=mesh_vertices('campus-huizi-ground.mesh')
huizi_canopy=mesh_vertices('campus-huizi-canopy.mesh')
huizi_paving=mesh_vertices('campus-huizi-paving.mesh')
huizi_roof_mesh=mesh_vertices('campus-huizi-roof.mesh')
huizi_equipment=mesh_vertices('campus-huizi-equipment.mesh')
campus_roofs=mesh_vertices('campus-roofs.mesh')
anchor_roofs=vertical_surface_hits(huizi_roof_mesh,30,25)
if not anchor_roofs:
    raise ValueError('The HuiZi A main roof is missing')
huizi_roof=max(anchor_roofs)
sections=[]
for x,z in ((30,25),(6,-35),(-64,-70),(-50,-100)):
    tops=vertical_surface_hits(huizi_roof_mesh,x,z)
    if not tops: raise ValueError('A HuiZi roof section is missing')
    sections.append(max(tops))
if (abs((sections[1]-sections[0])-3.3)>.1 or abs((sections[2]-sections[1])-4.0)>.1
        or abs(sections[3]-sections[2])>.1):
    raise ValueError(f'HuiZi roof level relationships are incorrect: {sections}')
huizi_north_roof=sections[2]
for x,z in ((-14,-34),(-39,-76),(4,5)):
    if vertical_surface_hits(huizi_roof_mesh,x,z):
        raise ValueError('A HuiZi courtyard was accidentally covered by the main roof')
covered=vertical_surface_hits(huizi_canopy,-39,-76)
if not covered or abs(covered[0]-(huizi_north_roof-7.2))>.1:
    raise ValueError('The lower gridded area in the northern HuiZi courtyard is missing')
# 航拍可见的两组屋顶设备应有各自的屋面和支撑，不能悬在中庭上方。
for x,z in ((-36.4,-47.7),(-46.6,-45.1)):
    base=vertical_surface_hits(huizi_roof_mesh,x,z)
    cap=vertical_surface_hits(campus_roofs,x,z)
    if (not any(abs(y-huizi_north_roof)<.1 for y in base)
            or not any(abs(y-(huizi_north_roof+2.55))<.1 for y in cap)):
        raise ValueError('A HuiZi rooftop equipment unit is missing or outside the roof')
if (len(huizi_equipment)<48 or min(v[1] for v in huizi_equipment)<huizi_north_roof+.6
        or max(v[1] for v in huizi_equipment)>huizi_north_roof+2.6):
    raise ValueError('HuiZi rooftop equipment geometry is incomplete')

# A 座外柱与退进底墙之间必须有真正的空间，不能仅用平面纹理冒充柱廊。
a,b=mapped_points('230847102')[1:3]
dx,dz=b[0]-a[0],b[1]-a[1]
length=math.hypot(dx,dz)
nx,nz=-dz/length,dx/length
for t in (1/6,3/6,5/6):
    x,z=a[0]+dx*t,a[1]+dz*t
    y=huizi_roof-13.2+1.5
    def arcade_point(offset):
        return (x+nx*offset,y,z+nz*offset)
    if any(segment_hits_triangles(arcade_point(.5),arcade_point(-1.7),mesh)
           for mesh in (strip_walls,huizi_shell,huizi_ground)):
        raise ValueError('HuiZi colonnade is blocked before the recessed ground-floor wall')
    if not segment_hits_triangles(arcade_point(-1.7),arcade_point(-3.1),huizi_ground):
        raise ValueError('HuiZi recessed ground-floor facade is missing')

# 照片近似的入口平台及四级踏步应露出地面，且向外逐级下降。
a,b=mapped_points('230847102')[0],mapped_points('230847102')[2]
dx,dz=b[0]-a[0],b[1]-a[1]
length=math.hypot(dx,dz)
nx,nz=-dz/length,dx/length
step_levels=[]
for distance in (2,5.4,6.2,7.0,7.8):
    x,z=a[0]+dx*.4+nx*distance,a[1]+dz*.4+nz*distance
    levels=vertical_surface_hits(huizi_paving,x,z)
    if not levels or max(levels)<terrain_at(x,z)+.04:
        raise ValueError('HuiZi entrance landing or steps are buried in the terrain')
    step_levels.append(max(levels))
if any(left-right<.1 for left,right in zip(step_levels,step_levels[1:])):
    raise ValueError('HuiZi entrance steps do not descend from the landing')

# 东侧道路仍应连续接入两级场地；这里检验的是估计路面的平顺性，不代表实测坡度。
road=mapped_points('232096325')[:4]
for a,b in zip(road,road[1:]):
    length=math.hypot(b[0]-a[0],b[1]-a[1]);count=max(1,math.ceil(length/2))
    samples=[terrain_at(a[0]+(b[0]-a[0])*i/count,a[1]+(b[1]-a[1])*i/count)
             for i in range(count+1)]
    if any(abs(right-left)/(length/count)>.15 for left,right in zip(samples,samples[1:])):
        raise ValueError('The HuiZi east-side road has an abrupt terrain transition')
ground_a=terrain_at(24.3,4.4);ground_b=terrain_at(4.1,-34.1)
ground_c=terrain_at(-16.1,-71.4);ground_d=terrain_at(-44.2,-101)
if (abs(ground_a-ground_b)>.3 or abs(ground_c-ground_d)>.3
        or not 3.5 < (ground_c-ground_b) < 4.5):
    raise ValueError('The estimated HuiZi terrace heights are inconsistent')

# 演会中心采用真实屋顶轮廓、半开放侧面、屋架和照片近似的观众席。
hall_roof=mesh_vertices('campus-yanhui-roof.mesh')
hall_steel=mesh_vertices('campus-yanhui-steel.mesh')
hall_stone=mesh_vertices('campus-yanhui-stone.mesh')
hall_floor=mesh_vertices('campus-yanhui-floor.mesh')
hall_seats=mesh_vertices('campus-yanhui-seats.mesh')
hall_steps=mesh_vertices('campus-yanhui-steps.mesh')
hall_outline=mapped_points('1455962776')
if len(hall_steel)<3000 or len(hall_seats)<3000 or max(v[1] for v in hall_roof)-min(v[1] for v in hall_roof)<1:
    raise ValueError('Yanhui roof, steel trusses or seats are missing')
hall_base=terrain_at(-256.75,-126.77)
if segment_hits_triangles((-289,hall_base+4,-125),(-224,hall_base+4,-125),strip_walls):
    raise ValueError('Yanhui is closed by the old opaque building facade')

# 楼前地图台阶延伸至最近屋顶边界，使用渲染三角面的高程核对全宽无遮挡。
start,end=mapped_points('231885484')
closest=None
for a,b in zip(hall_outline[:-1],hall_outline[1:]):
    dx,dz=b[0]-a[0],b[1]-a[1]
    length2=dx*dx+dz*dz
    t=max(0,min(1,((end[0]-a[0])*dx+(end[1]-a[1])*dz)/length2))
    point=(a[0]+dx*t,a[1]+dz*t)
    distance=math.hypot(point[0]-end[0],point[1]-end[1])
    if closest is None or distance<closest[0]: closest=(distance,point)
end=closest[1]
dx,dz=end[0]-start[0],end[1]-start[1]
length=math.hypot(dx,dz)
nx,nz=-dz/length,dx/length
for offset in (-2.8,0,2.8):
    heights=[]
    for t in (.05,.2,.4,.6,.8,.95):
        x,z=start[0]+dx*t+nx*offset,start[1]+dz*t+nz*offset
        levels=vertical_surface_hits(hall_steps,x,z)
        if not levels or max(levels)<terrain_triangle_at(x,z)+.025:
            raise ValueError('Yanhui entrance stairs are buried in the rendered terrain')
        heights.append(max(levels))
    if any(b<a-.02 for a,b in zip(heights,heights[1:])) or heights[-1]-heights[0]<.8:
        raise ValueError('Yanhui entrance stairs are not a continuous rising path')

# 后排中央通道应从室外入口逐渐升至座位区，不能由一整面高墙拦住。
twice_area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(hall_outline[:-1],hall_outline[1:]))
cx=sum((a[0]+b[0])*(a[0]*b[1]-b[0]*a[1]) for a,b in zip(hall_outline[:-1],hall_outline[1:]))/(3*twice_area)
cz=sum((a[1]+b[1])*(a[0]*b[1]-b[0]*a[1]) for a,b in zip(hall_outline[:-1],hall_outline[1:]))/(3*twice_area)
dx,dz=hall_outline[1][0]-hall_outline[0][0],hall_outline[1][1]-hall_outline[0][1]
length=math.hypot(dx,dz)
vx,vz=-dz/length,dx/length
aisle_heights=[]
for v in (26.8,24,21.5,19,18.3):
    levels=vertical_surface_hits(hall_floor,cx+vx*v,cz+vz*v)
    if not levels: raise ValueError('Yanhui rear aisle has a missing floor segment')
    aisle_heights.append(max(levels))
if any(b<a-.03 for a,b in zip(aisle_heights,aisle_heights[1:])) or aisle_heights[-1]-aisle_heights[0]<1:
    raise ValueError('Yanhui rear aisle is disconnected from the raised seats')

# 门前常绿树不应占据地图上的入口台阶。
pines=mesh_vertices('campus-yanhui-pine-trunks.mesh')
a,b=mapped_points('231885484')
dx,dz=b[0]-a[0],b[1]-a[1]
length2=dx*dx+dz*dz
for p in pines:
    t=max(0,min(1,((p[0]-a[0])*dx+(p[2]-a[1])*dz)/length2))
    if math.hypot(p[0]-(a[0]+dx*t),p[2]-(a[1]+dz*t))<3.2:
        raise ValueError('Yanhui entrance pine blocks the mapped staircase')

# 时光广场与汇元楼入口属于连续铺装空间，绿地缘石应露出地表。
square_points=((-241.85,-47.02),(-256,-47),(-228,-47),(-242,-61),(-242,-33),(-224,-23))
square_levels=[terrain_at(x,z) for x,z in square_points]
if max(square_levels)-min(square_levels)>.35:
    raise ValueError('Time Square or HuiYuan entry has an unintended terrain step')
square_curb=mesh_vertices('campus-time-curb.mesh')
if any(v[1]<terrain_triangle_at(v[0],v[2])+.02 for v in square_curb[::3] if v[4]>.7):
    raise ValueError('Time Square lawn curb is buried in the terrain')
if len(mesh_vertices('campus-time-hedges.mesh'))<100:
    raise ValueError('Time Square radial hedges are missing')
if len(mesh_vertices('campus-time-paving.mesh'))<100 or len(mesh_vertices('campus-time-hedge-leaves.mesh'))<100:
    raise ValueError('Time Square paving or hedge foliage is missing')

# 北馆前高台阶的两端及横向边缘都应露出修正后的粗地形。
north_entry_steps=mesh_vertices('campus-north-library-entry-steps.mesh')
a,b=mapped_points('232602597')[0],mapped_points('232602597')[-1]
dx,dz=b[0]-a[0],b[1]-a[1]
length=math.hypot(dx,dz)
nx,nz=-dz/length,dx/length
for t in (.1,.3,.5,.7,.9):
    for offset in (-8,0,8):
        x,z=a[0]+dx*t+nx*offset,a[1]+dz*t+nz*offset
        levels=vertical_surface_hits(north_entry_steps,x,z)
        if not levels or max(levels)<terrain_at(x,z)-.05:
            raise ValueError('North-library entrance stairs are missing or buried in the terrain')

# 致理楼悬挑体应位于地图给出的平面范围内，并高于演播厅；幕墙需要真实退进白框。
crown = mesh_vertices('campus-zhili-crown.mesh')
crown_glass = mesh_vertices('campus-zhili-crown-glass.mesh')
crown_frame = mesh_vertices('campus-zhili-crown-frame.mesh')
studio_shell = mesh_vertices('campus-studio-shell.mesh')
base=terrain_at(393.4,432.5)
if (not crown_frame or abs(min(v[1] for v in crown)-(base+8*3.3))>.15
        or abs(max(v[1] for v in crown)-(base+13*3.3))>.15
        or min(v[1] for v in crown)-max(v[1] for v in studio_shell)<10
        or any(not (368.5<v[0]<398 and 383<v[2]<405) for v in crown)):
    raise ValueError('ZhiLi cantilever is missing, outside its mapped footprint, or too low over the studio')
corner_a,corner_b = mapped_points('776358623')[1:3]
edge_dx,edge_dz=corner_b[0]-corner_a[0],corner_b[1]-corner_a[1]
edge_length=math.hypot(edge_dx,edge_dz)
inset=[abs((v[0]-corner_a[0])*edge_dz-(v[2]-corner_a[1])*edge_dx)/edge_length for v in crown_glass]
if not inset or min(inset)<4.8 or max(inset)>5.2:
    raise ValueError('ZhiLi cantilever glazing must be recessed behind the large front frame')

# 校园高架楼梯使用 OSM 线位；逐点检查桥面—踏步—地面的高度连续与可见踏步。
stairs = {item['osm']:item for item in profiles[2].get('mapped_stairs',[])}
for osm_id in ('1272661382','1273971272'):
    if osm_id not in stairs:
        raise ValueError(f'Mapped stair {osm_id} was not generated')
    info=stairs[osm_id]
    line=mapped_points(osm_id)
    if len(line)!=2 or info['steps_estimate']<10:
        raise ValueError(f'Mapped stair {osm_id} has no usable tread profile')
    (x0,z0),(x1,z1)=line
    observed=[]
    for t in (0.02,.2,.4,.6,.8,.98):
        x,z=x0+(x1-x0)*t,z0+(z1-z0)*t
        expected=info['high_estimate_m']*(1-t)+info['low_estimate_m']*t
        hits=vertical_surface_hits(deck,x,z)
        candidates=[y for y in hits if abs(y-expected)<.3 and y>=terrain_at(x,z)-.15]
        if not candidates:
            raise ValueError(f'Mapped stair {osm_id} is missing or buried at {(x,z)}')
        observed.append(min(candidates,key=lambda y:abs(y-expected)))
    if len({round(y,1) for y in observed})<5:
        raise ValueError(f'Mapped stair {osm_id} is not stepped')
    if abs(info['low_estimate_m']-terrain_at(x1,z1))>.2:
        raise ValueError(f'Mapped stair {osm_id} does not connect to the estimated ground')

# 在高端转角按人体腰部高度采样，确认相交连廊、楼梯护墙及支柱未把入口封住。
bridge_rails = mesh_vertices('campus-bridge-rails.mesh')
bridge_support = mesh_vertices('campus-bridge-support.mesh')
for stair_id,path_id in (('1273971272','1273971271'),('1272661382','1272661381')):
    stair_path,approach = mapped_points(stair_id),mapped_points(path_id)
    p,q=stair_path
    a,b=approach[-2:]
    stair_length,approach_length=math.dist(p,q),math.dist(a,b)
    dx,dz=(q[0]-p[0])/stair_length,(q[1]-p[1])/stair_length
    ax,az=(b[0]-a[0])/approach_length,(b[1]-a[1])/approach_length
    high=stairs[stair_id]['high_estimate_m']
    start=(p[0]-1.5*ax,high+.60,p[1]-1.5*az)
    end=(p[0]+1.5*dx,high+.40,p[1]+1.5*dz)
    if any(segment_hits_triangles(start,end,mesh) for mesh in (deck,bridge_rails,bridge_support)):
        raise ValueError(f'Mapped stair {stair_id} is blocked where it turns onto the elevated walkway')

# 三个共用 OSM 节点的高架分支应从主通道两个方向都能进入，护墙不能横穿分支。
main_path=mapped_points('776358635')
for index,path_id,branch_index in ((4,'1272661381',1),(4,'1273971271',1),(5,'987586196',-2)):
    p=main_path[index]
    branch=mapped_points(path_id)[branch_index]
    branch_length=math.dist(p,branch)
    dx,dz=(branch[0]-p[0])/branch_length,(branch[1]-p[1])/branch_length
    y=terrain_at(*p)+4.8
    for neighbor in (main_path[index-1],main_path[index+1]):
        length=math.dist(p,neighbor)
        ax,az=(neighbor[0]-p[0])/length,(neighbor[1]-p[1])/length
        start=(p[0]+3*ax,y,p[1]+3*az)
        end=(p[0]+3*dx,y,p[1]+3*dz)
        if any(segment_hits_triangles(start,end,mesh) for mesh in (deck,bridge_rails,bridge_support)):
            raise ValueError(f'Junction from main walkway to {path_id} is blocked')

zhizhen_path = mapped_points('776358635')
zhizhen_outline = mapped_points('629940892')
zhizhen_podium = mesh_vertices('campus-zhizhen-podium.mesh')
zhizhen_frame = mesh_vertices('campus-zhizhen-frame.mesh')
intersections = []
for p,q in zip(zhizhen_path[:-1],zhizhen_path[1:]):
    v = (q[0]-p[0],q[1]-p[1])
    for a,b in zip(zhizhen_outline[:-1],zhizhen_outline[1:]):
        w = (b[0]-a[0],b[1]-a[1])
        den = cross2(v,w)
        if abs(den)<1e-8:
            continue
        diff = (a[0]-p[0],a[1]-p[1])
        t,u = cross2(diff,w)/den,cross2(diff,v)/den
        if 0<t<1 and 0<u<1:
            x,z = p[0]+t*v[0],p[1]+t*v[1]
            length = math.hypot(*v)
            dx,dz = v[0]/length,v[1]/length
            y = terrain_at(x,z)+5.5
            start,end = (x-dx,y,z-dz),(x+dx,y,z+dz)
            if any(segment_hits_triangles(start,end,mesh) for mesh in (zhizhen_podium,zhizhen_frame)):
                raise ValueError('ZhiZhen mapped elevated walkway is blocked by a wall or frame')
            surfaces = vertical_surface_hits(deck,x,z)
            if not any(abs(level-terrain_at(x,z)-4.2)<.35 for level in surfaces):
                raise ValueError('ZhiZhen mapped elevated walkway has no aligned deck')
            intersections.append((x,z))
if len(intersections)!=2:
    raise ValueError(f'Expected two ZhiZhen walkway wall openings, found {len(intersections)}')

# 致理楼 L2 与 L3 的地图步行路径也穿过外墙，检查代表性的进出口而非只检查贴图。
zhili_glass = mesh_vertices('campus-zhili-glass.mesh')
zhili_frame = mesh_vertices('campus-zhili-frame.mesh')
zhili_side = mesh_vertices('campus-zhili-side.mesh')
zhili_l3_walls = tuple(mesh_vertices(f'campus-zhizhen-{part}.mesh') for part in ('shell','glass','side'))
for path_id,x,z,blocking in (
        ('776358635',309.4484,438.1335,(zhili_glass,zhili_frame,zhili_side)),
        ('776358635',310.6668,440.8568,(zhili_glass,zhili_frame,zhili_side)),
        ('987586196',372.3410,434.7449,(zhili_glass,zhili_frame,zhili_side)),
        ('1272661381',396.5115,398.3861,(*zhili_l3_walls,zhizhen_podium,zhizhen_frame)),
        ('1272661381',397.7290,399.5037,(*zhili_l3_walls,zhizhen_podium,zhizhen_frame))):
    path = mapped_points(path_id)
    choices=[]
    for a,b in zip(path[:-1],path[1:]):
        dx,dz=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dz)
        t=max(0,min(1,((x-a[0])*dx+(z-a[1])*dz)/(length*length)))
        choices.append((math.hypot(x-a[0]-t*dx,z-a[1]-t*dz),dx/length,dz/length))
    distance,dx,dz=min(choices)
    if distance>.05:
        raise ValueError(f'ZhiLi crossing no longer matches mapped walkway {path_id}')
    y=terrain_at(x,z)+5.5
    if any(segment_hits_triangles((x-dx,y,z-dz),(x+dx,y,z+dz),mesh) for mesh in blocking):
        raise ValueError(f'ZhiLi walkway {path_id} is blocked near {(x,z)}')
    if not any(abs(level-terrain_at(x,z)-4.2)<.35 for level in vertical_surface_hits(deck,x,z)):
        raise ValueError(f'ZhiLi walkway {path_id} has no aligned deck near {(x,z)}')
for filename in ('campus-zhizhen-shell.mesh','campus-zhizhen-glass.mesh','campus-zhizhen-side.mesh',
                 'campus-studio-shell.mesh','campus-studio-glass.mesh','campus-studio-sign.mesh',
                 'campus-zhili-glass.mesh','campus-zhili-frame.mesh'):
    mesh_vertices(filename)

# 地图中的 L1—L3 高架线应有分层桥板，并在两端短墙上保留通行开口。
zhili_high_deck=mesh_vertices('campus-zhili-high-walkways.mesh')
zhili_high_rails=mesh_vertices('campus-zhili-high-walkway-rails.mesh')
zhili_l1_shell=mesh_vertices('campus-zhili-a-shell.mesh')
zhili_l1_glass=mesh_vertices('campus-zhili-a-glass.mesh')
zhili_l1_frame=mesh_vertices('campus-zhili-a-frame.mesh')
zhili_l1_sign=mesh_vertices('campus-zhili-a-sign.mesh')
if (len(zhili_high_rails)<100 or len(zhili_l1_shell)<300
        or len(zhili_l1_glass)<100 or len(zhili_l1_frame)<100 or len(zhili_l1_sign)!=6):
    raise ValueError('ZhiLi L1 facade or high walkway resources are missing')
blocking=(*zhili_l3_walls,zhizhen_podium,zhizhen_frame)
for path_id,floor in (('1272661380',2),('1272661379',4),('1272661378',6)):
    a,b=mapped_points(path_id)
    dx,dz=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dz)
    dx,dz=dx/length,dz/length
    mid=((a[0]+b[0])/2,(a[1]+b[1])/2)
    y=terrain_at(343,512)+(floor-1)*3.3
    if not any(abs(level-y-.03)<.06 for level in vertical_surface_hits(zhili_high_deck,*mid)):
        raise ValueError(f'ZhiLi high walkway deck missing: way/{path_id}')
    for x,z in (a,b):
        start,end=(x-dx*.7,y+1.2,z-dz*.7),(x+dx*.7,y+1.2,z+dz*.7)
        if any(segment_hits_triangles(start,end,mesh) for mesh in blocking):
            raise ValueError(f'ZhiLi high walkway is blocked at {(x,z)}: way/{path_id}')

# L1 裙房入口应是后退的凹廊，沿进门方向不应被外墙封死。
outline=mapped_points('776358622')
a,b=outline[6],outline[0]
dx,dz=b[0]-a[0],b[1]-a[1]
length=math.hypot(dx,dz)
nx,nz=dz/length,-dx/length
x,z=a[0]+.59*dx,a[1]+.59*dz
y=terrain_at(x,z)+2
if segment_hits_triangles((x+nx,y,z+nz),(x-nx,y,z-nz),zhili_l1_shell):
    raise ValueError('ZhiLi L1 recessed entrance is blocked by a solid wall')

bridge_ways = {w.get('id'):w for w in osm.findall('way') if w.get('id') in ('776358616','574956548')}
with (ASSETS/'campus-trees.bin').open('rb') as f:
    count, = struct.unpack('<I',f.read(4))
    tree_data = f.read()
if len(tree_data) != count*16:
    raise ValueError('Truncated campus tree instances')
trees = list(struct.iter_unpack('<4f',tree_data))
# L1 入口前留出步行通道，避免程序随机树遮住凹廊。
for tx,_,tz,_ in trees:
    along=((tx-a[0])*dx+(tz-a[1])*dz)/(length*length)
    outward=(tx-a[0])*nx+(tz-a[1])*nz
    if .43 <= along <= .76 and -1 <= outward <= 19:
        raise ValueError('A procedural tree blocks the ZhiLi L1 entrance approach')
for stair_id in ('1272661382','1273971272'):
    a,b=mapped_points(stair_id)
    dx,dz=b[0]-a[0],b[1]-a[1]
    length2=dx*dx+dz*dz
    for x,_,z,_ in trees:
        t=max(0,min(1,((x-a[0])*dx+(z-a[1])*dz)/length2))
        if math.hypot(x-a[0]-t*dx,z-a[1]-t*dz)<2.5:
            raise ValueError(f'A procedural tree blocks mapped stair {stair_id}')
for osm_id,minimum in (('776358616',9.3),('574956548',13.3)):
    points = [node_xy[n.get('ref')] for n in bridge_ways[osm_id].findall('nd')]
    for x,_,z,_ in trees:
        for a,b in zip(points[:-1],points[1:]):
            dx,dz=b[0]-a[0],b[1]-a[1]
            length2=dx*dx+dz*dz
            if length2 == 0:
                continue
            u=max(0,min(1,((x-a[0])*dx+(z-a[1])*dz)/length2))
            if math.hypot(x-a[0]-u*dx,z-a[1]-u*dz) < minimum-.01:
                raise ValueError(f'Tree overlaps the mapped main bridge: way/{osm_id}')

# 两岸的亲水台阶应具有多级高度、露出粗地形，且不被随机植被占据。
terraces=mesh_vertices('campus-wenshan-terraces.mesh')
paving_detail=mesh_vertices('campus-wenshan-paving-detail.mesh')
if len(paving_detail)<100:
    raise ValueError('Wenshan paving detail is missing')
for name,bounds in (('west',(-395,-360,174,205)),('east',(-250,-220,193,229))):
    x0,x1,z0,z1=bounds
    tops=[v for v in terraces if x0<v[0]<x1 and z0<v[2]<z1 and v[4]>.7]
    if len(tops)<100 or len({round(v[1],2) for v in tops})<4:
        raise ValueError(f'Wenshan {name} terrace is missing its stepped surfaces')
    if any(v[1] < terrain_triangle_at(v[0],v[2])-.02 for v in tops):
        raise ValueError(f'Wenshan {name} terrace is buried by the terrain')
if any(math.hypot(x+377,z-190)<22.9 for x,_,z,_ in trees):
    raise ValueError('Procedural trees block the Wenshan west terrace')
if any(((x+231)/15)**2+((z-211)/21)**2<1 for x,_,z,_ in trees):
    raise ValueError('Procedural trees block the Wenshan east terrace')
# 防止湖面与东侧环路间的粗网格重新出现近乎垂直的地形折坡；阈值仅作几何回归。
for a,b in (((-210,190),(-200,190)),((-220,230),(-210,230)),
            ((-210,200),(-200,200)),((-210,220),(-200,220))):
    if abs(terrain_triangle_at(*a)-terrain_triangle_at(*b))>7.0:
        raise ValueError(f'Wenshan east-bank terrain has an abrupt step between {a} and {b}')
# 汇智楼前保留照片中的棕榈行道树带，避免随机宽叶树穿进相同位置。
ax,az,bx,bz=-110.0,117.0,-25.0,100.0
dx,dz=bx-ax,bz-az
for x,_,z,_ in trees:
    t=max(0,min(1,((x-ax)*dx+(z-az)*dz)/(dx*dx+dz*dz)))
    if math.hypot(x-ax-t*dx,z-az-t*dz)<6.49:
        raise ValueError('Procedural broadleaf tree overlaps the South Library palm row')

with (DATA/'building-index.csv').open(encoding='utf-8-sig',newline='') as f:
    buildings = {row['osm']:row for row in csv.DictReader(f)}
if buildings['way/230847105']['height_m']!='25.0':
    raise ValueError('North-library building height should match the university inspection notice')
# 汇萃楼总平面是低层大底座与沿东、南边延伸的高层 L 形，不能退回中央方塔。
if (buildings['way/938357240']['height_m']!='44.01'
        or buildings['way/1271852286']['height_m']!='46.8'
        or buildings['way/1271852287']['height_m']!='57.7'):
    raise ValueError('Art complex or neighboring tower heights no longer match their documented basis')
if (buildings['way/1081134194']['height_m']!='19.0'
        or buildings['way/1207358023']['height_m']!='20.45'
        or buildings['way/1455957244']['height_m']!='36.75'):
    raise ValueError('Huiyi original, five-storey extension and teaching lab were mixed up')
art_a,art_b,art_c,art_d=mapped_points('938357240')[:4]
ex,ez=art_b[0]-art_a[0],art_b[1]-art_a[1]
sx,sz=art_d[0]-art_a[0],art_d[1]-art_a[1]
det=ex*sz-ez*sx
def art_uv(x,z):
    dx,dz=x-art_a[0],z-art_a[1]
    return ((dx*sz-dz*sx)/det,(ex*dz-ez*dx)/det)
art_base=terrain_at(52.8,-641.7)
art_upper=mesh_vertices('campus-huicui-upper.mesh')
art_shell=mesh_vertices('campus-huicui-shell.mesh')
art_glass=mesh_vertices('campus-huicui-glass.mesh')
art_frames=mesh_vertices('campus-huicui-frames.mesh')
art_stage=mesh_vertices('campus-huicui-stage.mesh')
art_lights=mesh_vertices('campus-huicui-skylights.mesh')
if (abs(min(v[1] for v in art_upper)-(art_base+15))>.8
        or abs(max(v[1] for v in art_upper)-(art_base+42.51))>.8
        or any(not (-.02<u<1.02 and -.02<w<1.02 and (u>.88 or w>.78))
               for v in art_upper for u,w in [art_uv(v[0],v[2])])
        or not art_stage or not art_lights):
    raise ValueError('Huicui L-shaped upper wing, low rooftop stages or skylights are missing')

# 总平面标出的两个北侧入口需在外墙和第一层幕墙留出通道。
edge_length=math.hypot(ex,ez)
nx,nz=ez/edge_length,-ex/edge_length
for t in (.29,.59):
    x,z=art_a[0]+ex*t,art_a[1]+ez*t
    start=(x+nx*.6,art_base+1.6,z+nz*.6)
    end=(x-nx*1.6,art_base+1.6,z-nz*1.6)
    if any(segment_hits_triangles(start,end,mesh) for mesh in (art_shell,art_glass,art_frames)):
        raise ValueError(f'Huicui entrance near {t:.2f} is blocked at the exterior')
    # 内退门厅背景应覆盖整个开口及斜视角，不能露出楼后的地面。
    for offset in (-2.0,0.0,2.0):
        bx,bz=x+ex/edge_length*offset,z+ez/edge_length*offset
        inner=(bx-nx*3.7,art_base+1.6,bz-nz*3.7)
        behind=(bx-nx*4.3,art_base+1.6,bz-nz*4.3)
        if not segment_hits_triangles(inner,behind,art_glass):
            raise ValueError(f'Huicui entrance near {t:.2f} lacks a recessed lobby backdrop')
    if abs(terrain_at(x+nx*5,z+nz*5)-art_base)>1.0:
        raise ValueError(f'Huicui entrance near {t:.2f} is disconnected from its estimated forecourt')

wyz=[v for v in art_shell if -95.3<v[0]<-45.1 and -628.8<v[2]<-577.0]
if not wyz or abs(max(v[1] for v in wyz)-terrain_at(-70.1,-602.7)-46.8)>1.0:
    raise ValueError('Wu Yuzhang building no longer has the 13-storey estimated tower')
# 建规学院九层教学实验楼的最高屋顶格架和五层扩建楼均位于各自轮廓内。
lab=[v for v in art_frames if 132<v[0]<158.2 and -682.1<v[2]<-620.3]
if not lab or abs(max(v[1] for v in lab)-terrain_at(145,-650)-36.75)>1.0:
    raise ValueError('Huiyi teaching lab lacks its documented nine-storey roof height')
extension=[v for v in art_shell if 158.4<v[0]<173 and -682.1<v[2]<-620.3]
if (not 158<float(buildings['way/1207358023']['x_m'])<174
        or not extension or abs(max(v[1] for v in extension)-terrain_at(165,-650)-20.45)>1.0):
    raise ValueError('Huiyi five-storey extension overlaps the lab or has an incorrect roof height')
if not any(159<v[0]<172 and -676<v[2]<-632 for v in art_lights):
    raise ValueError('Huiyi five-storey extension roof skylights are missing')
# 老院馆南侧悬挑下须形成柱廊，而不是原先完整的实心立面。
old_base=terrain_at(202,-630)
old_start,old_end=(202,old_base+6,-618),(202,old_base+6,-624)
if any(segment_hits_triangles(old_start,old_end,mesh)
       for mesh in (mesh_vertices('campus-wall-strip.mesh'),art_shell,art_glass,art_frames)):
    raise ValueError('The approximate open colonnade at the old Huiyi building is blocked')
if buildings['way/230847103']['height_m'] != '95.0':
    raise ValueError('Huixing height must use the design institute value')
huixing_steps=[v for v in mesh_vertices('campus-huixing-steps.mesh') if v[4]>.7]
huixing_hedges=mesh_vertices('campus-huixing-hedges.mesh')
if (len(huixing_steps)<100 or len(huixing_hedges)<300
        or max(v[1] for v in huixing_steps)-min(v[1] for v in huixing_steps)<2
        or any(not (-19 < v[0] < -10 and -346 < v[2] < -331) for v in huixing_steps)
        or any(v[1]-terrain_at(v[0],v[2])<.02 for v in huixing_steps)):
    raise ValueError('Huixing mapped stairs or garden geometry is missing or buried')
if any(-42<x<14 and -364<z<-345 for x,_,z,_ in trees):
    raise ValueError('Procedural trees block the Huixing north gardens')
if buildings['way/319101984']['height_m'] != '59.0' or '致原楼' not in buildings['way/319101984']['name']:
    raise ValueError('Zhiyuan Building mapped-part information is missing')
activity = mesh_vertices('campus-wall-activity.mesh')
if (buildings['way/231795526']['height_m'] != '10.2'
        or '汇昀楼' not in buildings['way/231795526']['name']
        or len(activity) < 30 or any(not 80 < v[0] < 105 or not 25 < v[2] < 72 for v in activity)):
    raise ValueError('Student activity center red wing or approximate three-storey information is missing')
qiaoyuan_dorms=json.loads((ASSETS/'campus-qiaoyuan-dorms.json').read_text(encoding='utf8'))
solar=[vertex for layer in qiaoyuan_dorms['layers'] if layer.get('role')=='solar'
       for vertex in mesh_vertices(layer['mesh'])]
if (buildings['way/629940893']['name'] != '乔森阁' or len(solar) < 12
        or any(v[1]-terrain_at(v[0],v[2]) < 25 for v in solar)):
    raise ValueError('Qiaoyuan dormitory name or rooftop panels are missing')
for filename, minimum in (('campus-basketball.mesh',3.0),('campus-tennis.mesh',8.0)):
    vertices = mesh_vertices(filename)
    if len(vertices) != 18 or any(v[1]-terrain_at(v[0],v[2]) < minimum for v in vertices):
        raise ValueError(f'{filename}: expected three mapped rooftop courts above the surrounding terrain')
boards = mesh_vertices('campus-basketball-boards.mesh')
netting = mesh_vertices('campus-sports-netting.mesh')
if (len(boards) != 36 or len(netting) < 90
        or any(v[1]-terrain_at(v[0],v[2]) < 5 for v in boards)):
    raise ValueError('Rooftop basketball boards or sports netting are missing or buried')
with (ASSETS/'sports-fence.rgba').open('rb') as f:
    w,h = struct.unpack('<II',f.read(8))
    pixels = f.read()
    if len(pixels) != w*h*4 or min(pixels[3::4]) != 0 or max(pixels[3::4]) == 0:
        raise ValueError('Sports net texture must contain both transparent holes and visible strands')

# 校方照片证实石头坞与泳池为不同层级；检查平面仍在地图位置、表面高于粗地形。
stone = mesh_vertices('campus-stone-plaza.mesh')
pool = mesh_vertices('campus-south-pool.mesh')
stone_y = sum(v[1] for v in stone)/len(stone)
pool_y = sum(v[1] for v in pool)/len(pool)
if (not 2.0 < stone_y-pool_y < 3.0
        or not all(90 < v[0] < 140 and 78 < v[2] < 158 for v in stone)
        or not all(145 < v[0] < 200 and 48 < v[2] < 100 for v in pool)
        or any(v[1]-terrain_at(v[0],v[2]) < .05 for v in stone+pool)):
    raise ValueError('Stone plaza / pool position, approximate height difference or terrain clearance is invalid')
canopy = mesh_vertices('campus-stone-canopies.mesh')
if len(canopy) < 100 or max(v[1] for v in canopy) < stone_y+4:
    raise ValueError('Stone plaza canopies are missing')
for filename in ('campus-stone-shell.mesh','campus-stone-flat.mesh','campus-stone-rails.mesh',
                 'campus-wall-fortress.mesh'):
    mesh_vertices(filename)
# 学活两翼独立基础墙要与地形及原墙脚连接，不能封住两翼之间的中庭。
activity_foundation=mesh_vertices('campus-activity-foundation.mesh')
activity_wall=mesh_vertices('campus-wall-activity.mesh')
top=max(v[1] for v in activity_foundation)
if len(activity_foundation)<100 or abs(top-min(v[1] for v in activity_wall)-.02)>.03:
    raise ValueError('The activity center foundation does not meet the building walls')
for v in activity_foundation:
    if v[1]<top-.05 and abs(v[1]-terrain_triangle_at(v[0],v[2])+.12)>.04:
        raise ValueError('The activity center foundation floats above the terrain')
if segment_hits_triangles((101.25,16.0,37.27),(132.26,16.0,25.98),activity_foundation):
    raise ValueError('The activity center foundation blocks its open atrium')
# 两处新增临坡基础的底边接地，且不穿入相邻的地图道路。
for group,way_id,distances in (('teachers','232598300',(145,160,175)),
                                ('northsports','231676386',(140,150,160,170))):
    foundation=mesh_vertices(f'campus-{group}-foundation.mesh')
    if len(foundation)<100:
        raise ValueError(f'{group} foundation is incomplete')
    for i in range(0,len(foundation),6):
        face=foundation[i:i+6]
        top=max(v[1] for v in face)
        for v in face:
            if v[1]<top-.05 and abs(v[1]-terrain_triangle_at(v[0],v[2])+.12)>.04:
                raise ValueError(f'{group} foundation is detached from terrain')
    for d in distances:
        a=point_along(mapped_points(way_id),d);b=point_along(mapped_points(way_id),d+4)
        start=(a[0],terrain_triangle_at(*a)+1.0,a[1])
        end=(b[0],terrain_triangle_at(*b)+1.0,b[1])
        if segment_hits_triangles(start,end,foundation):
            raise ValueError(f'{group} foundation blocks the neighboring road')

def inside_ring(x,z,ring):
    # 射线法只用来发现树干落入已知水面或广场，精确树木位置仍是程序化近似。
    inside=False
    for a,b in zip(ring,ring[1:]):
        if ((a[1]>z) != (b[1]>z)) and x < (b[0]-a[0])*(z-a[1])/(b[1]-a[1])+a[0]:
            inside=not inside
    return inside

# 致工楼入口有真实架空和逐级台阶；停车场内不应生成树干。
zhigong_shell=mesh_vertices('campus-zhigong-shell.mesh')
zhigong_glass=mesh_vertices('campus-zhigong-glass.mesh')
zhigong_steps=mesh_vertices('campus-zhigong-steps.mesh')
zhigong_rails=mesh_vertices('campus-zhigong-rails.mesh')
if (buildings['way/231086366']['height_m'] != '16.5'
        or len(zhigong_shell)<1000 or len(zhigong_glass)<100
        or len(zhigong_rails)<200 or len({round(v[1],2) for v in zhigong_steps})<18):
    raise ValueError('ZhiGong facade, entrance steps or approximate mapped height is missing')
entry_y=terrain_at(184,672)+5
if segment_hits_triangles((182,entry_y,672),(188,entry_y,672),zhigong_shell):
    raise ValueError('ZhiGong open portico is blocked by a wall')
if not 3.2 < max(v[1] for v in zhigong_steps)-terrain_at(175,680) < 3.6:
    raise ValueError('ZhiGong entrance stair height is not aligned with the estimated platform')
for filename in ('campus-zhigong-dark.mesh','campus-zhigong-perforated.mesh',
                 'campus-zhigong-louvers.mesh','campus-zhigong-sign.mesh',
                 'campus-zhigong-cars-light.mesh','campus-zhigong-cars-dark.mesh',
                 'campus-zhigong-cars-glass.mesh','campus-zhigong-cars-wheels.mesh',
                 'campus-zhigong-chargers.mesh'):
    mesh_vertices(filename)
lot_ring=mapped_points('976046850')
if any(inside_ring(x,z,lot_ring) for x,_,z,_ in trees):
    raise ValueError('A procedural tree occupies the ZhiGong parking area')
for filename in ('campus-zhigong-cars-light.mesh','campus-zhigong-cars-dark.mesh'):
    if any(not inside_ring(v[0],v[2],lot_ring) for v in mesh_vertices(filename)):
        raise ValueError(f'Parked car extends beyond the mapped lot: {filename}')
chargers=mesh_vertices('campus-zhigong-chargers.mesh')
if (any(not inside_ring(v[0],v[2],lot_ring) for v in chargers)
        or max(v[1] for v in chargers)-terrain_at(100,655.5)<1.1):
    raise ValueError('ZhiGong parking chargers are outside the lot or below ground')

# 立德门到时光广场的草坪在航拍中是开敞的，随机树及其树冠不应遮住这条轴线。
gate_lawn=mapped_points('1273530488')
for x,_,z,_ in trees:
    if inside_ring(x,z,gate_lawn):
        raise ValueError('A procedural tree occupies the mapped entrance lawn')
    for a,b in zip(gate_lawn,gate_lawn[1:]):
        dx,dz=b[0]-a[0],b[1]-a[1]
        length2=dx*dx+dz*dz
        if length2:
            t=max(0,min(1,((x-a[0])*dx+(z-a[1])*dz)/length2))
            if math.hypot(x-a[0]-t*dx,z-a[1]-t*dz)<4.99:
                raise ValueError('A procedural tree crown overlaps the entrance lawn edge')

# 教学楼组团中庭保留开敞铺装和草坪，不放入程序随机树。
south_courtyard_ring=mapped_points('724264876')
if any(inside_ring(x,z,south_courtyard_ring) for x,_,z,_ in trees):
    raise ValueError('A procedural tree occupies the mapped south teaching courtyard')

# 致知楼和相邻 L4 分属两块地图轮廓，七层折线立面与四层屋顶不可合成一栋。
zhizhi_frame=mesh_vertices('campus-zhizhi-frame.mesh')
zhizhi_glass=mesh_vertices('campus-zhizhi-glass.mesh')
l4_garden=mesh_vertices('campus-l4-garden.mesh')
l4_equipment=mesh_vertices('campus-l4-equipment.mesh')
if (buildings['way/984606893']['height_m'] != '23.1'
        or buildings['way/984606892']['height_m'] != '13.2'
        or len(zhizhi_frame)<3000 or len(zhizhi_glass)<30
        or max(v[0] for v in zhizhi_frame)-min(v[0] for v in zhizhi_frame)<100
        or max(v[1] for v in zhizhi_glass)-min(v[1] for v in zhizhi_glass)<22):
    raise ValueError('ZhiZhi facade or the mapped heights of the two separate buildings are missing')
for x in (194,212,230):
    garden_y=vertical_surface_hits(l4_garden,x,572)
    equipment_y=vertical_surface_hits(l4_equipment,x,572)
    if (not garden_y or not equipment_y
            or not 13.1 < max(garden_y)-terrain_at(x,572) < 13.4
            or not 1.7 < max(equipment_y)-max(garden_y) < 2.1):
        raise ValueError(f'L4 roof garden or rooftop equipment is missing near x={x}')
if any(183 < x < 249 and 585 < z < 599 for x,_,z,_ in trees):
    raise ValueError('A procedural tree occupies the open courtyard between ZhiZhi and L4')

# 汇萃楼北侧规划硬质广场不应被随机树木阻断。
forecourt=[(art_a[0]-sx*.17,art_a[1]-sz*.17),(art_b[0]-sx*.17,art_b[1]-sz*.17),
           art_b,art_a,(art_a[0]-sx*.17,art_a[1]-sz*.17)]
if any(inside_ring(x,z,forecourt) for x,_,z,_ in trees):
    raise ValueError('A procedural tree occupies the Huicui entrance forecourt')

# 汇文楼 H3 入口按照片保留实际可穿过的架空门洞，而非将红墙画成整块实心楼。
h3_way = next(w for w in osm.findall('way') if w.get('id') == '231406259')
_,b,c,d,e = [node_xy[n.get('ref')] for n in h3_way.findall('nd')[:5]]
axis = (d[0]-c[0],d[1]-c[1])
axis_length = math.hypot(*axis)
outward = (axis[1]/axis_length,-axis[0]/axis_length)
approach = [c,d,(d[0]+20*outward[0],d[1]+20*outward[1]),
            (c[0]+20*outward[0],c[1]+20*outward[1]),c]
if any(inside_ring(x,z,approach) for x,_,z,_ in trees):
    raise ValueError('A procedural tree blocks the Huiwen entrance approach')
steps = mesh_vertices('campus-huiwen-steps.mesh')
terrace = mesh_vertices('campus-huiwen-terrace.mesh')
red_wall = mesh_vertices('campus-huiwen-red.mesh')
if (len({round(v[1],2) for v in steps}) < 6
        or any(v[1]-terrain_at(v[0],v[2]) < -.15 for v in steps)
        or any(not (-269 < v[0] < -226 and -548 < v[2] < -520) for v in terrace)
        or min(v[1]-terrain_at(v[0],v[2]) for v in terrace) < 5
        or buildings['way/231406259']['height_m'] != '26.4'):
    raise ValueError('Huiwen steps, mapped H4 roof or OSM height is invalid')

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))

def hit_segment(origin,direction,triangle):
    # Möller–Trumbore：水平线段与三角面求交，用于确认门洞没有被实体墙封死。
    a,b,c = (v[:3] for v in triangle)
    edge1 = tuple(b[i]-a[i] for i in range(3))
    edge2 = tuple(c[i]-a[i] for i in range(3))
    p = cross(direction,edge2)
    det = dot(edge1,p)
    if abs(det) < 1e-8:
        return False
    offset = tuple(origin[i]-a[i] for i in range(3))
    u = dot(offset,p)/det
    q = cross(offset,edge1)
    v = dot(direction,q)/det
    distance = dot(edge2,q)/det
    return 0 <= u <= 1 and 0 <= v and u+v <= 1 and .001 < distance < .999

ground = min(v[1] for v in red_wall)+2
front = (d[0]+(c[0]-d[0])*.65,d[1]+(c[1]-d[1])*.65)
back = (e[0]+(b[0]-e[0])*.65,e[1]+(b[1]-e[1])*.65)
origin = (front[0]+outward[0],ground+3,front[1]+outward[1])
end = (back[0]-outward[0],ground+3,back[1]-outward[1])
direction = tuple(end[i]-origin[i] for i in range(3))
for filename in ('campus-wall-huiwen.mesh','campus-huiwen-detail.mesh','campus-huiwen-red.mesh'):
    vertices = mesh_vertices(filename)
    if any(hit_segment(origin,direction,vertices[i:i+3]) for i in range(0,len(vertices),3)):
        raise ValueError(f'Huiwen entrance is blocked by {filename}')

mapped_ways = {w.get('id'):w for w in osm.findall('way') if w.get('id') in ('231665580','230849087')}
for osm_id,way in mapped_ways.items():
    ring=[node_xy[n.get('ref')] for n in way.findall('nd')]
    if any(inside_ring(x,z,ring) for x,_,z,_ in trees):
        raise ValueError(f'Tree overlaps stone plaza or south pool: way/{osm_id}')

# 北夏馆两个地图泳池应有水平水面，不能被粗地形或随机树木穿过。
for osm_id,filename,sample in (('230849085','campus-beixia-practice.mesh',(155.5,-497.5)),
                               ('230849086','campus-beixia-lanes.mesh',(197.2,-497.9))):
    water=mesh_vertices(filename)
    levels=vertical_surface_hits(water,*sample)
    if not levels or min(levels)-terrain_triangle_at(*sample)<.05:
        raise ValueError(f'North swimming pool is missing or below terrain: way/{osm_id}')
    if any(v[1]-terrain_triangle_at(v[0],v[2])<.05 for v in water):
        raise ValueError(f'North swimming pool intersects the terrain: way/{osm_id}')
    way=next(w for w in osm.findall('way') if w.get('id')==osm_id)
    ring=[node_xy[n.get('ref')] for n in way.findall('nd')]
    if any(inside_ring(x,z,ring) for x,_,z,_ in trees):
        raise ValueError(f'Tree overlaps north swimming pool: way/{osm_id}')

# 门廊中央保留穿过外墙的入口；退后的玻璃门单独建模，不算实体墙堵塞。
gym=next(w for w in osm.findall('way') if w.get('id')=='230956722')
gym_ring=[node_xy[n.get('ref')] for n in gym.findall('nd')]
a,b=gym_ring[3],gym_ring[4]
dx,dz=b[0]-a[0],b[1]-a[1]
length=math.hypot(dx,dz)
out=(-dz/length,dx/length)
mid=((a[0]+b[0])/2,(a[1]+b[1])/2)
height=terrain_at(204.1,-373.8)+2
origin=(mid[0]+4*out[0],height,mid[1]+4*out[1])
end=(mid[0]-.5*out[0],height,mid[1]-.5*out[1])
direction=tuple(end[i]-origin[i] for i in range(3))
for filename in ('campus-yuanping-shell.mesh','campus-yuanping-base.mesh'):
    vertices=mesh_vertices(filename)
    if any(hit_segment(origin,direction,vertices[i:i+3]) for i in range(0,len(vertices),3)):
        raise ValueError(f'Yuanping entrance is blocked by {filename}')

# 高尔夫练习区应保持开阔、缓坡并有弧形击球位；外缘树带不做逐株实测声明。
golf_samples=((403,-649),(442,-617),(481,-586))
golf_levels=[terrain_at(*p) for p in golf_samples]
if max(golf_levels)-min(golf_levels)>2.0:
    raise ValueError('Golf driving range still has extreme DEM bumps')
golf=next(w for w in osm.findall('way') if w.get('id')=='231522459')
golf_ring=[node_xy[n.get('ref')] for n in golf.findall('nd')]
def golf_edge_distance(x,z):
    nearest=[]
    for a,b in zip(golf_ring[:-1],golf_ring[1:]):
        dx,dz=b[0]-a[0],b[1]-a[1]
        t=max(0,min(1,((x-a[0])*dx+(z-a[1])*dz)/(dx*dx+dz*dz)))
        nearest.append(math.hypot(x-a[0]-t*dx,z-a[1]-t*dz))
    return min(nearest)
if any(x>390 and z<-580 and inside_ring(x,z,golf_ring) and golf_edge_distance(x,z)>5.01
       for x,_,z,_ in trees):
    raise ValueError('Random tree remains in the golf driving area')
if len(mesh_vertices('campus-golf-mats.mesh'))<60 or len(mesh_vertices('campus-golf-signs.mesh'))<24:
    raise ValueError('Golf hitting mats or distance signs are missing')

# 校外深南大道应在地图车道位置可见，且不能被全球 DEM 的异常峰谷抬成陡坡。
shennan_samples=((300,-808),(400,-812),(500,-816),(600,-822),(700,-828),
                 (300,-776),(400,-781),(500,-786),(600,-791),(700,-797))
shennan_y=[terrain_triangle_at(*point) for point in shennan_samples]
if min(shennan_y)<22 or max(shennan_y)>30 or max(shennan_y)-min(shennan_y)>5:
    raise ValueError(f'Shennan Avenue is not a continuous estimated road surface: {shennan_y}')
with (ASSETS/'campus-ground.ppm').open('rb') as f:
    if f.readline().strip()!=b'P6':
        raise ValueError('Invalid ground texture header')
    width,height=map(int,f.readline().split())
    if f.readline().strip()!=b'255':
        raise ValueError('Unsupported ground texture depth')
    offset=f.tell()
    for x,z in ((500,-814.7),(500,-784.4)):
        px=round((x+800)/1520*(width-1));py=round((z+830)/1690*(height-1))
        f.seek(offset+(py*width+px)*3)
        rgb=f.read(3)
        if len(rgb)!=3 or max(rgb)-min(rgb)>18:
            raise ValueError(f'Shennan Avenue carriageway missing from texture at {(x,z)}: {rgb}')
    # 广场内部不能再出现底图道路的深色斑块，外侧道路仍应保留。
    for x,z,light in ((-270,-40,True),(-265,-65,True),(-285,-39,False)):
        px=round((x+800)/1520*(width-1));py=round((z+830)/1690*(height-1))
        f.seek(offset+(py*width+px)*3)
        rgb=f.read(3)
        if len(rgb)!=3 or (min(rgb)<170 if light else max(rgb)>155):
            raise ValueError(f'Time Square ground texture is inconsistent at {(x,z)}: {rgb}')

# 新增的 L 形远景只接在原地形之外；两条公共边上的每个网格点必须无高度裂缝。
city_ground=mesh_vertices('campus-north-city-ground.mesh')
north_edge={(round(v[0],2),v[1]) for v in city_ground if abs(v[2]+830)<.01 and v[0]<=720.01}
east_edge={(round(v[2],2),v[1]) for v in city_ground if abs(v[0]-720)<.01 and -830.01<=v[2]<=-499.99}
if (len(north_edge)<153 or len(east_edge)<34
        or any(abs(y-terrain_at(x,-830))>.01 for x,y in north_edge)
        or any(abs(y-terrain_at(720,z))>.01 for z,y in east_edge)
        or max(v[0] for v in city_ground)<1049 or min(v[2] for v in city_ground)>-1159):
    raise ValueError('North/east city background does not meet the campus terrain')
if buildings['way/600815118']['height_m']!='193.2':
    raise ValueError('Tencent building background height no longer matches its mapped value')
with (ASSETS/'campus-north-city-ground.ppm').open('rb') as f:
    if f.readline().strip()!=b'P6': raise ValueError('Invalid city-ground texture header')
    width,height=map(int,f.readline().split())
    if f.readline().strip()!=b'255': raise ValueError('Unsupported city-ground texture depth')
    offset=f.tell()
    for x,z in ((800,-832),(800,-801)):
        px=round((x+800)/1850*(width-1));py=round((-500-z)/660*(height-1))
        f.seek(offset+(py*width+px)*3)
        rgb=f.read(3)
        if len(rgb)!=3 or max(rgb)-min(rgb)>18:
            raise ValueError(f'Shennan Avenue is not continued into the city background at {(x,z)}: {rgb}')

dial = mesh_vertices('campus-sundial-face.mesh')
if (abs(sum(v[0] for v in dial)/len(dial)+243.2) > 1.0
        or abs(sum(v[2] for v in dial)/len(dial)+48.6) > 1.0
        or max(v[1] for v in dial)-min(v[1] for v in dial) < 3.0):
    raise ValueError('Sundial is missing its mapped location or tilted face')
# 田径场专用平面需要位于调平地形上方；球门和高杆照明也应存在于球场附近。
stadium_surface=mesh_vertices('campus-stadium-surface.mesh')
for x,z in ((424,-404),(390,-450),(456,-360),(424,-490)):
    hits=vertical_surface_hits(stadium_surface,x,z)
    if not hits or min(hits)-terrain_triangle_at(x,z)<.02 or max(hits)-terrain_triangle_at(x,z)>.3:
        raise ValueError('Stadium surface is missing or intersects the terrain')
stadium_goals=mesh_vertices('campus-stadium-goals.mesh')
if (max(v[1] for v in stadium_goals)-min(v[1] for v in stadium_goals)<2.0
        or min(v[2] for v in stadium_goals)>-450 or max(v[2] for v in stadium_goals)<-355):
    raise ValueError('Stadium football goals are missing or misplaced')
stadium_lamps=mesh_vertices('campus-stadium-lamps.mesh')
if len(stadium_lamps)<400 or max(v[1] for v in stadium_lamps)-terrain_at(420,-404)<27:
    raise ValueError('Stadium floodlight arrays are missing or too low')
# 采样实际生成的纹理，确认场心、草坪、跑道及外侧缓冲区没有被覆盖或错位。
stadium_way=next(w for w in osm.findall('way') if w.get('id')=='230848519')
stadium_coords=[node_xy[n.get('ref')] for n in stadium_way.findall('nd')]
stadium_bounds=(min(p[0] for p in stadium_coords),min(p[1] for p in stadium_coords),
                max(p[0] for p in stadium_coords),max(p[1] for p in stadium_coords))
with (ASSETS/'stadium-surface.ppm').open('rb') as f:
    if f.readline().strip()!=b'P6': raise ValueError('Invalid stadium texture format')
    sw,sh=map(int,f.readline().split())
    if f.readline().strip()!=b'255': raise ValueError('Unsupported stadium texture depth')
    stadium_pixels=f.read()
if len(stadium_pixels)!=sw*sh*3: raise ValueError('Truncated stadium texture')
def stadium_pixel(x,z):
    x0,z0,x1,z1=stadium_bounds
    u=round((x-x0)/(x1-x0)*(sw-1));v=round((z-z0)/(z1-z0)*(sh-1))
    return stadium_pixels[(v*sw+u)*3:(v*sw+u+1)*3]
spot,grass,track,apron=(stadium_pixel(*p) for p in ((423.77,-404.02),(424,-390),(378.5,-404),(360,-404)))
if not (min(spot)>200 and grass[1]>grass[0] and track[0]>2*track[1] and apron[1]>apron[0]):
    raise ValueError('Stadium markings, grass, track or apron is missing')
# 北门路应连续且横向近似平整，旁边不能重新出现粗高程造成的尖锐深沟。
for z in range(-740,-529,10):
    middle=terrain_at(286,z)
    if max(abs(terrain_at(x,z)-middle) for x in (278,294))>.4:
        raise ValueError(f'North-gate roadway has an implausible cross slope at z={z}')
    if z < -530 and abs(terrain_at(286,z+10)-middle)>1.3:
        raise ValueError(f'North-gate roadway is discontinuous at z={z}')
for z in range(-730,-659,10):
    heights=[terrain_at(x,z) for x in (300,310,320,330,340)]
    if max(abs(b-a) for a,b in zip(heights,heights[1:]))>1.5:
        raise ValueError(f'Terrain between the north gate and golf range has a sharp ditch at z={z}')
# 学活至石头坞的主路保留整体高差，但不能由粗网格造成突降或严重横坡。
slope=json.loads((DATA/'road-slope-calibration.json').read_text(encoding='utf-8'))['routes'][0]
if slope['osm_way']!='231664133' or slope['before_max_sample_grade']<.4:
    raise ValueError('The Student West Road terrain calibration is missing or unexpected')
centerline=mapped_points('231664133')
start,end=slope['start_m'],slope['end_m']
samples=[point_along(centerline,start+i*(end-start)/44) for i in range(45)]
levels=[terrain_triangle_at(x,z) for x,z in samples]
if (not 8.0<levels[0]-levels[-1]<13.0
        or max(abs(b-a)/((end-start)/44) for a,b in zip(levels,levels[1:]))>.20):
    raise ValueError('Student West Road has lost its slope or regained an abrupt drop')
for i in range(14,35):
    a,b=samples[i-1],samples[i+1]
    dx,dz=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dz)
    x,z=samples[i]
    left=terrain_triangle_at(x-3*dz/length,z+3*dx/length)
    right=terrain_triangle_at(x+3*dz/length,z-3*dx/length)
    if abs(left-right)>.90:
        raise ValueError(f'Student West Road has an abrupt transverse slope at {(x,z)}')
# 师范楼和北夏馆的坡路保留两端高差，坡中及横向不得出现此前的尖折。
slope_profiles={r['osm_way']:r for r in json.loads((DATA/'road-slope-calibration.json').read_text(encoding='utf-8'))['routes']}
for osm_id,limit in (('232598300',.27),('231676386',.23)):
    route=slope_profiles[osm_id];centerline=mapped_points(osm_id)
    start,end=route['start_m'],route['end_m']
    count=round((end-start)/2.5)
    points=[point_along(centerline,start+i*(end-start)/count) for i in range(count+1)]
    elevations=[terrain_triangle_at(*p) for p in points]
    if not 8<elevations[-1]-elevations[0]<18:
        raise ValueError(f'An estimated slope lost its overall height difference: way/{osm_id}')
    if max(abs(b-a)/((end-start)/count) for a,b in zip(elevations,elevations[1:]))>limit:
        raise ValueError(f'A road still has an abrupt height change: way/{osm_id}')
    for i in range(5,count-4):
        a,b=points[i-1],points[i+1]
        dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz);x,z=points[i]
        if abs(terrain_triangle_at(x-2*dz/length,z+2*dx/length)-
               terrain_triangle_at(x+2*dz/length,z-2*dx/length))>.70:
            raise ValueError(f'A road has an abrupt cross slope: way/{osm_id}')
# 立德门与广场之间两条路同时拟合，避免一侧修坡后另一侧出现新的断层。
time_profile=slope_profiles['220071403']
for osm_id,route,limit in (('220071403',time_profile,.23),
                            ('220071398',time_profile['additional_roads'][0],.27)):
    start,end=route['start_m'],route['end_m'];count=round((end-start)/2.5)
    points=[point_along(mapped_points(osm_id),start+i*(end-start)/count) for i in range(count+1)]
    levels=[terrain_triangle_at(*p) for p in points]
    if (not 5<abs(levels[-1]-levels[0])<9
            or max(abs(b-a)/((end-start)/count) for a,b in zip(levels,levels[1:]))>limit):
        raise ValueError(f'The road approaching Time Square has an abrupt slope: way/{osm_id}')

# 汇德楼B座外围弯路保留模型中的整体下降，弯中不能出现短距离突折。
huide_curve=mapped_points('232598300')
huide_levels=[terrain_triangle_at(*point_along(huide_curve,d)) for d in range(15,121)]
if (not 12<huide_levels[0]-huide_levels[-1]<15.5 or
        max(abs(b-a) for a,b in zip(huide_levels,huide_levels[1:]))>.24):
    raise ValueError('The Hui De outer bend lost its relief or has an abrupt grade')
for d in range(20,117,3):
    x,z=point_along(huide_curve,d)
    a,b=point_along(huide_curve,d-1),point_along(huide_curve,d+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    if abs(terrain_triangle_at(x-2*dz/length,z+2*dx/length)-
           terrain_triangle_at(x+2*dz/length,z-2*dx/length))>.35:
        raise ValueError(f'The Hui De outer bend has an abrupt cross slope at {d} m')
# 新弯道和广场共用一块细分地形，不能牵动已相接的主路和北侧连接路。
for osm_id,distances in (('231407864',range(575,701,3)),('231983448',range(0,320,3))):
    centerline=mapped_points(osm_id)
    for d in distances:
        x,z=point_along(centerline,d)
        if not (-460<=x<=-250 and -230<=z<=-10):
            continue
        a,b=point_along(centerline,max(0,d-1)),point_along(centerline,d+1)
        dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
        for offset in (-2.5,0,2.5):
            px,pz=x-offset*dz/length,z+offset*dx/length
            if abs(terrain_triangle_at(px,pz)-coarse_terrain_triangle_at(px,pz))>.03:
                raise ValueError(f'A road adjoining the Hui De bend changed height: way/{osm_id} {d} m')

# 秋瑟旁的道路保留原有整体高差，同时消除短距离陡折。
alignments=json.loads((DATA/'road-alignment-profile.json').read_text(encoding='utf-8'))['adjustments']
south_dorm_road=next(item['centerline_m'] for item in alignments if item['osm_way']=='722200013')
mapped_south_dorm=mapped_points('722200013')
if (math.dist(south_dorm_road[0],mapped_south_dorm[0])>.02 or
        math.dist(south_dorm_road[-1],mapped_south_dorm[-1])>.02):
    raise ValueError('Adjusted south dormitory road does not reconnect to the mapped network')
route=slope_profiles['722200013'];start,end=route['start_m'],route['end_m']
count=round((end-start)/2.5)
points=[point_along(south_dorm_road,start+i*(end-start)/count) for i in range(count+1)]
levels=[terrain_triangle_at(*p) for p in points]
if not 4<levels[-1]-levels[0]<7 or max(abs(b-a)/((end-start)/count) for a,b in zip(levels,levels[1:]))>.19:
    raise ValueError('The south dormitory road has an abrupt slope or lost its height difference')

# 风槐斋至深大东路保留整体下坡，同时控制路口前的短距离突坡和横坡。
wind_road=mapped_points('232096326')
wind_levels=[terrain_triangle_at(*point_along(wind_road,d)) for d in range(60,118)]
if (not 5.0<wind_levels[0]-wind_levels[-1]<8.0 or
        max(abs(b-a) for a,b in zip(wind_levels,wind_levels[1:]))>.23):
    raise ValueError('The Wind Huai dormitory road has an abrupt height change')
for d in range(70,111,5):
    x,z=point_along(wind_road,d)
    a,b=point_along(wind_road,d-1),point_along(wind_road,d+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    if abs(terrain_triangle_at(x-2*dz/length,z+2*dx/length)-
           terrain_triangle_at(x+2*dz/length,z-2*dx/length))>.6:
        raise ValueError(f'The Wind Huai dormitory road has an abrupt cross slope at {d} m')
# 相接的主环路保留原高度，细分地形不能改动车道或路口横向位置。
east_ring=mapped_points('488453361')
for d in range(330,401,5):
    x,z=point_along(east_ring,d)
    a,b=point_along(east_ring,d-1),point_along(east_ring,d+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    for offset in (-2.5,0,2.5):
        px,pz=x-offset*dz/length,z+offset*dx/length
        if abs(terrain_triangle_at(px,pz)-coarse_terrain_triangle_at(px,pz))>.03:
            raise ValueError(f'The Wind Huai junction changed the east ring road at {d} m')

# 田径场东侧的两条道路联合调整，沿途和横向不能再出现粗 DEM 造成的尖峰。
stadium_roads=(('239834471',225,285,.25,.70),('239872140',130,205,.20,.60))
for osm_id,start,end,grade_limit,cross_limit in stadium_roads:
    centerline=mapped_points(osm_id)
    levels=[terrain_triangle_at(*point_along(centerline,d)) for d in range(start,end+1)]
    if max(abs(b-a) for a,b in zip(levels,levels[1:]))>grade_limit:
        raise ValueError(f'The stadium-side road has an abrupt grade: way/{osm_id}')
    for d in range(start+4,end-3,3):
        x,z=point_along(centerline,d)
        a,b=point_along(centerline,d-1),point_along(centerline,d+1)
        dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
        if abs(terrain_triangle_at(x-2*dz/length,z+2*dx/length)-
               terrain_triangle_at(x+2*dz/length,z-2*dx/length))>cross_limit:
            raise ValueError(f'The stadium-side road has an abrupt cross slope: way/{osm_id}')
for x,z in ((480,-490),(460,-500),(549.5,-445.5)):
    if abs(terrain_triangle_at(x,z)-coarse_terrain_triangle_at(x,z))>.03:
        raise ValueError(f'The stadium or Hui Yan building core moved at {(x,z)}')

# 汇文楼支路有显著整体高差；仅消除突折并保留相接回车区域的原有地形。
huiwen_road=mapped_points('801959630')
levels=[terrain_triangle_at(*point_along(huiwen_road,d)) for d in range(1,32)]
if not 6.0<levels[-1]-levels[0]<8.5 or max(abs(b-a) for a,b in zip(levels,levels[1:]))>.32:
    raise ValueError('The Hui Wen side road lost its slope or has an abrupt grade')
for d in range(4,29,3):
    x,z=point_along(huiwen_road,d)
    a,b=point_along(huiwen_road,d-1),point_along(huiwen_road,d+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    if abs(terrain_triangle_at(x-2*dz/length,z+2*dx/length)-
           terrain_triangle_at(x+2*dz/length,z-2*dx/length))>.65:
        raise ValueError(f'The Hui Wen side road has an abrupt cross slope at {d} m')
for d in range(0,41,3):
    x,z=point_along(mapped_points('319099130'),d)
    if abs(terrain_triangle_at(x,z)-coarse_terrain_triangle_at(x,z))>.03:
        raise ValueError(f'The Hui Wen turnaround changed height at {d} m')

# 教育学部北侧道路接到两条既有道路，邻接线位与高程需一起保留。
faculty_road=mapped_points('237892845')
levels=[terrain_triangle_at(*point_along(faculty_road,d)) for d in range(310,375)]
if max(abs(b-a) for a,b in zip(levels,levels[1:]))>.24 or max(levels)-min(levels)<2:
    raise ValueError('The north faculty road has an abrupt grade or lost its terrain relief')
for d in range(315,370,3):
    x,z=point_along(faculty_road,d)
    a,b=point_along(faculty_road,d-1),point_along(faculty_road,d+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    if abs(terrain_triangle_at(x-2*dz/length,z+2*dx/length)-
           terrain_triangle_at(x+2*dz/length,z-2*dx/length))>.75:
        raise ValueError(f'The north faculty road has an abrupt cross slope at {d} m')
for osm_id,distances in (('231676802',range(108,137,3)),('231983448',range(0,25,3))):
    path=mapped_points(osm_id)
    for d in distances:
        x,z=point_along(path,d)
        if abs(terrain_triangle_at(x,z)-coarse_terrain_triangle_at(x,z))>.03:
            raise ValueError(f'A connecting north faculty road moved: way/{osm_id} {d} m')

# 体育馆后侧环路经过泳池与场馆之间；保留整体起伏，抑制短距离尖峰和横坡。
gym_road=mapped_points('231676381')
gym_levels=[terrain_triangle_at(*point_along(gym_road,d)) for d in range(69,186)]
if max(abs(b-a) for a,b in zip(gym_levels,gym_levels[1:]))>.25 or max(gym_levels)-min(gym_levels)<4:
    raise ValueError('The gymnasium loop has an abrupt grade or lost its terrain relief')
for d in range(72,182,3):
    x,z=point_along(gym_road,d)
    a,b=point_along(gym_road,d-1),point_along(gym_road,d+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    if abs(terrain_triangle_at(x-2*dz/length,z+2*dx/length)-
           terrain_triangle_at(x+2*dz/length,z-2*dx/length))>.55:
        raise ValueError(f'The gymnasium loop has an abrupt cross slope at {d} m')
for x,z in ((190,-400),(200,-390),(210,-380),(210,-455)):
    if abs(terrain_triangle_at(x,z)-coarse_terrain_triangle_at(x,z))>.03:
        raise ValueError(f'The gymnasium or swimming facility core moved at {(x,z)}')

# 细分补丁边界的每个顶点与粗地形吻合，避免产生肉眼可见的裂缝。
patches=terrain_patches()
if len(patches)!=8:
    raise ValueError('Expected all eight campus road terrain patches')
for pnx,pnz,px0,px1,pz0,pz1,pvalues in patches:
    for j in range(pnz):
        for i in range(pnx):
            if i not in (0,pnx-1) and j not in (0,pnz-1):
                continue
            x=px0+i*(px1-px0)/(pnx-1);z=pz0+j*(pz1-pz0)/(pnz-1)
            if abs(pvalues[j*pnx+i]-coarse_terrain_triangle_at(x,z))>.001:
                raise ValueError(f'Terrain patch has a boundary seam at {(x,z)}')
# 致远门分缝铺装覆盖入口，地形不能穿过它；大草坪中心不应生成随机大树。
angle=math.radians(42)
center_y=terrain_at(192.9,394.8)
for u in (-16,0,16):
    for v in (-9,0,9):
        x=192.9+u*math.cos(angle)+v*math.sin(angle)
        z=394.8-u*math.sin(angle)+v*math.cos(angle)
        if abs(terrain_triangle_at(x,z)-center_y)>.07:
            raise ValueError(f'ZhiYuan gate paving intersects uneven terrain at {(x,z)}')
with (ASSETS/'campus-trees.bin').open('rb') as f:
    count,=struct.unpack('<I',f.read(4))
    trees=list(struct.iter_unpack('<4f',f.read()))
if len(trees)!=count:
    raise ValueError('Tree instance data is truncated')
for x,z in ((220,445),(250,478)):
    if any(math.hypot(t[0]-x,t[2]-z)<8 for t in trees):
        raise ValueError(f'South-gate open lawn is obstructed near {(x,z)}')
for filename in ('campus-water-bank.mesh','campus-bridge-roof.mesh','campus-stadium-canopy.mesh',
                 'campus-wall-teng.mesh','campus-archive-seal.mesh'):
    mesh_vertices(filename)

# 路面、标线和山路必须与实际三角地形吻合，同时保留杜鹃山的整体高差。
for filename,offset in (('campus-road-surface.mesh',.035),('campus-road-markings.mesh',.065),
                        ('campus-dujuan-trail.mesh',.055),('campus-dujuan-boardwalk.mesh',.055)):
    vertices=mesh_vertices(filename)
    for v in vertices[::max(1,len(vertices)//180)]:
        if abs(v[1]-terrain_triangle_at(v[0],v[2])-offset)>.015:
            raise ValueError(f'{filename} floats above or sinks into the slope')
    if filename=='campus-dujuan-trail.mesh' and max(v[1] for v in vertices)-min(v[1] for v in vertices)<5:
        raise ValueError('Dujuan hill trails no longer retain the modeled terrain elevation')

# 三段主要环校道路在 OSM 共节点相接；检查两侧路面与路缘开口。
road_surface=mesh_vertices('campus-road-surface.mesh')
road_curbs=mesh_vertices('campus-road-curbs.mesh')
# 两条接入道路与广场独立砖面连续，草坪和汇元楼门前不被重复铺装覆盖。
time_paving=mesh_vertices('campus-time-paving.mesh')
for z in (-39,-65.5):
    for x in (-292,-286,-280,-276,-272,-268):
        hits=vertical_surface_hits(road_surface,x,z)+vertical_surface_hits(time_paving,x,z)
        ground=terrain_triangle_at(x,z)
        if not hits or min(abs(y-ground-.035) for y in hits)>.04:
            raise ValueError(f'Time Square road approach has a pavement gap at {(x,z)}')
for x,z in ((-243.2,-48.6),(-224,-23)):
    if vertical_surface_hits(time_paving,x,z):
        raise ValueError(f'Time Square paving covers lawn or the separate building entry at {(x,z)}')
# 地形补丁、道路网格与贴地高度必须使用同一三角面插值。
terrain_mesh=mesh_vertices('campus-terrain.mesh')
for x,z in ((141.8,-524.1),(125.5,-536.4),(80.0,-540.0),(220.0,-520.0),
            (-302,-39.1),(-285,-65.5),(-390,-45),(-250,-60),
            (523.6,239.4),(535.4,233.9),(480,245),(590,245),
            (260,-43.3),(275,-42.4),(150,-70),(320,-70),
            (501,-506),(480,-490),(450,-500),(570,-500),
            (-264,-609),(-320,-610),(-220,-610),(-287.9,-320.6),(-360,-330),(-220,-330),
            (195.1,-426.6),(100,-420),(280,-420),(170,-460),(170,-340),
            (-379,-94),(-400,-90),(-460,-150),(-350,-230)):
    hits=vertical_surface_hits(terrain_mesh,x,z)
    if not hits or min(abs(y-terrain_triangle_at(x,z)) for y in hits)>.02:
        raise ValueError(f'Terrain detail mesh and walking heights disagree at {(x,z)}')
for distance in (130,150,170):
    x,z=point_along(mapped_points('231676386'),distance)
    hits=vertical_surface_hits(road_surface,x,z)
    if not hits or min(abs(y-terrain_triangle_at(x,z)-.035) for y in hits)>.02:
        raise ValueError(f'The north-sports road is detached from terrain at {(x,z)}')
for distance in (10,20,30,40,50,60):
    x,z=point_along(south_dorm_road,distance)
    hits=vertical_surface_hits(road_surface,x,z)
    if not hits or min(abs(y-terrain_triangle_at(x,z)-.035) for y in hits)>.02:
        raise ValueError(f'The south dormitory road is detached from terrain at {(x,z)}')
if len(mesh_vertices('campus-southdorm-foundation.mesh'))<100:
    raise ValueError('The south dormitory foundation is missing after adjusting the road')
if len(mesh_vertices('campus-windzhai-foundation.mesh'))<100:
    raise ValueError('The Wind Huai dormitory foundations are missing after adjusting the road')
for filename in ('campus-huiwenroad-foundation.mesh','campus-facultyroad-foundation.mesh'):
    if len(mesh_vertices(filename))<18:
        raise ValueError(f'The roadside building foundation is missing: {filename}')
for distance in (70,80,90,100,110,119):
    x,z=point_along(wind_road,distance)
    hits=vertical_surface_hits(road_surface,x,z)
    if not hits or min(abs(y-terrain_triangle_at(x,z)-.035) for y in hits)>.02:
        raise ValueError(f'The Wind Huai road is detached from terrain at {(x,z)}')
for osm_id,distances in (('239834471',(230,245,260,275)),('239872140',(135,155,175,195))):
    centerline=mapped_points(osm_id)
    for distance in distances:
        x,z=point_along(centerline,distance)
        hits=vertical_surface_hits(road_surface,x,z)
        if not hits or min(abs(y-terrain_triangle_at(x,z)-.035) for y in hits)>.02:
            raise ValueError(f'The stadium-side road is detached from terrain at {(x,z)}')
for osm_id,distances in (('801959630',(5,10,15,20,25,30)),('237892845',(310,325,340,355,370))):
    centerline=mapped_points(osm_id)
    for distance in distances:
        x,z=point_along(centerline,distance)
        hits=vertical_surface_hits(road_surface,x,z)
        if not hits or min(abs(y-terrain_triangle_at(x,z)-.035) for y in hits)>.02:
            raise ValueError(f'The north campus road is detached from terrain at {(x,z)}')
for distance in (75,95,115,135,155,175):
    x,z=point_along(gym_road,distance)
    hits=vertical_surface_hits(road_surface,x,z)
    if not hits or min(abs(y-terrain_triangle_at(x,z)-.035) for y in hits)>.02:
        raise ValueError(f'The gymnasium loop is detached from terrain at {(x,z)}')
teachers_foundation=mesh_vertices('campus-teachers-foundation.mesh')
for distance in range(20,120,4):
    p,q=point_along(huide_curve,distance),point_along(huide_curve,distance+2)
    hits=vertical_surface_hits(road_surface,*p)
    if not hits or min(abs(y-terrain_triangle_at(*p)-.035) for y in hits)>.02:
        raise ValueError(f'The Hui De bend road is detached from terrain at {distance} m')
    dx,dz=q[0]-p[0],q[1]-p[1];length=math.hypot(dx,dz)
    for offset in (-2,0,2):
        nx,nz=-offset*dz/length,offset*dx/length
        a=(p[0]+nx,terrain_triangle_at(p[0]+nx,p[1]+nz)+1.3,p[1]+nz)
        b=(q[0]+nx,terrain_triangle_at(q[0]+nx,q[1]+nz)+1.3,q[1]+nz)
        if segment_hits_triangles(a,b,teachers_foundation):
            raise ValueError(f'A Hui De foundation blocks the bend at {distance} m')
gym_foundation=mesh_vertices('campus-yuanping-foundation.mesh')
if len(gym_foundation)<18:
    raise ValueError('The gymnasium foundation is missing')
for distance in range(70,180,4):
    p,q=point_along(gym_road,distance),point_along(gym_road,distance+2)
    dx,dz=q[0]-p[0],q[1]-p[1];length=math.hypot(dx,dz)
    for offset in (-2,0,2):
        nx,nz=-offset*dz/length,offset*dx/length
        start=(p[0]+nx,terrain_triangle_at(p[0]+nx,p[1]+nz)+1.3,p[1]+nz)
        end=(q[0]+nx,terrain_triangle_at(q[0]+nx,q[1]+nz)+1.3,q[1]+nz)
        if segment_hits_triangles(start,end,gym_foundation):
            raise ValueError(f'The gymnasium foundation blocks its loop road at {distance} m')
for path_id,start_d,end_d in (('801959630',0,7),('237892845',367,374)):
    centerline=mapped_points(path_id)
    p,q=point_along(centerline,start_d),point_along(centerline,end_d)
    a=(p[0],terrain_triangle_at(*p)+.1,p[1]);b=(q[0],terrain_triangle_at(*q)+.1,q[1])
    if segment_hits_triangles(a,b,road_curbs):
        raise ValueError(f'The mapped north campus road is blocked by a curb: way/{path_id}')
wind_junction=node_xy['2404417658']
wind_approach=point_along(wind_road,114.0)
start=(wind_approach[0],terrain_triangle_at(*wind_approach)+.1,wind_approach[1])
end=(wind_junction[0],terrain_triangle_at(*wind_junction)+.1,wind_junction[1])
if segment_hits_triangles(start,end,road_curbs):
    raise ValueError('The Wind Huai road is blocked by a curb at the east-ring junction')
# 车道及其两侧在通行高度均不能再被秋瑟的通用墙体或新基础封住。
south_dorm_walls=mesh_vertices('campus-wall-perforated.mesh')
south_dorms=json.loads((ASSETS/'campus-south-dorms.json').read_text(encoding='utf8'))
south_dorm_walls += [vertex for layer in south_dorms['layers'] if layer['id']=='qiuse'
                     for vertex in mesh_vertices(layer['mesh'])]
south_dorm_foundation=mesh_vertices('campus-southdorm-foundation.mesh')
for distance in range(20,49,2):
    p=point_along(south_dorm_road,distance);q=point_along(south_dorm_road,distance+2)
    dx,dz=q[0]-p[0],q[1]-p[1];length=math.hypot(dx,dz)
    for offset in (-2,0,2):
        nx,nz=-dz/length*offset,dx/length*offset
        a=(p[0]+nx,terrain_triangle_at(p[0]+nx,p[1]+nz)+1.3,p[1]+nz)
        b=(q[0]+nx,terrain_triangle_at(q[0]+nx,q[1]+nz)+1.3,q[1]+nz)
        if any(segment_hits_triangles(a,b,mesh) for mesh in (south_dorm_walls,south_dorm_foundation)):
            raise ValueError(f'The south dormitory road is blocked by a building near {distance} m')
ring=('231407864','1434756580','488453361')
for ref in ('2400181200','2310684965','2398019406'):
    center=node_xy[ref]
    connected=0
    for osm_id in ring:
        refs=[n.get('ref') for n in osm_ways[osm_id].findall('nd')]
        if ref not in refs:
            continue
        i=refs.index(ref)
        for side in (-1,1):
            if not 0<=i+side<len(refs):
                continue
            other=node_xy[refs[i+side]]
            length=math.dist(center,other)
            point=tuple(center[j]+(other[j]-center[j])*min(3/length,1) for j in range(2))
            if not vertical_surface_hits(road_surface,*point):
                raise ValueError(f'Ring road pavement is missing near node {ref}')
            start=(center[0],terrain_triangle_at(*center)+.1,center[1])
            end=(point[0],terrain_triangle_at(*point)+.1,point[1])
            if segment_hits_triangles(start,end,road_curbs):
                raise ValueError(f'Ring road is blocked by a curb at node {ref}')
            connected+=1
    if connected<2:
        raise ValueError(f'Mapped ring-road node {ref} is not connected')

# 两座桥的新增接线不能跨层跳变或被致艺楼墙体封住。
ramp=mapped_points('611264728')
ramp_length=sum(math.dist(a,b) for a,b in zip(ramp,ramp[1:]))
ramp_levels=[]
for i in range(13):
    x,z=point_along(ramp,ramp_length*i/12)
    hits=vertical_surface_hits(deck,x,z)
    ground=terrain_triangle_at(x,z)
    if not hits:
        raise ValueError('ZhouNan ground ramp has a missing deck')
    ramp_levels.append(min(hits,key=lambda y:abs(y-ground)))
if (not vertical_surface_hits(road_surface,*ramp[0]) or
        abs(ramp_levels[0]-terrain_triangle_at(*ramp[0]))>.10 or
        any(abs(a-b)>.2 for a,b in zip(ramp_levels,ramp_levels[1:]))):
    raise ValueError('ZhouNan ground ramp is not continuous with the campus road')
if profiles[1].get('south_ground_ramp_osm')!='611264728':
    raise ValueError('ZhouNan ground connection is missing from bridge metadata')

passage=mapped_points('776358640')
passage_floor=vertical_surface_hits(deck,*passage[0])[0]
for a,b in zip(passage,passage[1:]):
    for offset in (.9,1.7):
        start,end=(a[0],passage_floor+offset,a[1]),(b[0],passage_floor+offset,b[1])
        if any(segment_hits_triangles(start,end,mesh) for mesh in
               (zhiyi_walls,zhiyi_glass,mesh_vertices('campus-zhiyi-podium.mesh'),lower_rails)):
            raise ValueError('Zhaonan building passage is obstructed')
    middle=((a[0]+b[0])/2,(a[1]+b[1])/2)
    if not any(abs(y-passage_floor)<.06 for y in vertical_surface_hits(deck,*middle)):
        raise ValueError('Zhaonan building passage has a missing floor')
incline=mapped_points('776358641')
incline_length=sum(math.dist(a,b) for a,b in zip(incline,incline[1:]))
levels=[]
for i in range(9):
    x,z=point_along(incline,incline_length*i/8)
    hits=vertical_surface_hits(deck,x,z)
    if not hits:
        raise ValueError('Zhaonan passage to south-district walkway has a gap')
    levels.append(min(hits,key=lambda y:abs(y-(levels[-1] if levels else passage_floor))))
if abs(levels[0]-passage_floor)>.06 or any(abs(a-b)>.4 for a,b in zip(levels,levels[1:])):
    raise ValueError('Zhaonan passage to south-district walkway has an abrupt height change')

# 乔苑三栋楼采用历史图纸的设计基准；这里检查生成结果，不将其当作现状实测。
for x,z,reference in ((-338.258,594.204,15.595),(-275.076,567.042,13.54),(-214.934,539.039,11.49)):
    if abs(terrain_at(x,z)-reference)>.15:
        raise ValueError('Qiaoyuan terrain no longer follows the documented historical design reference')

# 用户确认阁区街道有坡。逐段检查道路仍然有高差、宽度和连续表面。
street_heights=[terrain_triangle_at(-426,z) for z in range(520,641,10)]
if (street_heights[0]-street_heights[-1]<4 or
        any(not -.85<b-a<-.2 for a,b in zip(street_heights,street_heights[1:]))):
    raise ValueError('Xiyuan shop street has lost its modeled slope or has a sudden height jump')
for z in (525,545,565,585,605,625):
    if not vertical_surface_hits(road_surface,-426,z):
        raise ValueError(f'Xiyuan shop street has a missing road surface at z={z}')
    if max(abs(terrain_triangle_at(x,z)-terrain_triangle_at(-426,z)) for x in (-430,-422))>.5:
        raise ValueError(f'Xiyuan shop street has an excessive modeled cross-slope at z={z}')
    start=(-426,terrain_triangle_at(-426,z)+.4,z)
    end=(-426,terrain_triangle_at(-426,z+8)+.4,z+8)
    if segment_hits_triangles(start,end,road_curbs):
        raise ValueError(f'Xiyuan shop street is blocked by a curb at z={z}')

# 店招网格的先后和相对位置应与用户提供的店铺顺序一致。
shops=json.loads((DATA/'qiaoyuan-shops.json').read_text(encoding='utf-8'))
row=shops['road_side']
expected=('manner','youjian','fruit','seven','tea','repair')
if tuple(item['id'] for item in row if item['id']!='print')!=expected:
    raise ValueError('Xiyuan shop order no longer matches the user-provided sequence')
signs=mesh_vertices('campus-xiyuan-shops-signs.mesh')
if len(signs)!=12*6:
    raise ValueError('Xiyuan shop sign mesh is incomplete')
for i,item in enumerate(row):
    section=signs[i*6:(i+1)*6]
    z=sum(v[2] for v in section)/len(section)
    if abs(z-(item['start_z']+item['end_z'])/2)>.1:
        raise ValueError(f"Xiyuan shop sign is misplaced: {item['id']}")
    # 店面中部入口应跟随坡地，不能形成跨整家店的高台阶。
    glass=mesh_vertices('campus-xiyuan-shops-glass.mesh')[i*6:(i+1)*6]
    center_x=sum(v[0] for v in glass)/len(glass)
    if abs(min(v[1] for v in glass)-terrain_triangle_at(center_x,z))>.22:
        raise ValueError(f"Xiyuan storefront entrance is detached from the sidewalk: {item['id']}")
repair=row[-1]
sf=shops['across_from_repair']
if min(repair['end_z'],sf['end_z'])-max(repair['start_z'],sf['start_z'])<4:
    raise ValueError('SF station is no longer opposite the repair shop')
sf_sign=signs[7*6:8*6]
if not all(-421<v[0]<-417 for v in sf_sign):
    raise ValueError('SF station is not on the opposite side of the street')
opposite=[item['id'] for item in shops['opposite_qiaoxiang']]
if not any(opposite[i:i+3]==['glasses','szu','barber'] for i in range(len(opposite)-2)):
    raise ValueError('The three neighboring Qiaoxiang storefronts no longer follow the 2026 photos')

# 7-11 和果果切外的膜棚需覆盖人行道，支柱不能穿过中央车道。
membrane=mesh_vertices('campus-xiyuan-shops-membrane.mesh')
awning_frame=mesh_vertices('campus-xiyuan-shops-awning-frame.mesh')
for z in (535,543,551,559,567,573):
    hits=vertical_surface_hits(membrane,-430.5,z)
    if not hits or not 3.1<min(hits)-terrain_triangle_at(-430.5,z)<4.4:
        raise ValueError(f'Xiyuan canopy is missing or too low above the sidewalk at z={z}')
    start=(-426,terrain_triangle_at(-426,z)+1.5,z)
    end=(-426,terrain_triangle_at(-426,z+2)+1.5,z+2)
    if segment_hits_triangles(start,end,awning_frame):
        raise ValueError(f'Xiyuan canopy supports obstruct the roadway at z={z}')

# 乔相阁花园的人字形园路及商铺铺装应与地形保持一致。
garden=mesh_vertices('campus-qiaoyuan-garden-walk.mesh')
for x,z in ((-288.904,524.391),(-280.235,526.940),(-275.195,526.646),
            (-268.919,527.431),(-274.332,520.763),(-273.724,534.001)):
    hits=vertical_surface_hits(garden,x,z)
    if not hits or min(abs(y-terrain_triangle_at(x,z)-.083) for y in hits)>.015:
        raise ValueError(f'Qiaoxiang garden path is missing or floating at {(x,z)}')
for filename in ('campus-qiaoyuan-garden-frame.mesh','campus-xiyuan-shops-bicycle.mesh',
                 'campus-xiyuan-shops-glass.mesh','campus-xiyuan-shops-paving.mesh',
                 'campus-xiyuan-shops-opposite-paving.mesh','campus-xiyuan-shops-plant-leaves.mesh'):
    mesh_vertices(filename)

# 全校道路的人行层为独立资源，按渲染地形铺设，覆盖南北校区的多条主支路。
sidewalk=mesh_vertices('campus-network-sidewalk.mesh')
mapped_walk=mesh_vertices('campus-network-paths.mesh')
corridor_walk=mesh_vertices('campus-lide-corridor-paving.mesh')
south_spine=mesh_vertices('campus-south-spine-paving.mesh')
roadside_shrubs=mesh_vertices('campus-network-shrubs.mesh')
# 人行网格分桶供整校检查共用，避免对每一个采样点扫描全部三角面。
walk_bins=collections.defaultdict(list)
cell=12.0
for mesh in (sidewalk,mapped_walk,corridor_walk,south_spine):
    for i in range(0,len(mesh),3):
        tri=mesh[i:i+3]
        xa,xb=min(v[0] for v in tri),max(v[0] for v in tri)
        za,zb=min(v[2] for v in tri),max(v[2] for v in tri)
        for gx in range(math.floor(xa/cell),math.floor(xb/cell)+1):
            for gz in range(math.floor(za/cell),math.floor(zb/cell)+1):
                walk_bins[(gx,gz)].append(tri)
def in_walkable_surface(x,z):
    for a,b,c in walk_bins[(math.floor(x/cell),math.floor(z/cell))]:
        denominator=(b[2]-c[2])*(a[0]-c[0])+(c[0]-b[0])*(a[2]-c[2])
        if abs(denominator)<1e-7:
            continue
        u=((b[2]-c[2])*(x-c[0])+(c[0]-b[0])*(z-c[2]))/denominator
        w=((c[2]-a[2])*(x-c[0])+(a[0]-c[0])*(z-c[2]))/denominator
        if min(u,w,1-u-w)>-1e-5:
            return True
    return False
for name,mesh in (('sidewalk',sidewalk),('mapped walkways',mapped_walk)):
    if len(mesh)<10000:
        raise ValueError(f'Campus {name} layer is unexpectedly incomplete')
    for v in mesh[::max(1,len(mesh)//180)]:
        if abs(v[1]-terrain_triangle_at(v[0],v[2])-.055)>.015:
            raise ValueError(f'Campus {name} floats above or sinks into the terrain')
for x,z in ((326.7,383.3),(331,380),(337.4,374.1),(344.3,380.5),(351,413.3),(355.9,410.75)):
    height=terrain_triangle_at(x,z)+.06
    if not any(abs(y-height)<.004 for y in vertical_surface_hits(south_spine,x,z)):
        raise ValueError(f'The South Spine lower path is missing or detached at {(x,z)}')
    if z>405 and not any(y-height>3.7 for y in vertical_surface_hits(deck,x,z)):
        raise ValueError(f'The South Spine upper and lower walks have merged at {(x,z)}')
for x,z in ((-115.9,-241.3),(-112.9,-233.3),(292.0,-622.2),(283.5,-622.0),
            (-440.8,296.1),(-432.3,295.6),(-176.9,210.7),(-183.5,205.4),
            (59.7,-25.2),(51.7,-22.4),(105.0,205.9),(97.5,201.9),
            (300.5,637.1),(300.3,628.6),(164.4,732.9),(157.4,733.0)):
    # 草带隔开的路段将原有路侧步道向外移动，附近仍须有人行铺装。
    offsets=((0,0),(.5,0),(-.5,0),(0,.5),(0,-.5),(1,0),(-1,0),(0,1),(0,-1))
    if not any(in_walkable_surface(x+dx,z+dz) for dx,dz in offsets):
        raise ValueError(f'Campus sidewalk is missing from a sampled mapped road at {(x,z)}')
for distance in (95,110):
    x,z=point_along(gym_road,distance)
    a,b=point_along(gym_road,distance-1),point_along(gym_road,distance+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    for offset in (-4.1,4.1):
        px,pz=x-offset*dz/length,z+offset*dx/length
        if not any(in_walkable_surface(px+u,pz+v) for u,v in ((0,0),(.5,0),(-.5,0),(0,.5),(0,-.5))):
            raise ValueError(f'The gymnasium loop sidewalk is missing at {distance} m')
# 弯路两侧的独立人行道持续存在，并避免被临路坡地拉成突然的横向陡折。
for distance in (50,70,90):
    x,z=point_along(huide_curve,distance)
    a,b=point_along(huide_curve,distance-1),point_along(huide_curve,distance+1)
    dx,dz=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dz)
    for offset in (-3.5,3.5):
        px,pz=x-offset*dz/length,z+offset*dx/length
        if not in_walkable_surface(px,pz):
            raise ValueError(f'The Hui De bend sidewalk is missing at {distance} m')
        if abs(terrain_triangle_at(px,pz)-terrain_triangle_at(x,z))>.6:
            raise ValueError(f'The Hui De bend sidewalk tilts abruptly at {distance} m')
# 该点位于 OSM way/231885481 的内部且离边界超过 7 米，应被面状铺装覆盖。
if not vertical_surface_hits(mapped_walk,-12.23,-243.63):
    raise ValueError('A mapped pedestrian area was treated as a perimeter-only path')

# 地图台阶只对地形较可信的路段补三维踏面，抽样核对其连续性和贴地高度。
stairways=json.loads((DATA/'campus-stair-profile.json').read_text(encoding='utf-8'))['routes']
stair_mesh=mesh_vertices('campus-network-stairs.mesh')
if not {'way/237893510','way/256680429','way/239872944'}.issubset({route['osm'] for route in stairways}):
    raise ValueError('Expected hillside stairways are missing')
for route in stairways:
    if route['max_rise_m']>.28:
        raise ValueError(f"An estimated stair riser is too high: {route['osm']}")
    points=route['centerline_m']
    length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
    for fraction in (.137,.361,.613,.887):
        x,z=point_along(points,length*fraction)
        levels=vertical_surface_hits(stair_mesh,x,z)
        height=terrain_triangle_at(x,z)
        if not levels or not .035 < max(levels)-height < .55:
            raise ValueError(f"Stair tread is missing or detached at {route['osm']} {fraction}")

# 聚翰斋旧台阶线接到车道中心，修正后的两端应接路侧人行面。
zhai_stair=next(route for route in stairways if route['osm']=='way/239872944')
for x,z in (zhai_stair['start_m'],zhai_stair['end_m']):
    if not any(in_walkable_surface(x+dx,z+dz) for dx,dz in ((0,0),(.15,0),(-.15,0),(0,.15),(0,-.15))):
        raise ValueError(f'Zhai stair has no adjoining pedestrian surface near {(x,z)}')
zhai_paving=mesh_vertices('campus-zhai-paving.mesh')
for x,z in ((277,-153),(300,-154),(303,-135),(299,-120)):
    if not any(abs(y-terrain_triangle_at(x,z)-.065)<.02 for y in vertical_surface_hits(zhai_paving,x,z)):
        raise ValueError(f'Zhai court paving is missing or detached at {(x,z)}')

# 抽查南北校区已映射的步行线与车道共节点，低路缘不能堵住这些通路。
for node_id,way_id in (('11904646221','1282922811'),('2400176757','231903097'),
                       ('2403047888','234530352'),('11885471547','1280430901'),
                       ('2399009540','256680456'),('11817861015','1272661383'),
                       ('2402100811','231885291')):
    refs=[n.get('ref') for n in osm_ways[way_id].findall('nd')]
    i=refs.index(node_id)
    adjacent=i+1 if i+1<len(refs) else i-1
    center=node_xy[node_id]
    other=node_xy[refs[adjacent]]
    distance=math.dist(center,other)
    if distance<.5:
        continue
    reach=min(7.0,distance*.85)
    end=tuple(center[j]+(other[j]-center[j])*reach/distance for j in range(2))
    start3=(center[0],terrain_triangle_at(*center)+.10,center[1])
    end3=(end[0],terrain_triangle_at(*end)+.10,end[1])
    if segment_hits_triangles(start3,end3,road_curbs):
        raise ValueError(f'Mapped pedestrian connection is blocked by a curb at {node_id}')

# 核对全部树干及路侧灌木的中心不落在人行带上。
for x,y,z,scale in trees:
    if abs(y-terrain_triangle_at(x,z))>.02:
        raise ValueError(f'A tree does not meet the rendered terrain at {(x,z)}')
    if in_walkable_surface(x,z):
        raise ValueError(f'A tree trunk obstructs a sidewalk at {(x,z)}')
for i in range(0,len(roadside_shrubs),18):
    cluster=roadside_shrubs[i:i+18]
    x=sum(v[0] for v in cluster)/len(cluster)
    z=sum(v[2] for v in cluster)/len(cluster)
    if in_walkable_surface(x,z):
        raise ValueError(f'Roadside planting obstructs a sidewalk at {(x,z)}')

# 小巴演示路线应对应地图车道，四个轮子的投影不能驶出路面。
with (ASSETS/'campus-shuttle-route.bin').open('rb') as f:
    if f.read(4)!=b'SZR1':
        raise ValueError('Invalid shuttle route header')
    route_count,=struct.unpack('<I',f.read(4))
    shuttle_points=list(struct.iter_unpack('<2f',f.read()))
reference=mapped_points('231530782')
if (route_count!=len(shuttle_points) or len(reference)!=route_count or
        any(math.dist(a,b)>.01 for a,b in zip(shuttle_points,reference))):
    raise ValueError('Shuttle route no longer follows the mapped road')
route_length=sum(math.dist(a,b) for a,b in zip(shuttle_points,shuttle_points[1:]))
if not 430<route_length<445:
    raise ValueError('Unexpected shuttle route length')

road_bins=collections.defaultdict(list)
for i in range(0,len(road_surface),3):
    tri=road_surface[i:i+3]
    for gx in range(math.floor(min(v[0] for v in tri)/cell),math.floor(max(v[0] for v in tri)/cell)+1):
        for gz in range(math.floor(min(v[2] for v in tri)/cell),math.floor(max(v[2] for v in tri)/cell)+1):
            road_bins[gx,gz].append(tri)
def on_vehicle_road(x,z):
    for a,b,c in road_bins[(math.floor(x/cell),math.floor(z/cell))]:
        denominator=(b[2]-c[2])*(a[0]-c[0])+(c[0]-b[0])*(a[2]-c[2])
        if abs(denominator)<1e-7:
            continue
        u=((b[2]-c[2])*(x-c[0])+(c[0]-b[0])*(z-c[2]))/denominator
        v=((c[2]-a[2])*(x-c[0])+(a[0]-c[0])*(z-c[2]))/denominator
        if min(u,v,1-u-v)>=-1e-5:
            return True
    return False

# 同时检查通用阶梯和旧桥板资源，防止残留的旧踏面继续伸入车道。
for mesh in (stair_mesh,deck):
    for x,_,z,_,ny,_,_,_ in mesh:
        if ny>.7 and 356<x<371 and -189<z<-180 and on_vehicle_road(x,z):
            raise ValueError('Zhai stair treads overlap the Shenda East Road carriageway')

# 桥柱的竖向方梁不得落在地面车道或人行层；斜撑不按竖向柱检查。
bridge_supports=mesh_vertices('campus-bridge-support.mesh')
for i in range(0,len(bridge_supports),24):
    pillar=bridge_supports[i:i+24]
    if len(pillar)!=24:
        raise ValueError('Bridge support geometry is truncated')
    xmin,xmax=min(v[0] for v in pillar),max(v[0] for v in pillar)
    zmin,zmax=min(v[2] for v in pillar),max(v[2] for v in pillar)
    if xmax-xmin>2 or zmax-zmin>2:
        continue
    x,z=(xmin+xmax)/2,(zmin+zmax)/2
    if on_vehicle_road(x,z) or in_walkable_surface(x,z):
        raise ValueError(f'A bridge support obstructs a road or sidewalk at {(x,z)}')

# 草带和灯具是可拆分的场景层：检查南北校区均有布置，并避开人行带与车行面。
verges=mesh_vertices('campus-network-verges.mesh')
grass=mesh_vertices('campus-network-verge-grass.mesh')
lamps=mesh_vertices('campus-network-lamps.mesh')
lamp_heads=mesh_vertices('campus-network-lamp-heads.mesh')
roadside=json.loads((DATA/'roadside-environment.json').read_text(encoding='utf-8'))
lamp_points=roadside['lamp_points_estimate_m']
if (len(verges)<1000 or len(grass)<100 or len(lamp_points)<20
        or len(lamps)!=24*len(lamp_points) or len(lamp_heads)!=48*len(lamp_points)):
    raise ValueError('Roadside verges or lamps are missing')
if not any(z<200 for x,z in lamp_points) or not any(z>400 for x,z in lamp_points):
    raise ValueError('Roadside lamps do not cover both campus areas')
for i in range(0,len(verges),max(3,(len(verges)//900//3)*3)):
    tri=verges[i:i+3]
    if len(tri)<3:
        continue
    # 草带与步道的公共边界经 float32 量化会产生极小三角形，不视为占道。
    a,b,c=tri
    area=abs((b[0]-a[0])*(c[2]-a[2])-(c[0]-a[0])*(b[2]-a[2]))*.5
    if area<.001:
        continue
    x=sum(v[0] for v in tri)/3;z=sum(v[2] for v in tri)/3
    on_walk,on_road=in_walkable_surface(x,z),on_vehicle_road(x,z)
    if on_walk or on_road:
        raise ValueError(f'Roadside verge blocks a road or footway at {(x,z)}: walk={on_walk}, road={on_road}')
    if abs(sum(v[1] for v in tri)/3-terrain_triangle_at(x,z)-.050)>.02:
        raise ValueError(f'Roadside verge is detached from terrain at {(x,z)}')
for x,z in lamp_points:
    if in_walkable_surface(x,z) or on_vehicle_road(x,z):
        raise ValueError(f'Roadside lamp blocks a road or footway at {(x,z)}')
    if any(math.hypot(x-a,z-b)<3.98 for a,_,b,_ in trees):
        raise ValueError(f'A tree intersects a roadside lamp at {(x,z)}')

def shuttle_center(distance):
    return point_along(shuttle_points,max(0,min(route_length,distance)))
def shuttle_lane(distance):
    distance=max(0,min(route_length,distance))
    x,z=shuttle_center(distance)
    a,b=shuttle_center(distance-1.5),shuttle_center(distance+1.5)
    dx,dz=b[0]-a[0],b[1]-a[1]
    magnitude=math.hypot(dx,dz)
    return x-dz/magnitude*1.2,z+dx/magnitude*1.2
for distance in list(range(0,int(route_length)+1,2))+[route_length]:
    x,z=shuttle_lane(distance)
    a,b=shuttle_lane(distance-1.5),shuttle_lane(distance+1.5)
    dx,dz=b[0]-a[0],b[1]-a[1]
    magnitude=math.hypot(dx,dz)
    dx,dz=dx/magnitude,dz/magnitude
    wheels={}
    for along in (-1.61,1.61):
        for side in (-.83,.83):
            px,pz=x+along*dx-side*dz,z+along*dz+side*dx
            if not on_vehicle_road(px,pz):
                raise ValueError(f'Shuttle wheel leaves the mapped road at {distance:.1f} m')
            wheels[along,side]=terrain_triangle_at(px,pz)
    level=sum(wheels.values())/4
    pitch=(wheels[1.61,.83]+wheels[1.61,-.83]-wheels[-1.61,.83]-wheels[-1.61,-.83])/(4*1.61)
    roll=(wheels[1.61,.83]+wheels[-1.61,.83]-wheels[1.61,-.83]-wheels[-1.61,-.83])/(4*.83)
    if max(abs(level+along*pitch+side*roll-y) for (along,side),y in wheels.items())>.15:
        raise ValueError(f'Shuttle road has an abrupt wheel-height discontinuity at {distance:.1f} m')

# 古石采用可调整的独立模型；检查其近景范围没有被路面或随机树干占据。
stone_config=json.loads((DATA/'wenshan-ancient-stone.json').read_text(encoding='utf-8'))
stone_x,stone_z=stone_config['position_xz_m']
ancient_stone=mesh_vertices('campus-wenshan-ancient-stone.mesh')
if len(ancient_stone)<3000 or not 2.0<max(v[1] for v in ancient_stone)-min(v[1] for v in ancient_stone)<3.0:
    raise ValueError('Ancient stone mesh is missing or has implausible proportions')
if any(math.hypot(x-stone_x,z-stone_z)<8.45 for x,_,z,_ in trees):
    raise ValueError('A procedural tree obstructs the ancient stone')
for v in ancient_stone[::max(1,len(ancient_stone)//120)]:
    if on_vehicle_road(v[0],v[2]) or in_walkable_surface(v[0],v[2]):
        raise ValueError('The ancient stone overlaps a mapped road or footpath')
stone_planting=mesh_vertices('campus-wenshan-ancient-stone-planting.mesh')
if len(stone_planting)<100 or any(on_vehicle_road(v[0],v[2]) or in_walkable_surface(v[0],v[2])
                                  for v in stone_planting):
    raise ValueError('Ancient stone planting is missing or blocks a road or footpath')
if min(v[1] for v in ancient_stone)>terrain_triangle_at(stone_x,stone_z)+.05:
    raise ValueError('Ancient stone floats above the terrain')
# 人行区域的底图蒙版必须保留石刻草地的孔洞，不能只在三维铺装上挖空。
with (ASSETS/'campus-ground.ppm').open('rb') as f:
    if f.readline().strip()!=b'P6':
        raise ValueError('Invalid campus ground texture')
    width,height=map(int,f.readline().split())
    if f.readline().strip()!=b'255':
        raise ValueError('Invalid campus ground texture range')
    _,_,x0,x1,z0,z1,_=terrain_data()
    px=round((stone_x-x0)/(x1-x0)*(width-1))
    pz=round((stone_z-z0)/(z1-z0)*(height-1))
    f.seek((pz*width+px)*3,1)
    red,green,blue=f.read(3)
    if green<=red+5 or green<=blue+5:
        raise ValueError('Ancient stone grass clearing was painted over by a pedestrian area')

# 核对完整演示环线及身体、水纹外缘均位于上文山湖水面，水位与渲染网格一致。
swan_config=json.loads((DATA/'wenshan-swans.json').read_text(encoding='utf-8'))
with (ASSETS/'campus-wenshan-swan-route.bin').open('rb') as f:
    if f.read(4)!=b'SWN1':
        raise ValueError('Invalid swan route header')
    swan_values=struct.unpack('<11f',f.read(44))
cx,cz,rx,rz,water_y,speed,*phases=swan_values
if (abs(cx-swan_config['center_xz_m'][0])>.001 or abs(cz-swan_config['center_xz_m'][1])>.001
        or len(phases)!=5 or speed<=0):
    raise ValueError('Swan route does not match its editable configuration')
lake_surface=mesh_vertices('campus-water.mesh')
for i in range(72):
    a=2*math.pi*i/72
    x,z=cx+rx*math.cos(a),cz+rz*math.sin(a)
    for dx,dz in ((0,0),(2,0),(-2,0),(0,2),(0,-2)):
        hits=vertical_surface_hits(lake_surface,x+dx,z+dz)
        if not any(abs(y-water_y)<.02 for y in hits):
            raise ValueError('Swan route leaves the rendered lake surface')
for name in ('adult','feather','bill','bill-tip','adult-eye','chick','chick-wing','chick-bill','chick-eye','wake'):
    if len(mesh_vertices(f'campus-wenshan-swan-{name}.mesh'))<100:
        raise ValueError(f'Swan model part is missing: {name}')

# 凯风餐厅须保留真实门洞，并使两个入口铺装与已有路侧步行层衔接。
kaifeng_config=json.loads((DATA/'kaifeng-canteen.json').read_text(encoding='utf-8'))
with (ASSETS/'campus-kaifeng-entry.bin').open('rb') as f:
    if f.read(4)!=b'KFE1':
        raise ValueError('Invalid Kai Feng entrance header')
    left_x,left_z,right_x,right_z,ux,uz,floor_y=struct.unpack('<7f',f.read(28))
cx,cz=(left_x+right_x)/2,(left_z+right_z)/2
nx,nz=-uz,ux
def kaifeng_point(s,d,y=0):
    distance=s-kaifeng_config['entry_center_m']
    return (cx+ux*distance+nx*(d-.12),floor_y+y,cz+uz*distance+nz*(d-.12))
entry_a,entry_b=kaifeng_point(18.5,2,1.5),kaifeng_point(18.5,-2,1.5)
for name in ('campus-kaifeng-shell.mesh','campus-wall-cream.mesh'):
    if segment_hits_triangles(entry_a,entry_b,mesh_vertices(name)):
        raise ValueError(f'Kai Feng doorway is blocked by {name}')
kaifeng_apron=mesh_vertices('campus-kaifeng-apron.mesh')
kaifeng_stairs=mesh_vertices('campus-kaifeng-stairs.mesh')
for s,inner,outer in ((18.5,8.8,9.3),(45.0,10.8,11.3)):
    x,_,z=kaifeng_point(s,inner)
    hits=vertical_surface_hits(kaifeng_apron,x,z)
    if not hits or abs(max(hits)-terrain_triangle_at(x,z))>.15:
        raise ValueError('Kai Feng apron does not meet surrounding terrain')
    x,_,z=kaifeng_point(s,outer)
    if not in_walkable_surface(x,z):
        raise ValueError('Kai Feng apron does not meet the roadside walking layer')
if any(on_vehicle_road(v[0],v[2]) for v in kaifeng_apron):
    raise ValueError('Kai Feng apron overlaps the vehicle road')
step_x,_,step_z=kaifeng_point(44.2,4)
land_x,_,land_z=kaifeng_point(44.4,4)
step=vertical_surface_hits(kaifeng_stairs,step_x,step_z)
landing=vertical_surface_hits(kaifeng_apron,land_x,land_z)
if not step or not landing or abs(max(step)-max(landing))>.12:
    raise ValueError('Kai Feng external staircase does not meet its ground landing')
top_x,_,top_z=kaifeng_point(34.05,3.2)
if len(vertical_surface_hits(kaifeng_stairs,top_x,top_z))<2:
    raise ValueError('Kai Feng staircase does not meet the upper balcony')
if buildings['way/230935723']['height_m']!='14.4':
    raise ValueError('Kai Feng landmark metadata is inconsistent')

print('Scene data verified: road junctions, slope surfaces, bridge connections and landmark resources.')
