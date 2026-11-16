"""周南桥斋区端下层平台和踏步。地图固定两端，照片约束平面关系。"""
import math

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union


def layout(axis, stair, vehicle, patches, height_at):
    axis = LineString(axis)
    stair = LineString(stair)
    opening = axis.project(stair.interpolate(0))
    # 横向开口在靠斋区的桥侧，桥上层及门外车道独立保留。
    # 对齐主桥既有 4 米断面，避免为了下层改写上层三角网。
    opening_end = 4*math.floor(min(axis.length-.1,opening+.9)/4)
    opening_start = opening_end-5.2
    transition = opening_start-27.0
    top = height_at(*stair.coords[-1])+.055
    low = top-1.05
    upper_ground = height_at(*axis.coords[-1])
    step_start = 2.3
    step_run = .38
    steps = 7
    flight_end = step_start+steps*step_run
    tangent = (np.asarray(stair.coords[-1])-stair.coords[0])/stair.length
    normal = np.asarray((-tangent[1],tangent[0]))

    def bridge_side(s):
        p = axis.interpolate(s)
        a,b = axis.interpolate(s-.5),axis.interpolate(s+.5)
        tx,tz = b.x-a.x,b.y-a.y
        return np.asarray((p.x+7.42*tz/math.hypot(tx,tz),p.y-7.42*tx/math.hypot(tx,tz)))

    def edges(s):
        mid = np.asarray(stair.interpolate(s).coords[0])
        outside = np.interp(s,(step_start,flight_end,stair.length),(1.8,1.8,.85))
        # 靠深大东路的一边比绿地侧短，避免让踏面钻进受保护的道路边坡。
        street = np.interp(s,(step_start,flight_end,flight_end+2.,stair.length),(.96,.96,.62,.62))
        return mid-normal*outside,mid+normal*street

    # 侧向粉砖平台与桥板边缘相接，外侧较窄的踏步通过末端小平台接路缘。
    bridge_edges = [bridge_side(opening_start),bridge_side(opening_end)]
    foot_edges = list(edges(step_start))
    terrace = Polygon((*bridge_edges,*foot_edges)).convex_hull
    lead = substring(axis,transition,opening_end).buffer(7.6,cap_style='flat')
    stair_area = Polygon((*edges(step_start),*edges(stair.length),*edges(flight_end))).convex_hull
    portal_area = unary_union((terrace,stair_area))
    if portal_area.intersection(vehicle).area>.01:
        raise ValueError('斋区端步行平台与地图车道重叠')

    def floor(station):
        blend = np.clip((station-transition)/(opening_start-transition),0.,1.)
        return 12.0+(low-12.0)*blend

    # 只降低桥板下方和出口平台占据的局部地表。临街车道的高度取原值。
    index,old = next((i,p) for i,p in enumerate(patches) if p.contains(*stair.coords[0]))
    spacing = old.spacing/2
    east = old.x1+10.
    xs = np.arange(old.x0,east+spacing*.5,spacing)
    zs = np.arange(old.z0,old.z1+spacing*.5,spacing)
    values = np.array([[height_at(x,z) for x in xs] for z in zs])
    patch = type(old)(values,(old.x0,east,old.z0,old.z1),spacing)
    area = unary_union((lead,portal_area))
    xmin,zmin,xmax,zmax = area.buffer(2.3).bounds
    for j,z in enumerate(zs[1:-1],1):
        if not zmin<z<zmax:
            continue
        for i,x in enumerate(xs[1:-1],1):
            if not xmin<x<xmax:
                continue
            p = Point(x,z)
            if vehicle.distance(p)<.88:
                continue
            current = values[j,i]
            target = current
            if lead.distance(p)<2.3:
                weight = np.clip((2.3-lead.distance(p))/1.55,0.,1.)
                target = min(target,current-weight*max(0.,current-floor(axis.project(p))+.48))
            if portal_area.distance(p)<2.1:
                s = stair.project(p)
                rise = .15*min(7,max(0,math.floor((s-step_start)/step_run)))
                # 地面接头的最后半米回到原地表，不在道路旁挖出断坎。
                tie = np.clip((stair.length-s)/.75,0.,1.)
                weight = np.clip((2.1-portal_area.distance(p))/1.2,0.,1.)*tie
                target = min(target,current-weight*max(0.,current-(low+rise-.70)))
            values[j,i] = target
    patches[index] = patch
    return dict(axis=axis,stair=stair,terrace=terrace,edges=edges,bridge_edges=bridge_edges,
                floor=floor,low=low,top=top,opening_start=opening_start,opening_end=opening_end,
                transition=transition,step_start=step_start,step_run=step_run,steps=steps,upper_ground=upper_ground,
                flight_end=flight_end,area=portal_area,
                reserved=area.buffer(1.35).difference(vehicle))


def save_tile_texture(path):
    # 照片是细小方形粉砖，图案以米制 UV 重绘。
    tile = Image.new('RGB',(256,256),(168,145,140))
    draw = ImageDraw.Draw(tile)
    for row in range(16):
        for column in range(16):
            light = (11*row+7*column+3*row*column)%11-5
            x,y = column*16,row*16
            draw.rectangle((x+1,y+1,x+15,y+15),fill=(202+light,164+light,155+light))
    tile.save(path)


def build(plan, treads, walls, rails, beam, ground):
    stair = plan['stair']
    low,top = plan['low'],plan['top']
    step_start,run = plan['step_start'],plan['step_run']
    uv = ((0,0),(1,0),(1,1),(0,1))

    def vertex(p,y):
        return (p[0],y,p[1])

    def rail(points):
        for (a,ay),(b,by) in zip(points[:-1],points[1:]):
            beam(rails,vertex(a,ay+1.06),vertex(b,by+1.06),.034)
            beam(rails,vertex(a,ay+.10),vertex(b,by+.10),.019)
            count = max(1,math.ceil(math.dist(a,b)/.14))
            for t in np.linspace(0,1,count,endpoint=False):
                p,y = a+(b-a)*t,ay+(by-ay)*t
                beam(rails,vertex(p,y+.06),vertex(p,y+1.06),.016)

    def side_wall(a,b,y0,y1):
        dx,dz = b-a
        length = math.hypot(dx,dz)
        walls.quad(vertex(a,min(ground(*a)-.12,y0-.28)),vertex(b,min(ground(*b)-.12,y1-.28)),
                   vertex(b,y1),vertex(a,y0),(dz/length,0,-dx/length),uv)

    treads.horizontal_mapped(plan['terrace'],low,lambda x,z:(x/1.9,z/1.9))
    edges = plan['edges']
    sides = edges(step_start)
    # 从踏步的两侧各接回一个桥侧端点，开口两端不跨画栏杆。
    bridge = plan['bridge_edges']
    if math.dist(bridge[0],sides[0])+math.dist(bridge[1],sides[1]) > math.dist(bridge[0],sides[1])+math.dist(bridge[1],sides[0]):
        bridge = bridge[::-1]
    for a,b in zip(bridge,sides):
        side_wall(a,b,low,low)
        rail([(a,low),(b,low)])
    # 靠桥端的浅灰墙沿平台边缘立起，给出照片里的侧向开口轮廓。
    a = plan['bridge_edges'][-1]
    b = min(sides,key=lambda p:math.dist(a,p))
    direction = (b-a)/math.dist(a,b)
    normal = np.asarray((direction[1],-direction[0]))
    if plan['terrace'].covers(Point(a+direction+normal*.3)):
        normal = -normal
    start = a+normal*.14
    for near,far,height in ((0.,.8,3.2),(.8,2.7,1.25)):
        a,b = start+direction*near,start+direction*far
        c,d = b+normal*.65,a+normal*.65
        walls.horizontal(Polygon((a,b,c,d)),low+height)
        for p,q in ((a,b),(b,c),(c,d),(d,a)):
            side_wall(p,q,low+height,low+height)

    stations = [step_start+i*run for i in range(plan['steps']+1)]
    for i,(start,end) in enumerate(zip(stations[:-1],stations[1:])):
        a,b = edges(start)
        c,d = edges(end)
        y = low+.15*(i+1)
        treads.horizontal_mapped(Polygon((a,c,d,b)),y,lambda x,z:(x/1.9,z/1.9))
        walls.quad(vertex(a,y-.15),vertex(b,y-.15),vertex(b,y),vertex(a,y),
                   (-(c[0]-a[0])/run,0,-(c[1]-a[1])/run),uv)
        side_wall(a,c,y,y)
        side_wall(d,b,y,y)
    landing = (stations[-1],stations[-1]+2.,stair.length)
    for start,end in zip(landing[:-1],landing[1:]):
        a,b = edges(start)
        c,d = edges(end)
        treads.horizontal_mapped(Polygon((a,c,d,b)),top,lambda x,z:(x/1.9,z/1.9))
        side_wall(a,c,top,top)
        side_wall(d,b,top,top)
    for index in (0,1):
        rail([(edges(step_start)[index],low),(edges(landing[0])[index],top),
              (edges(landing[1])[index],top),(edges(stair.length-.35)[index],top)])

    return {'steps':plan['steps'],'bottom_y_m':round(low,3),'top_y_m':round(top,3),
            'stair_osm':'984608622','sidewalk_osm':'1282922809',
            'flight_start_m':step_start,'flight_end_m':plan['flight_end'],
            'opening_station_m':[plan['opening_start'],plan['opening_end']]}
