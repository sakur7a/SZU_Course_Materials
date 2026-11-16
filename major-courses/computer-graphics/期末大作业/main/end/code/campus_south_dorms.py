"""南区两组U形学生宿舍：根据实拍制作可投影阴影的框洞、外廊及竖向端屏。

保留传入的OSM轮廓、估计高度和基底；六个命名分部独立输出，不修改全校地形。
参考图与证据边界见 data/south-dorm-sources.md；照片不用于运行纹理。
"""
import json
import math
import struct
from array import array
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union


def _polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom]
    return [p for g in getattr(geom, 'geoms', ()) for p in _polygons(g)]


def _lines(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'LineString':
        return [geom]
    return [p for g in getattr(geom, 'geoms', ()) for p in _lines(g)]


def _align_winding(mesh):
    """共享extrude的部分面按旧约定绕序；保存时与显式法线统一。"""
    v = mesh.vertices
    for i in range(0, len(v), 24):
        a, b, c = v[i:i+3], v[i+8:i+11], v[i+16:i+19]
        u, w = [b[k]-a[k] for k in range(3)], [c[k]-a[k] for k in range(3)]
        n = (u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0])
        if sum(n[k]*v[i+3+k] for k in range(3)) < 0:
            v[i+8:i+16], v[i+16:i+24] = v[i+16:i+24], v[i+8:i+16]


def _read_mesh(path):
    blob = path.read_bytes()
    if blob[:4] != b'SZM1' or len(blob) < 8:
        raise ValueError('Invalid bridge mesh: '+str(path))
    count, = struct.unpack_from('<I', blob, 4)
    if count % 3 or len(blob) != 8+count*32:
        raise ValueError('Invalid bridge vertex count')
    return list(struct.iter_unpack('<8f', blob[8:]))


def _mesh_height(vertices, x, z):
    """垂直取实际桥面上表面，不从地形或层数重新估算桥高。"""
    heights = []
    for i in range(0, len(vertices), 3):
        a, b, c = vertices[i:i+3]
        if min(a[4], b[4], c[4]) < .7:
            continue
        den = (b[2]-c[2])*(a[0]-c[0])+(c[0]-b[0])*(a[2]-c[2])
        if abs(den) < 1e-9:
            continue
        u = ((b[2]-c[2])*(x-c[0])+(c[0]-b[0])*(z-c[2]))/den
        v = ((c[2]-a[2])*(x-c[0])+(a[0]-c[0])*(z-c[2]))/den
        if min(u, v, 1-u-v) >= -1e-4:
            heights.append(u*a[1]+v*b[1]+(1-u-v)*c[1])
    return max(heights) if heights else None


def _skywalk_specs(parameters, assets, data):
    if not parameters.get('skywalk_interfaces'):
        return {}, []
    # 根生成顺序须先完成桥梁；缺少桥网格时直接报错，不使用过时硬编码高度。
    bridge = _read_mesh(assets/'campus-bridge.mesh')
    root = ET.parse(data/'yuehai.osm').getroot()
    nodes = {n.get('id'): ((float(n.get('lon'))-113.9318)*102800,
                           (22.5355-float(n.get('lat')))*111320) for n in root.findall('node')}
    ways = {w.get('id'): w for w in root.findall('way')}
    result = {}
    for config in parameters['skywalk_interfaces']:
        way = ways[config['source_osm'].split('/')[1]]
        points = [nodes[p.get('ref')] for p in way.findall('nd')]
        at_end = config['endpoint'] == 'end'
        edge, previous = (points[-1], points[-2]) if at_end else (points[0], points[1])
        distance = math.dist(edge, previous)
        inward = tuple((edge[i]-previous[i])/distance for i in range(2))
        deck = _mesh_height(bridge, *edge)
        if deck is None:
            raise ValueError('Mapped skywalk endpoint has no rendered bridge deck: '+config['id'])
        spec = dict(config)
        spec.update(edge_xz_m=list(edge), inward_xz=list(inward), deck_y_m=deck,
                    platform_top_y_m=deck+.012, platform_width_m=config['clear_width_m']+.30)
        cross = (-inward[1], inward[0])
        def point(s, t=0):
            return [edge[0]+inward[0]*s+cross[0]*t, edge[1]+inward[1]*s+cross[1]*t]
        spec['path_xz_m'] = [point(-config['outside_overlap_m']), point(0), point(config['inside_length_m'])]
        half = spec['platform_width_m']/2
        spec['platform_polygon_xz_m'] = [point(-config['outside_overlap_m'], -half),
            point(config['inside_length_m'], -half), point(config['inside_length_m'], half),
            point(-config['outside_overlap_m'], half)]
        profile = []
        for i in range(4):
            s = -config['outside_overlap_m']*(1-i/3)
            height = _mesh_height(bridge, *point(s))
            if height is None:
                raise ValueError('Skywalk overlap has no bridge surface: '+config['id'])
            profile.append({'s_m': s, 'bridge_y_m': height, 'platform_y_m': height+.012})
        spec['approach_profile'] = profile
        line = LineString(points)
        eye = line.interpolate(max(0, line.length-4) if at_end else min(4, line.length))
        eye_deck = _mesh_height(bridge, eye.x, eye.y)
        if eye_deck is None:
            raise ValueError('Skywalk preview eye has no bridge deck: '+config['id'])
        target = point(2.5)
        spec['view_eye_deck_y_m'] = eye_deck
        spec['view_eye'] = [eye.x, eye_deck+1.8, eye.y]
        spec['view_target'] = [target[0], deck+1.8, target[1]]
        spec['notes'] = parameters['skywalk_note']
        result.setdefault(config['owner_id'], []).append(spec)
    return result, bridge


def _clip_polygon(vertices, plane, keep_inside):
    """平面裁剪插值全部8个顶点分量，保留原法线、纹理方向和UV连续性。"""
    if not vertices:
        return []
    def distance(v):
        return plane[0]*v[0]+plane[1]*v[1]+plane[2]*v[2]+plane[3]
    result = []
    previous = vertices[-1]
    dp = distance(previous)
    previous_inside = dp >= -1e-8 if keep_inside else dp <= 1e-8
    for current in vertices:
        dc = distance(current)
        current_inside = dc >= -1e-8 if keep_inside else dc <= 1e-8
        if current_inside != previous_inside:
            alpha = dp/(dp-dc)
            result.append([previous[k]+(current[k]-previous[k])*alpha for k in range(8)])
        if current_inside:
            result.append(current)
        previous, dp, previous_inside = current, dc, current_inside
    return result


def _subtract_portal(mesh, spec):
    """逐三角面减去局部门洞盒；所有材质同样处理，避免仅移柱却留下玻璃。"""
    edge, u = spec['edge_xz_m'], spec['inward_xz']
    cross = (-u[1], u[0])
    s0, s1 = -spec['outside_overlap_m']-.18, spec['inside_length_m']+.35
    half = spec['clear_width_m']/2+.10
    y0 = min(p['bridge_y_m'] for p in spec['approach_profile'])-.22
    y1 = spec['platform_top_y_m']+spec['clear_height_m']+.05
    planes = [(u[0],0,u[1],-u[0]*edge[0]-u[1]*edge[1]-s0),
              (-u[0],0,-u[1],u[0]*edge[0]+u[1]*edge[1]+s1),
              (cross[0],0,cross[1],-cross[0]*edge[0]-cross[1]*edge[1]+half),
              (-cross[0],0,-cross[1],cross[0]*edge[0]+cross[1]*edge[1]+half),
              (0,1,0,-y0),(0,-1,0,y1)]
    corners = [(edge[0]+u[0]*s+cross[0]*t, edge[1]+u[1]*s+cross[1]*t)
               for s in (s0,s1) for t in (-half,half)]
    lower = (min(p[0] for p in corners),y0,min(p[1] for p in corners))
    upper = (max(p[0] for p in corners),y1,max(p[1] for p in corners))
    original, output = mesh.vertices, array('f')
    def emit(polygon):
        for i in range(1,len(polygon)-1):
            tri = (polygon[0],polygon[i],polygon[i+1])
            a,b,c = tri
            ab,ac = [b[k]-a[k] for k in range(3)],[c[k]-a[k] for k in range(3)]
            n = (ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0])
            if sum(q*q for q in n) < 1e-16:
                continue
            for v in tri:
                norm = math.sqrt(sum(v[k]*v[k] for k in (3,4,5)))
                output.extend((*v[:3],*(v[k]/norm for k in (3,4,5)),*v[6:8]))
    for i in range(0,len(original),24):
        triangle = [list(original[i+k:i+k+8]) for k in (0,8,16)]
        if any(max(v[k] for v in triangle)<lower[k]-1e-6 or
               min(v[k] for v in triangle)>upper[k]+1e-6 for k in range(3)):
            output.extend(original[i:i+24])
            continue
        pending = triangle
        for plane in planes:
            emit(_clip_polygon(pending,plane,False))
            pending = _clip_polygon(pending,plane,True)
            if len(pending)<3:
                break
    mesh.vertices = output


def _sign_texture(assets, config):
    # 仅夏筝取得清晰中英文门牌照片；不替其他五个分部虚构实景门牌。
    im = Image.new('RGB', (512, 256), (229, 229, 220))
    draw = ImageDraw.Draw(im)
    chinese = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 82)
    english = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 29)
    draw.text((256, 93), config['name'], font=chinese, anchor='mm', fill=(72, 57, 45))
    draw.text((256, 173), config['romanized'], font=english, anchor='mm', fill=(72, 57, 45))
    filename = f"dorm-{config['id']}-sign.ppm"
    im.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(assets/filename)
    return filename


class _Dorm:
    def __init__(self, config, parameters, geom, base, height, mesh_type, extrude, beam, height_at):
        self.config, self.parameters = config, parameters
        self.geom, self.base, self.height = orient(geom, sign=1.0), base, height
        self.mesh_type, self.extrude, self.beam, self.height_at = mesh_type, extrude, beam, height_at
        self.meshes = {role: mesh_type() for role in ('wall', 'frame', 'glass', 'metal', 'roof', 'solar', 'sign', 'paving')}
        self.walk = mesh_type()
        self.bays, self.fin_count, self.solar_count = 0, 0, 0
        self.beam_safe = {}
        self.recess = parameters['recess_m']
        self.podium = min(parameters['residential_start_m'], height*.4)
        self.floor_height = (height-self.podium)/(config['floors']-2)
        self.entry_lines, self.openings = [], Polygon()

    @staticmethod
    def edge(a, b):
        length = math.dist(a, b)
        tx, tz = (b[0]-a[0])/length, (b[1]-a[1])/length
        normal = (tz, -tx)
        def point(s, inset=0):
            return (a[0]+tx*s+normal[0]*inset, a[1]+tz*s+normal[1]*inset)
        return length, normal, point

    def solid(self, role, geom, low, high, entry=False):
        if high-low < .002:
            return
        geom = geom.intersection(self.geom)
        if entry:
            geom = geom.difference(self.openings)
        for poly in _polygons(geom):
            if poly.area > .0001:
                self.extrude(self.meshes[role], self.meshes[role], poly, self.base+low, self.base+high)

    def strip(self, role, p, start, end, front, back, low, high, entry=False):
        self.solid(role, Polygon((p(start, front), p(end, front), p(end, back), p(start, back))),
                   low, high, entry)

    def panel(self, role, p, start, end, inset, low, high, normal):
        a, b = p(start, inset), p(end, inset)
        original = LineString((a, b))
        # 斜尖角的平行内退面仍可能穿过相邻边，逐段裁回实际轮廓。
        for line in _lines(original.intersection(self.geom)):
            if line.length < .025:
                continue
            points = sorted((line.coords[0], line.coords[-1]), key=lambda q: original.project(Point(q)))
            a, b = points
            ua, ub = (1-original.project(Point(q))/original.length for q in points)
            self.meshes[role].quad((a[0], self.base+low, a[1]), (a[0], self.base+high, a[1]),
                                   (b[0], self.base+high, b[1]), (b[0], self.base+low, b[1]),
                                   (normal[0], 0, normal[1]), ((ua, 0), (ua, 1), (ub, 1), (ub, 0)))

    def member(self, role, p, s0, y0, s1, y1, inset, radius):
        a, b = p(s0, inset), p(s1, inset)
        if radius not in self.beam_safe:
            self.beam_safe[radius] = self.geom.buffer(-radius*1.5, join_style=2)
        safe = self.beam_safe[radius]
        length = math.dist(a, b)
        if length < .00001:
            if safe.covers(Point(a)):
                self.beam(self.meshes[role], (a[0], self.base+y0, a[1]), (b[0], self.base+y1, b[1]), radius)
            return
        original = LineString((a, b))
        for line in _lines(original.intersection(safe)):
            pa, pb = line.coords[0], line.coords[-1]
            ya = y0+(y1-y0)*original.project(Point(pa))/length
            yb = y0+(y1-y0)*original.project(Point(pb))/length
            self.beam(self.meshes[role], (pa[0], self.base+ya, pa[1]), (pb[0], self.base+yb, pb[1]), radius)

    def entrance(self, paths):
        if paths is None:
            # 尚无逐栋门洞测绘；默认通道只用于公共底层连通，不声称真实门位。
            rect = list(self.geom.minimum_rotated_rectangle.exterior.coords)
            a, b = max(zip(rect, rect[1:]), key=lambda pair: math.dist(*pair))
            length, normal, _ = self.edge(a, b)
            center = self.geom.representative_point()
            distance = max(length, 60)
            paths = LineString(((center.x-normal[0]*distance, center.y-normal[1]*distance),
                                (center.x+normal[0]*distance, center.y+normal[1]*distance)))
        if isinstance(paths, (list, tuple)):
            lines = [line for path in paths for line in _lines(path)]
        else:
            lines = _lines(paths)
        extended = []
        for line in lines:
            coords = list(line.coords)
            if not line.is_ring:
                for index, neighbor in ((0, 1), (-1, -2)):
                    end, previous = coords[index], coords[neighbor]
                    if self.geom.boundary.distance(Point(end)) < .4:
                        distance = math.dist(end, previous)
                        if distance > .01:
                            # 地图线常止于门边；仅在楼内延伸门洞约3米，不移动校外道路。
                            coords[index] = tuple(end[k]+(end[k]-previous[k])*3.0/distance for k in range(2))
            extended.append(LineString(coords))
        lines = extended
        self.entry_lines = [line for line in lines if line.intersects(self.geom)]
        self.openings = unary_union([line.buffer(self.parameters['entry_clear_width_m']/2+.18,
                                                cap_style=2, join_style=2) for line in self.entry_lines])
        # 入口铺面贴当前渲染地形，不把斜坡强制铺成悬空的同一水平板。
        paving = self.openings.intersection(self.geom)
        if paving.is_empty:
            return
        xmin, zmin, xmax, zmax = paving.bounds
        for xi in range(math.floor(xmin), math.ceil(xmax)):
            for zi in range(math.floor(zmin), math.ceil(zmax)):
                patch = paving.intersection(box(xi, zi, xi+1, zi+1))
                if patch.is_empty or patch.area < .001:
                    continue
                temp = self.mesh_type()
                temp.horizontal(patch, 0)
                for i in range(0, len(temp.vertices), 24):
                    points = [(temp.vertices[i+k], self.height_at(temp.vertices[i+k], temp.vertices[i+k+2])+.025,
                               temp.vertices[i+k+2]) for k in (0, 8, 16)]
                    a, b, c = points
                    u, v = [b[k]-a[k] for k in range(3)], [c[k]-a[k] for k in range(3)]
                    n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
                    if n[1] < 0:
                        points[1], points[2] = points[2], points[1]
                        n = [-q for q in n]
                    magnitude = math.sqrt(sum(q*q for q in n))
                    self.walk.triangle(*points, [q/magnitude for q in n], ((0, 0), (1, 0), (0, 1)))
        self.meshes['paving'].vertices.extend(self.walk.vertices)

    def lower_floors(self):
        core = self.geom.buffer(-2.15, join_style=2)
        self.solid('wall', core, -2, self.podium-.18, entry=True)
        # 上部公共楼层的底板与柱廊：照片支持两层基座，具体门窗布置为近似。
        self.solid('frame', self.geom, self.podium-.18, self.podium)
        lowest = min(self.height_at(x, z)-self.base for x, z in self.geom.exterior.coords)-.12
        self.solid('wall', self.geom, min(-.3, lowest), .015, entry=True)

    def podium_face(self, a, b):
        length, normal, p = self.edge(a, b)
        count = max(1, math.ceil(length/3.2))
        for j in range(count):
            s0, s1 = length*j/count+.06, length*(j+1)/count-.06
            pane = LineString((p(s0, -2.10), p(s1, -2.10)))
            if pane.intersects(self.openings.buffer(.08)):
                continue
            for floor in range(2):
                y0 = floor*self.podium/2+.20
                y1 = (floor+1)*self.podium/2-.20
                self.panel('glass', p, s0, s1, -2.10, y0, y1, normal)
                for s in (s0, s1):
                    self.member('metal', p, s, y0, s, y1, -2.07, .025)
            self.member('metal', p, s0, self.podium/2, s1, self.podium/2, -2.07, .035)
        count = max(1, round(length/8.0))
        for j in range(count+1):
            s = max(.34, min(length-.34, length*j/count))
            column = Polygon((p(s-.25, -.45), p(s+.25, -.45), p(s+.25, -1.0), p(s-.25, -1.0)))
            if column.intersects(self.openings.buffer(.12)):
                continue
            x, z = p(s, -.72)
            self.solid('wall', column, min(-.15, self.height_at(x, z)-self.base-.10), self.podium)

    def open_face(self, a, b, visible_from):
        length, normal, p = self.edge(a, b)
        bay = self.parameters['bay_width_m']
        thickness = self.parameters['frame_width_m']
        for floor in range(self.config['floors']-2):
            y = self.podium+floor*self.floor_height
            if y+self.floor_height < visible_from+.05:
                continue
            # 错排的长框洞是真正凹入的外廊；外框、楼板、后退门和栏杆独立成面。
            self.strip('frame', p, 0, length, -.015, -self.recess-.06, y-.13, y+.04)
            offset = (floor % 2)*bay*.5
            for j in range(-1, math.ceil(length/bay)+1):
                lo, hi = max(.08, j*bay+offset+.14), min(length-.08, (j+1)*bay+offset-.14)
                if hi-lo < .7:
                    continue
                top = min(self.height-.12, y+self.floor_height-.12)
                self.strip('frame', p, lo, hi, -.012, -.28, y+.04, y+.20)
                self.strip('frame', p, lo, hi, -.012, -.28, top-.18, top)
                for s in (lo+thickness/2, hi-thickness/2):
                    self.strip('frame', p, s-thickness/2, s+thickness/2, -.012, -.28, y+.13, top-.08)
                # 每个框内只概括一处后退门，不将公共晾晒/外廊误作封闭小玻璃窗。
                center = (lo+hi)/2
                half = min(.75, (hi-lo)*.22)
                self.panel('glass', p, center-half, center+half, -self.recess+.025,
                           y+.12, min(top-.2, y+2.40), normal)
                self.member('metal', p, center, y+.12, center, min(top-.2, y+2.40), -self.recess+.05, .018)
                for h in (.18, 1.05):
                    self.member('metal', p, lo+.12, y+h, hi-.12, y+h, -.36, .019)
                posts = max(1, math.ceil((hi-lo)/2.1))
                for k in range(posts+1):
                    s = lo+.12+(hi-lo-.24)*k/posts
                    self.member('metal', p, s, y+.15, s, y+1.05, -.36, .013)
                self.bays += 1

    def end_screen(self, a, b, visible_from):
        length, normal, p = self.edge(a, b)
        bottom = max(self.podium, visible_from)
        if bottom >= self.height-.1:
            return
        # 端墙的窄竖向暗缝与白色实体竖条分开建模，避免把端面画成普通点窗。
        self.panel('metal', p, .03, length-.03, -.23, bottom, self.height, normal)
        pitch, width = self.parameters['end_fin_pitch_m'], self.parameters['end_fin_width_m']
        count = max(1, round(length/pitch))
        for j in range(count):
            mid = (j+.5)*length/count
            self.strip('frame', p, max(.012, mid-width/2), min(length-.012, mid+width/2),
                       -.008, -.22, bottom, self.height+.20)
            self.fin_count += 1
        for i in range(self.config['floors']-1):
            y = self.podium+i*self.floor_height
            if bottom-.01 <= y <= self.height+.02:
                self.strip('frame', p, .01, length-.01, -.004, -.26, y-.075, y+.075)

    def shell(self, other_geoms, other_tops):
        self.lower_floors()
        contact = self.geom.boundary.intersection(other_geoms.buffer(.035))
        core = self.geom.buffer(-self.recess, join_style=2)
        # U形连接翼与高层相接处保留实体，不因每栋内缩生成两条竖向裂缝。
        if not contact.is_empty:
            core = core.union(self.geom.intersection(contact.buffer(self.recess+.06, cap_style=2, join_style=2)))
        self.solid('wall', core, self.podium, self.height-.12)
        self.solid('roof', self.geom, self.height-.16, self.height)
        ring = self.geom.difference(self.geom.buffer(-.18, join_style=2))
        self.solid('frame', ring, self.height, self.height+.46)
        coords = list(self.geom.exterior.coords)
        for a, b in zip(coords, coords[1:]):
            length = math.dist(a, b)
            if length < .6:
                continue
            line = LineString((a, b))
            mid = line.interpolate(.5, normalized=True)
            neighbor = max((top-self.base for geom, top in other_tops if geom.distance(mid) < .10), default=0)
            if neighbor < self.podium-.1:
                self.podium_face(a, b)
            if neighbor >= self.height-.05:
                continue
            # 冬筑庭院长侧被OSM拆成多个短段；只对照片支持的明确边段覆盖长度判据。
            # 此处仍位于相邻分部高度遮挡判断之后，不重新画出内部共墙。
            photo_open = any(min(max(math.dist(a, ends[0]), math.dist(b, ends[1])),
                                 max(math.dist(a, ends[1]), math.dist(b, ends[0]))) < .08
                             for ends in self.config.get('open_facade_segments_xz_m', ()))
            if length >= 28 or photo_open:
                self.open_face(a, b, neighbor)
            else:
                self.end_screen(a, b, neighbor)

    def solar(self):
        if not self.config.get('solar'):
            return
        safe = self.geom.buffer(-1.05, join_style=2)
        coords = list(self.geom.minimum_rotated_rectangle.exterior.coords)
        a, b = max(zip(coords, coords[1:]), key=lambda pair: math.dist(*pair))
        length, normal, p = self.edge(a, b)
        # 实拍屋顶为成排倾斜板状设备；不把尚未核实的用途标成光伏发电。
        for across in (2.3, 5.4, 8.5, 11.6, 14.7, 17.8, 20.9):
            for s in range(2, math.floor(length)-2, 3):
                plan = [p(s, -across), p(s+2.45, -across), p(s+2.45, -across-1.45), p(s, -across-1.45)]
                panel = Polygon(plan)
                if not safe.covers(panel):
                    continue
                low, high = self.base+self.height+.34, self.base+self.height+1.02
                points = [(plan[0][0], low, plan[0][1]), (plan[1][0], low, plan[1][1]),
                          (plan[2][0], high, plan[2][1]), (plan[3][0], high, plan[3][1])]
                n = (normal[0]*.68, 1.45, normal[1]*.68)
                mag = math.sqrt(sum(q*q for q in n))
                self.meshes['solar'].quad(*points, tuple(q/mag for q in n), ((0, 0), (1, 0), (1, 1), (0, 1)))
                # 细支架与低位板，构件不用于凑设计院项目最大69.9米。
                for q in (0, 2):
                    x, z = plan[q]
                    y = low if q == 0 else high
                    self.beam(self.meshes['metal'], (x, self.base+self.height+.08, z), (x, y, z), .027)
                self.solar_count += 1

    def sign(self):
        if not self.config.get('sign'):
            return
        coords = list(self.geom.exterior.coords)
        edges = [(a, b) for a, b in zip(coords, coords[1:]) if 8 < math.dist(a, b) < 28]
        a, b = min(edges, key=lambda pair: (pair[0][0]+pair[1][0])/2)
        length, normal, p = self.edge(a, b)
        center = length*.26
        # 门牌字体参考可辨“夏筝 / XIAZHENG”照片；具体牌位仍为近似。
        self.strip('wall', p, center-1.7, center+1.7, -.02, -.30, 1.45, 3.20)
        self.panel('sign', p, center-1.65, center+1.65, -.012, 1.48, 3.15, normal)

    def skywalk(self, spec, bridge):
        """独立的高架面，不写入原来的首层ground walking。"""
        edge, inward = spec['edge_xz_m'], spec['inward_xz']
        cross = (-inward[1], inward[0])
        half = spec['platform_width_m']/2
        thickness = spec['platform_thickness_m']
        walking = self.mesh_type()
        def point(s, t):
            x, z = edge[0]+inward[0]*s+cross[0]*t, edge[1]+inward[1]*s+cross[1]*t
            height = spec['deck_y_m']
            if s < -.00001:
                sampled = _mesh_height(bridge, x, z)
                if sampled is None:
                    sampled = _mesh_height(bridge, edge[0]+inward[0]*s, edge[1]+inward[1]*s)
                if sampled is None:
                    raise ValueError('Skywalk approach lost bridge support: '+spec['id'])
                height = sampled
            return (x, height+.012, z)
        def triangle(points, meshes, desired):
            a,b,c = points
            u,v = [b[k]-a[k] for k in range(3)],[c[k]-a[k] for k in range(3)]
            normal = [u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
            if sum(normal[k]*desired[k] for k in range(3)) < 0:
                points = (points[0],points[2],points[1])
                normal = [-q for q in normal]
            magnitude = math.sqrt(sum(q*q for q in normal))
            normal = tuple(q/magnitude for q in normal)
            for mesh in meshes:
                mesh.triangle(*points,normal,((0,0),(1,0),(1,1)))
        ss = [p['s_m'] for p in spec['approach_profile']]+[spec['inside_length_m']]
        tt = [-half,-half/2,0,half/2,half]
        grid = [[point(s,t) for t in tt] for s in ss]
        paving = self.meshes['paving']
        for j in range(len(ss)-1):
            for i in range(len(tt)-1):
                corners = (grid[j][i],grid[j+1][i],grid[j+1][i+1],grid[j][i+1])
                bottom = tuple((p[0],p[1]-thickness,p[2]) for p in corners)
                for indices in ((0,1,2),(0,2,3)):
                    triangle(tuple(corners[k] for k in indices),(paving,walking),(0,1,0))
                    triangle(tuple(bottom[k] for k in indices),(paving,),(0,-1,0))
        edges = ((grid[0],(-inward[0],0,-inward[1])),(grid[-1],(inward[0],0,inward[1])),
                 ([row[0] for row in grid],(-cross[0],0,-cross[1])),
                 ([row[-1] for row in grid],(cross[0],0,cross[1])))
        for points,desired in edges:
            for a,b in zip(points,points[1:]):
                corners = (a,b,(b[0],b[1]-thickness,b[2]),(a[0],a[1]-thickness,a[2]))
                for indices in ((0,1,2),(0,2,3)):
                    triangle(tuple(corners[k] for k in indices),(paving,),desired)
        # 侧栏设在净宽之外，外重叠处随原桥坡度，门前不横加一道栏杆。
        for side in (-1,1):
            lateral = side*(half-.035)
            for a,b in zip(ss,ss[1:]):
                pa,pb = point(a,lateral),point(b,lateral)
                self.beam(self.meshes['metal'],(pa[0],pa[1]+1.02,pa[2]),(pb[0],pb[1]+1.02,pb[2]),.024)
            for s in (-spec['outside_overlap_m']+.06,.8,1.9,spec['inside_length_m']-.05):
                p = point(s,lateral)
                self.beam(self.meshes['metal'],(p[0],p[1]+.03,p[2]),(p[0],p[1]+1.02,p[2]),.022)
        return walking


def make_dorms(*, buildings, height_for, base_height, mesh_type, extrude, beam,
               height_at, assets, data, entrance_paths=None):
    """base_height沿用根生成器基底；entrance_paths可为osm_key->线几何字典。"""
    parameters = json.loads((data/'south-dorms.json').read_text(encoding='utf-8'))
    skywalk_specs, bridge = _skywalk_specs(parameters, assets, data)
    by_osm = {key: (tags, geom) for key, tags, geom in buildings}
    configs = parameters['buildings']
    geoms, levels = {}, {}
    for config in configs:
        key = config['osm']
        tags, geom = by_osm[key]
        if tags.get('name') != config['name'] or int(tags.get('building:levels', 0)) != config['floors']:
            raise ValueError('South dorm OSM name or floor binding changed: '+key)
        height, basis = height_for(key, tags, geom)
        geoms[key], levels[key] = geom, (float(base_height(key, tags, geom)), float(height), basis)
    palette = {'wall': ('stone', [.82, .83, .79], 'concrete.ppm'),
               'frame': ('stone', [.94, .95, .91], 'white.ppm'),
               'glass': ('glass', [.58, .68, .68], 'glass.ppm'),
               'metal': ('metal', [.43, .47, .46], 'white.ppm'),
               'roof': ('roof', [.72, .74, .71], 'concrete.ppm'),
               'paving': ('stone', [.68, .70, .65], 'concrete.ppm'),
               'solar': ('glass', [.72, .80, .81], 'glass.ppm'),
               'sign': ('sign', [1, 1, 1], None)}
    records, layers, walks = [], [], []
    for config in configs:
        key, ident = config['osm'], config['id']
        base, height, basis = levels[key]
        dorm = _Dorm(config, parameters, geoms[key], base, height, mesh_type, extrude, beam, height_at)
        paths = entrance_paths.get(key) if isinstance(entrance_paths, dict) else entrance_paths
        dorm.entrance(paths)
        others = unary_union([geom for other, geom in geoms.items() if other != key])
        other_tops = [(geom, levels[other][0]+levels[other][1]) for other, geom in geoms.items() if other != key]
        dorm.shell(others, other_tops)
        dorm.solar()
        dorm.sign()
        elevated = []
        for spec in skywalk_specs.get(ident, ()):
            for mesh in dorm.meshes.values():
                if mesh.vertices:
                    _subtract_portal(mesh,spec)
            elevated.append((spec,dorm.skywalk(spec,bridge)))
        sign_texture = _sign_texture(assets, config) if config.get('sign') else None
        own_layers, all_bounds = [], []
        for role, mesh in dorm.meshes.items():
            if not mesh.vertices:
                continue
            _align_winding(mesh)
            filename = f'campus-dorm-{ident}-{role}.mesh'
            count = mesh.save(assets/filename)
            material, color, texture = palette[role]
            layer = {'id': ident, 'role': role, 'material': material, 'mesh': filename,
                     'texture': sign_texture if role == 'sign' else texture, 'color': color, 'vertices': count}
            layers.append(layer)
            own_layers.append(layer)
            all_bounds.extend((mesh.vertices[i], mesh.vertices[i+1], mesh.vertices[i+2])
                              for i in range(0, len(mesh.vertices), 8))
        walking_file = None
        if dorm.walk.vertices:
            _align_winding(dorm.walk)
            walking_file = f'campus-dorm-{ident}-walk.mesh'
            count = dorm.walk.save(assets/walking_file)
            walks.append({'id': ident, 'mesh': walking_file, 'vertices': count})
        elevated_files = []
        skywalk_records = []
        for spec,mesh in elevated:
            _align_winding(mesh)
            filename = f'campus-dorm-{ident}-skywalk.mesh'
            count = mesh.save(assets/filename)
            elevated_files.append(filename)
            spec = dict(spec,walking_mesh=filename,walking_vertices=count)
            skywalk_records.append(spec)
            walks.append({'id':spec['id'],'owner_id':ident,'mesh':filename,'vertices':count,
                          'source_osm':spec['source_osm'],'deck_y_m':spec['deck_y_m']})
        records.append({'id': ident, 'name': config['name'], 'osm': key, 'group': config['group'],
                        'base_y_m': base, 'height_m': height, 'floors': config['floors'],
                        'height_source': basis+'；设计院项目最大69.9米尚未分解到本OSM分部，保留既有高度估计',
                        'project_max_height_m': parameters['project_max_height_m'],
                        'bounds_xyz_m': [[min(p[i] for p in all_bounds) for i in range(3)],
                                         [max(p[i] for p in all_bounds) for i in range(3)]],
                        'layers': own_layers, 'walking_mesh': walking_file,
                        'skywalks': skywalk_records, 'elevated_walking_meshes': elevated_files,
                        'walking_vertices': len(dorm.walk.vertices)//8,
                        'entry_xz_m': list(nearest_points(dorm.geom.boundary, dorm.entry_lines[0])[0].coords[0])
                                      if dorm.entry_lines else None,
                        'entry_paths_xz_m': [list(line.coords) for line in dorm.entry_lines],
                        'entry_clear_width_m': parameters['entry_clear_width_m'],
                        'entry_clear_height_m': dorm.podium-.2,
                        'balcony_bays': dorm.bays, 'end_fins': dorm.fin_count, 'solar_panels': dorm.solar_count,
                        'features': ['staggered_recessed_frames', 'metal_rails', 'vertical_end_screens', 'open_lower_levels'],
                        'notes': parameters['notes']})
    return {'buildings': records, 'layers': layers, 'walking_meshes': walks,
            'exclude_osm_buildings': [config['osm'] for config in configs]}
