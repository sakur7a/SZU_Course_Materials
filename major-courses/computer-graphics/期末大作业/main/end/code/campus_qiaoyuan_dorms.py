"""乔苑五栋独立宿舍模型：真实凹阳台、板带、栏杆、屋顶框架及一期太阳能。

坐标沿用 OSM 轮廓；名称以 campus-places.json 的当前绑定为准。校方楼层与
高度记录和照片不能确定的尺寸分别保存在 data/qiaoyuan-dorms.json。此模块
只生成自己的资源，不改地形、公共索引或 C++，照片不作为发布纹理。
"""
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union


def _unit(vector):
    length = math.hypot(*vector)
    return (vector[0]/length, vector[1]/length)


def _align_winding(mesh):
    """共用挤出工具的旧绕序与存储法线相反；只修正本模块输出的三角面。"""
    values = mesh.vertices
    for i in range(0, len(values), 24):
        a, b, c = values[i:i+8], values[i+8:i+16], values[i+16:i+24]
        u = [b[j]-a[j] for j in range(3)]
        v = [c[j]-a[j] for j in range(3)]
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        if sum(n[j]*a[j+3] for j in range(3)) < 0:
            values[i+8:i+16], values[i+16:i+24] = c, b


def _textures(assets):
    """灰色小面砖只重画低对比缝线，不把学生航拍或水印贴进模型。"""
    image = Image.new('RGB', (256, 256), (218, 221, 217))
    draw = ImageDraw.Draw(image)
    for j in range(16):
        y = j*16
        draw.line((0, y, 255, y), fill=(199, 204, 199), width=1)
        for x in range(-32 if j % 2 else 0, 256, 64):
            draw.line((x, y, x, y+15), fill=(201, 206, 202), width=1)
    filename = 'qiaoyuan-dorm-grey-tile.ppm'
    image.save(assets/filename)
    return filename


def _sign_texture(assets, config):
    red = config.get('sign_style') == 'red_plaque'
    image = Image.new('RGB', (768, 256), (166, 43, 43) if red else (224, 228, 222))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 113)
    small = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 39)
    color = (247, 222, 164) if red else (37, 43, 42)
    draw.text((384, 10), config['name'], font=font, fill=color, anchor='mt')
    draw.text((384, 161), config['id'].upper(), font=small, fill=color, anchor='mt')
    filename = f'qiaoyuan-{config["id"]}-sign.ppm'
    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(assets/filename)
    return filename


class _Dorm:
    def __init__(self, config, geom, base, mesh_type, extrude, beam, height_at):
        self.config, self.geom, self.base = config, orient(geom, sign=1), base
        self.mesh_type, self.extrude_fn, self.beam_fn = mesh_type, extrude, beam
        self.height_at = height_at
        self.meshes = {name: mesh_type() for name in
                       ('wall', 'slab', 'shadow', 'glass', 'metal', 'yellow', 'roof', 'solar', 'sign')}
        self.walk = mesh_type()
        self.recess = config['balcony_depth_m']
        self.core = self.geom.buffer(-self.recess, join_style=2)
        self.openings = None
        self.facade_bays = []
        self.entry_lines = []
        self.solar_count = 0

    def solid(self, role, polygon, bottom, top, *, openings=False):
        if openings and self.openings is not None:
            polygon = polygon.difference(self.openings)
        if polygon.is_empty or top-bottom < .001:
            return
        mesh = self.meshes[role]
        self.extrude_fn(mesh, mesh, polygon, self.base+bottom, self.base+top,
                        unit_width=.5 if role == 'wall' and self.config['phase'] == 1 else 4,
                        unit_height=.5 if role == 'wall' and self.config['phase'] == 1 else 3.3)
        # 共用 extrude 只有上盖；阳台、外挑檐板补朝下底面，近景不能透出天空。
        underside = self.mesh_type()
        underside.horizontal(polygon, self.base+bottom)
        values = underside.vertices
        for i in range(0, len(values), 24):
            vertices = [values[i+j:i+j+8] for j in (0, 8, 16)]
            vertices.reverse()
            mesh.triangle(*(tuple(v[:3]) for v in vertices), (0, -1, 0),
                          tuple((v[6], v[7]) for v in vertices))

    def quad(self, role, points, normal, uv=None):
        world = [(x, self.base+y, z) for x, y, z in points]
        # 法线和顶点绕序保持一致；反面不是靠取消裁剪临时掩盖。
        a, b, c = (np.asarray(p) for p in world[:3])
        n = np.cross(b-a, c-a)
        tex = uv or ((0, 0), (1, 0), (1, 1), (0, 1))
        if np.dot(n, normal) < 0:
            world = world[::-1]
            tex = tex[::-1]
        self.meshes[role].quad(*world, normal, tex)

    def beam(self, role, a, b, radius=.025):
        self.beam_fn(self.meshes[role], (a[0], self.base+a[1], a[2]),
                     (b[0], self.base+b[1], b[2]), radius)

    @staticmethod
    def edge_frame(a, b):
        tangent = _unit((b[0]-a[0], b[1]-a[1]))
        normal = (tangent[1], -tangent[0])
        def p(t, v):
            return (a[0]+t*tangent[0]+v*normal[0], a[1]+t*tangent[1]+v*normal[1])
        return tangent, normal, p

    @staticmethod
    def rectangle(p, t0, t1, v0, v1):
        return Polygon((p(t0, v0), p(t1, v0), p(t1, v1), p(t0, v1)))

    def façade_panel(self, role, p, t0, t1, depth, low, high, normal):
        a, b = p(t0, depth), p(t1, depth)
        self.quad(role, ((a[0], low, a[1]), (b[0], low, b[1]),
                         (b[0], high, b[1]), (a[0], high, a[1])),
                  (normal[0], 0, normal[1]))

    def window(self, p, normal, center, width, floor, pitch, *, outside_depth=None):
        depth = -self.recess+.035 if outside_depth is None else outside_depth
        low, high = floor+.43, floor+min(2.75, pitch-.28)
        left, right = center-width/2, center+width/2
        self.façade_panel('glass', p, left, right, depth, low, high, normal)
        # 门窗的实体边框、中央梃及横梃在玻璃外侧投下阴影。
        for t in (left, center, right):
            a = p(t, depth+.025)
            self.beam('metal', (a[0], low, a[1]), (a[0], high, a[1]), .024)
        for y in (low, low+(high-low)*.45, high):
            a, b = p(left, depth+.025), p(right, depth+.025)
            self.beam('metal', (a[0], y, a[1]), (b[0], y, b[1]), .024)

    def balcony(self, p, normal, left, right, floor, floor_index):
        width = right-left
        phase = self.config['phase']
        back = -self.recess+.08
        if phase == 2:
            front = (p(left+.04, -.03), p(left+width*.19, .65),
                     p(right-width*.19, .65), p(right-.04, -.03))
        else:
            front = (p(left+.04, .04), p(right-.04, .04))
        polygon = Polygon((p(left, back), p(right, back), *front[::-1]))
        self.solid('slab', polygon, floor-.19, floor)
        self.façade_panel('shadow', p, left+.07, right-.07, -self.recess+.015,
                          floor+.02, floor+3.05, normal)
        if phase == 2:
            for center in (left+width*.26, left+width*.74):
                self.window(p, normal, center, width*.38, floor, 3.3)
            # 照片里的宽阳台组由细白立柱分成两个凹口。
            middle = (left+right)/2
            self.solid('slab', self.rectangle(p, middle-.065, middle+.065, back-.08, .67),
                       floor, floor+3.12)
        else:
            self.window(p, normal, (left+right)/2, width*.69, floor, 3.3)
        # 分隔侧墙用实体薄片连接凹窗与外侧板，保留阳台的真实深度。
        for t in (left, right):
            self.solid('wall', self.rectangle(p, t-.07, t+.07, back-.08, .03),
                       floor, floor+3.12)
        if phase == 2 and floor_index % 2 == 0:
            # 实景为交替的百叶板/竖栏；用六片实体叶片保留透空，不贴平灰矩形。
            for a, b in zip(front, front[1:]):
                for j in range(6):
                    y = floor+.22+j*.12
                    self.beam('metal', (a[0], y, a[1]), (b[0], y, b[1]), .035)
                self.beam('slab', (a[0], floor+.08, a[1]), (b[0], floor+.08, b[1]), .065)
                self.beam('slab', (a[0], floor+.96, a[1]), (b[0], floor+.96, b[1]), .055)
        else:
            for a, b in zip(front, front[1:]):
                for y in (floor+.12, floor+1.04):
                    self.beam('metal', (a[0], y, a[1]), (b[0], y, b[1]), .025)
                if phase == 1:
                    for y in (floor+.36, floor+.59, floor+.81):
                        self.beam('metal', (a[0], y, a[1]), (b[0], y, b[1]), .022)
                else:
                    count = max(1, math.ceil(math.dist(a, b)/.32))
                    for j in range(count+1):
                        t = j/count
                        x, z = a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])
                        self.beam('metal', (x, floor+.12, z), (x, floor+1.04, z), .016)
        self.facade_bays.append({'floor': floor_index+1, 'floor_y_m': self.base+floor,
                                 'center_xz_m': list(p((left+right)/2, 0)),
                                 'clearance_sample_xz_m': list(p(left+width*.26, 0)) if phase == 2 else list(p((left+right)/2, 0)),
                                 'outward_xz': list(normal),
                                 'recess_m': self.recess})

    def entrance(self, paths):
        config = self.config
        if paths is None:
            anchor = nearest_points(self.geom.boundary, Point(config['entry_hint_xz_m']))[0]
            inside = self.geom.representative_point()
            inward = _unit((inside.x-anchor.x, inside.y-anchor.y))
            line = LineString(((anchor.x-inward[0]*1.8, anchor.y-inward[1]*1.8),
                               (anchor.x+inward[0]*4, anchor.y+inward[1]*4)))
            lines = [line]
        else:
            lines = [paths] if paths.geom_type == 'LineString' else list(paths.geoms)
            anchor = nearest_points(self.geom.boundary, lines[0])[0]
        self.entry_lines = lines
        self.openings = unary_union([line.buffer(config['entry_clear_width_m']/2+.18,
                                                cap_style=3, join_style=2) for line in lines])
        # 只在入口邻近面设置简洁檐板。精确入口与2022采购的雨棚尺寸尚未核实。
        coords = list(self.geom.exterior.coords)
        edge = min((ab for ab in zip(coords, coords[1:]) if math.dist(*ab) > 3.8),
                   key=lambda ab: LineString(ab).distance(anchor))
        _, normal, p = self.edge_frame(*edge)
        along = LineString(edge).project(anchor)
        canopy = self.rectangle(p, along-1.85, along+1.85, -1.25, 1.15)
        self.solid('roof', canopy, config['entry_clear_height_m']+.20,
                   config['entry_clear_height_m']+.39)
        # 估计柱体只留在门洞两侧的楼体内。入口若处于Y形凹角的边端，不能把
        # 沿边参数外推成轮廓外的孤柱；外挑小檐板按依附楼体的雨棚表达。
        for t in (along-1.70, along+1.70):
            column = self.rectangle(p, t-.12, t+.12, -.45, -.18).intersection(self.geom)
            self.solid('wall', column, -.15, config['entry_clear_height_m']+.20, openings=True)
        # 门牌逐面寻找真实不与切洞相交的侧墙位置；斜向园路不能被牌边挡住。
        for edge_a, edge_b in sorted(zip(coords, coords[1:]),
                                     key=lambda ab: LineString(ab).distance(anchor)):
            edge_length = math.dist(edge_a, edge_b)
            if edge_length < 2.1:
                continue
            _, sign_normal, sign_p = self.edge_frame(edge_a, edge_b)
            sign_along = LineString((edge_a, edge_b)).project(anchor)
            candidates = (sign_along+3, sign_along-3, edge_length/2, .95, edge_length-.95)
            found = False
            for sign_t in candidates:
                sign_t = max(.95, min(edge_length-.95, sign_t))
                support = self.rectangle(sign_p, sign_t-.94, sign_t+.94, -.38, -.06)
                if self.openings.intersects(support.buffer(.06)):
                    continue
                self.solid('wall', support, -.1, 2.25)
                self.façade_panel('sign', sign_p, sign_t-.85, sign_t+.85, -.045,
                                  1.20, 1.77, sign_normal)
                found = True
                break
            if found:
                break
        # 门前原地形有横坡和建筑基底差，入口不能再用 base+.035 的平板。
        # 可见铺面与步行面共享同一批细分贴地三角面，不改现有地形或楼体标高。
        padded = []
        for line in lines:
            coords = list(line.coords)
            start = _unit((coords[1][0]-coords[0][0], coords[1][1]-coords[0][1]))
            end = _unit((coords[-1][0]-coords[-2][0], coords[-1][1]-coords[-2][1]))
            # 2cm端余量让原线路端点处于铺面内部，避免float32边界舍入漏采样。
            coords[0] = (coords[0][0]-.02*start[0], coords[0][1]-.02*start[1])
            coords[-1] = (coords[-1][0]+.02*end[0], coords[-1][1]+.02*end[1])
            padded.append(LineString(coords))
        # 圆角连接把折线外侧也限制在1.05m距离带内，避免斜接尖角额外外扩。
        patch = unary_union([line.buffer(1.05, cap_style=2, join_style=1) for line in padded])
        self.draped_entry(patch)
        return [float(anchor.x), float(anchor.y)]

    def draped_entry(self, patch):
        step = .5
        x0, z0, x1, z1 = patch.bounds
        ix0, iz0 = math.floor(x0/step), math.floor(z0/step)
        for ix in range(ix0, math.ceil(x1/step)):
            for iz in range(iz0, math.ceil(z1/step)):
                cell = patch.intersection(box(ix*step, iz*step, (ix+1)*step, (iz+1)*step))
                if cell.is_empty or cell.area < 1e-8:
                    continue
                planar = self.mesh_type()
                planar.horizontal(cell, 0)
                values = planar.vertices
                for offset in range(0, len(values), 24):
                    a, b, c = [(values[offset+i], values[offset+i+2]) for i in (0, 8, 16)]
                    mid = lambda p, q: ((p[0]+q[0])/2, (p[1]+q[1])/2)
                    ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                    # 在共享边的中点再细分，保留连续边界并跟随DEM三角面的折坡。
                    for triangle in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)):
                        points = [(x, self.height_at(x, z)+.035, z) for x, z in triangle]
                        pa, pb, pc = (np.asarray(p) for p in points)
                        normal = np.cross(pb-pa, pc-pa)
                        if normal[1] < 0:
                            points[1], points[2] = points[2], points[1]
                            normal = -normal
                        length = np.linalg.norm(normal)
                        if length < 1e-9:
                            continue  # 裁剪边界小于float32精度的薄片不能生成退化三角面。
                        normal /= length
                        uv = tuple((p[0]/3, p[2]/3) for p in points)
                        # 向两份网格发送完全相同的顶点/法线，防止画面和脚下错层。
                        self.meshes['slab'].triangle(*points, tuple(normal), uv)
                        self.walk.triangle(*points, tuple(normal), uv)

    def shell(self):
        config = self.config
        roof = config.get('residential_roof_m', config['nominal_height_m'])
        entry_top = config['entry_clear_height_m']+.12
        # 首层按路径减去门洞，上层凹进核心不重新填住入口高度。
        self.solid('wall', self.core, -1.65, entry_top, openings=True)
        self.solid('wall', self.core, entry_top, roof)
        floor_ring = self.geom.difference(self.geom.buffer(-self.recess-.08, join_style=2))
        floors = [config['ground_floor_m']+j*config['standard_floor_m']
                  for j in range(config['floors']-1)]
        for y in floors:
            self.solid('slab', floor_ring, y-.20, y)
        self.solid('roof', self.geom, roof-.22, roof)
        # 基底沿现有地面向下补墙，不改变传入地形或将建筑整体抬高。
        foundation_low = min(self.height_at(x, z)-self.base for x, z in self.geom.exterior.coords)-.2
        self.solid('wall', self.geom.difference(self.openings), min(-.4, foundation_low), .025)
        coords = list(self.geom.exterior.coords)
        for edge_index, (a, b) in enumerate(zip(coords, coords[1:])):
            length = math.dist(a, b)
            _, normal, p = self.edge_frame(a, b)
            if length < .7:
                continue
            if length >= 10:
                count = max(1, round((length-.60)/config['bay_pitch_m']))
                width = (length-.70)/count
                for j in range(count):
                    left, right = .35+j*width+.12, .35+(j+1)*width-.12
                    for k, y in enumerate(floors):
                        self.balcony(p, normal, left, right, y, k+1)
                    # 一期黄竖带在照片中很醒目，按可辨节奏概括，不重复贴成整面黄色。
                    if config['phase'] == 1 and j % 3 == 1:
                        stripe = self.rectangle(p, right-.08, right+.62, -.35, .075)
                        self.solid('yellow', stripe, config['ground_floor_m'], roof-.30)
            else:
                # 端墙保留白灰主体和窄窗，不把每个Y形短端面都虚构成阳台。
                endwall = self.rectangle(p, .03, length-.03, -.34, -.06)
                self.solid('wall', endwall, -.2, entry_top, openings=True)
                self.solid('wall', endwall, entry_top, roof)
                if length > 2.6:
                    for y in floors:
                        self.window(p, normal, length/2, min(1.65, length-.7), y, 3.3,
                                    outside_depth=-.05)
            # 屋面低矮女儿墙与压顶留出太阳能/梁架内部空间。
            self.solid('roof', self.rectangle(p, .05, length-.05, -.23, -.02), roof, roof+.65)

    def solar(self):
        if not self.config['solar']:
            return
        roof = self.config['nominal_height_m']
        angle = self.config['solar_angle_degrees']
        origin = self.geom.centroid
        aligned = affinity.rotate(self.geom.buffer(-.85), -angle, origin=origin)
        width, depth = self.config['solar_panel_size_m']
        x0, z0, x1, z1 = aligned.bounds
        def world(x, z):
            a = math.radians(angle)
            return (origin.x+(x-origin.x)*math.cos(a)-(z-origin.y)*math.sin(a),
                    origin.y+(x-origin.x)*math.sin(a)+(z-origin.y)*math.cos(a))
        for ix in range(math.ceil((x1-x0)/(width+.30))):
            for iz in range(math.ceil((z1-z0)/(depth+.35))):
                x, z = x0+ix*(width+.30), z0+iz*(depth+.35)
                candidate = box(x, z, x+width, z+depth)
                if not aligned.covers(candidate):
                    continue
                corners = [world(xx, zz) for xx, zz in ((x, z), (x+width, z),
                                                       (x+width, z+depth), (x, z+depth))]
                points = [(a, roof+.27+(j >= 2)*.65, b) for j, (a, b) in enumerate(corners)]
                normal = np.cross(np.asarray(points[1])-points[0], np.asarray(points[2])-points[0])
                normal /= np.linalg.norm(normal)
                if normal[1] < 0:
                    normal = -normal
                self.quad('solar', points, tuple(normal))
                for a, b in zip(points, points[1:]+points[:1]):
                    self.beam('metal', a, b, .032)
                for j in range(1, 5):
                    t = j/5
                    a = tuple(points[0][k]+t*(points[1][k]-points[0][k]) for k in range(3))
                    b = tuple(points[3][k]+t*(points[2][k]-points[3][k]) for k in range(3))
                    self.beam('metal', a, b, .012)
                for j in (0, 1, 2, 3):
                    a, y, b = points[j]
                    self.beam('metal', (a, roof+.04, b), (a, y, b), .035)
                self.solar_count += 1

    def crown(self):
        if self.config['phase'] != 2:
            return
        roof = self.config['residential_roof_m']
        top = self.config['crown_top_m']
        beam_y = self.config['crown_beam_m']
        coords = list(self.geom.exterior.coords)
        # 实拍是有连续厚白檐口的开敞屋盖，不能用细杆和外露斜撑画成施工架；
        # 厚度与内挑深度为照片估计，梁下仍不填充成额外宿舍楼层。
        for a, b in zip(coords, coords[1:]):
            length = math.dist(a, b)
            if length < 10:
                continue
            _, normal, p = self.edge_frame(a, b)
            count = max(1, round(length/6.5))
            for j in range(count+1):
                t = .25+(length-.5)*j/count
                pier = self.rectangle(p, t-.25, t+.25, -.60, -.10)
                self.solid('roof', pier, roof, beam_y)
            canopy = self.rectangle(p, .05, length-.05, -2.30, -.08).intersection(self.geom)
            self.solid('roof', canopy, beam_y-1.40, beam_y+.22)
        # 最高端塔只取轮廓中凹口附近的短边，具体屋面构造仍是照片约束下估计。
        eligible = [(a, b) for a, b in zip(coords, coords[1:]) if 3.0 < math.dist(a, b) < 9.5]
        center = self.geom.centroid
        a, b = min(eligible, key=lambda ab: Point((ab[0][0]+ab[1][0])/2,
                                                   (ab[0][1]+ab[1][1])/2).distance(center))
        _, normal, p = self.edge_frame(a, b)
        length = math.dist(a, b)
        tower = self.rectangle(p, .05, length-.05, -2.2, -.08).intersection(self.geom)
        self.solid('roof', tower, roof-.5, top)
        # 狭长侧窗与端塔轮廓呼应照片；确切面别未测绘。
        self.façade_panel('glass', p, .28, max(.5, length-.28), -.055,
                          roof+.6, min(top-1.5, roof+5.9), normal)


def make_dorms(*, buildings, height_for, base_height, mesh_type, extrude, beam,
               height_at, assets, data, entrance_paths=None):
    """生成独立资源；base_height(key,tags,geom) 保持根生成器的同一建筑基底。

    buildings 为 (osm_key,tags,Shapely Polygon) 列表；entrance_paths 为可选的
    osm_key -> LineString/MultiLineString，优先保留既有花园园路，而不是重选门。
    """
    catalog = json.loads((data/'qiaoyuan-dorms.json').read_text(encoding='utf-8'))
    bindings = json.loads((data/'campus-places.json').read_text(encoding='utf-8'))['places']
    binding_names = {p['osm']: p['name'] for p in bindings if p.get('osm')}
    by_osm = {key: (tags, geom) for key, tags, geom in buildings}
    tile = _textures(assets)
    records, layers, walks = [], [], []
    palette = {'wall': ('stone', [.72, .74, .71]), 'slab': ('stone', [.88, .89, .86]),
               'shadow': ('stone', [.23, .28, .29]), 'glass': ('glass', [.38, .51, .54]),
               'metal': ('metal', [.61, .66, .65]), 'yellow': ('stone', [.92, .77, .35]),
               'roof': ('roof', [.87, .88, .84]), 'solar': ('glass', [.26, .39, .49]),
               'sign': ('sign', [1, 1, 1])}
    for config in catalog['buildings']:
        key, ident = config['osm'], config['id']
        if binding_names.get(key) != config['name']:
            raise ValueError(f'Qiaoyuan name binding changed: {key}')
        tags, geom = by_osm[key]
        height, height_basis = height_for(key, tags, geom)
        if abs(height-config['nominal_height_m']) > .05:
            raise ValueError(f'Qiaoyuan height configuration mismatch: {key}')
        base = float(base_height(key, tags, geom))
        dorm = _Dorm(config, geom, base, mesh_type, extrude, beam, height_at)
        entry = dorm.entrance((entrance_paths or {}).get(key))
        dorm.shell()
        dorm.solar()
        dorm.crown()
        sign = _sign_texture(assets, config)
        own_layers = []
        for role, mesh in dorm.meshes.items():
            if not len(mesh.vertices):
                continue
            _align_winding(mesh)
            filename = f'campus-qiaoyuan-{ident}-{role}.mesh'
            count = mesh.save(assets/filename)
            material, color = palette[role]
            layer = {'id': ident, 'material': material, 'role': role, 'mesh': filename,
                     'color': color, 'vertices': count}
            if role == 'sign':
                layer['texture'] = sign
            elif role == 'wall' and config['phase'] == 1:
                layer['texture'] = tile
            own_layers.append(layer)
            layers.append(layer)
        walking_file = f'campus-qiaoyuan-{ident}-walk.mesh'
        _align_winding(dorm.walk)
        walking_count = dorm.walk.save(assets/walking_file)
        walks.append({'id': ident, 'mesh': walking_file})
        records.append({'id': ident, 'name': config['name'], 'osm': key,
                        'base_y_m': base, 'height_m': height, 'height_source': height_basis,
                        'floors': config['floors'], 'layers': own_layers,
                        'walking_mesh': walking_file, 'walking_vertices': walking_count,
                        'walking_surface': 'terrain_draped', 'walking_visible_role': 'slab',
                        'walking_path_half_width_m': 1.05, 'walking_grid_step_m': .5,
                        'walking_end_padding_m': .02, 'walking_join_style': 'round',
                        'entry_xz_m': entry, 'entry_clear_width_m': config['entry_clear_width_m'],
                        'entry_clear_height_m': config['entry_clear_height_m'],
                        'entry_paths_xz_m': [list(line.coords) for line in dorm.entry_lines],
                        'balcony_bays': dorm.facade_bays, 'solar_panels': dorm.solar_count,
                        'features': config['features'], 'notes': config['notes']})
    return {'buildings': records, 'layers': layers, 'walking_meshes': walks,
            'exclude_osm_buildings': [c['osm'] for c in catalog['buildings']],
            'legacy_meshes_to_stop_drawing': ['campus-wall-qiaomu.mesh',
                                             'campus-wall-qiaosen.mesh',
                                             'campus-qiaoyuan-solar.mesh']}
