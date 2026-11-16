#!/usr/bin/env python3
"""独立核验地点标注资源；只用标准库，不导入生成器，不需要 OpenGL。

检查生成 JSON/18 列 TSV 的一致性、RGBA 图集和 UV、OSM 面内定位、校门标签、
用户商铺 manifest。OSM 几何检查含 multipolygon 的内环，能发现把湖标签放到
湖外或岛上的回归。高度仅检查资源一致性和门标签在已生成模型顶部之上；不把
标签高度、估计建筑高度或 OSM 快照当作现场测量。

未验证：汉字 OCR/朝向、运行时遮挡与可读性、真实建筑外貌、地图及商铺现状；
乔相阁对面四店的真实左右次序和打印店位置没有足够证据，不强行排序。
"""
import argparse
import csv
import json
import math
import struct
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
# 与数据使用的局部米制坐标约定一致；此处独立从经纬度计算，不读生成器结果。
LON0, LAT0, METERS_LON, METERS_LAT = 113.9318, 22.5355, 102800.0, 111320.0
KEY_AREAS = {
    'upper-wenshan': 'way/234739309',
    'lower-wenshan': 'relation/3085354',
    'south-sports': 'way/231692662',
    'haibin': 'way/1271848103',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def vector(value, length, label):
    require(isinstance(value, list) and len(value) == length, f'{label}: expected {length} numbers')
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in value), f'{label}: non-finite or nonnumeric component')
    return value


def index_records(records, label):
    result = {}
    for item in records:
        ident = item['id']
        require(isinstance(ident, str) and ident and ident not in result,
                f'{label}: missing or duplicate identifier {ident!r}')
        require(isinstance(item['name'], str) and item['name'].strip()
                and '\ufffd' not in item['name'], f'{label}/{ident}: invalid UTF-8 name')
        result[ident] = item
    return result


def verify_tsv(records, path):
    """按运行时字段顺序读取，不能照旧的头注释把 bounds/localY 误作名字。"""
    rows = {}
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.reader(stream, delimiter='\t'):
            if not row or row[0].startswith('#'):
                continue
            require(len(row) == 18, f'{path.name}/{row[0]}: expected 18 tab-separated columns, got {len(row)}')
            require(row[0] not in rows, f'{path.name}: duplicate identifier {row[0]}')
            rows[row[0]] = row
    require(rows.keys() == records.keys(), 'JSON/TSV identifiers differ (missing or stale labels)')
    for ident, item in records.items():
        row = rows[ident]
        require(row[17] == item['name'], f'{ident}: TSV UTF-8 name does not match JSON')
        expected = [*vector(item['position'], 3, ident + '/position'),
                    *vector(item['uv'], 4, ident + '/uv'),
                    *vector(item['size_px'], 2, ident + '/size_px'),
                    item['max_distance_m'], item['priority'],
                    *vector(item['body_bounds_xz'], 4, ident + '/body_bounds_xz'),
                    item['local_label_y']]
        actual = [float(value) for value in row[1:17]]
        require(all(math.isfinite(x) for x in expected + actual), f'{ident}: non-finite numeric field')
        # TSV 保存六位小数；这只允许序列化舍入，不允许位置或 UV 漂移。
        require(all(abs(a - b) <= 0.00000051 for a, b in zip(actual, expected)),
                f'{ident}: TSV numeric columns disagree with JSON (including bounds/localY)')
        require(item['max_distance_m'] > 0 and item['priority'] >= 0,
                f'{ident}: invalid visibility parameters')
        x0, z0, x1, z1 = item['body_bounds_xz']
        require(x0 <= x1 and z0 <= z1, f'{ident}: inverted building bounds')
        require(item['local_label_y'] <= item['position'][1], f'{ident}: local label is above its overview label')


def verify_atlas(records, path):
    raw = path.read_bytes()
    require(len(raw) >= 8, 'RGBA atlas header is truncated')
    width, height = struct.unpack_from('<II', raw)
    require(width > 0 and height > 0 and len(raw) == 8 + width * height * 4,
            f'RGBA atlas dimensions/payload disagree: {width}x{height}, {len(raw)} bytes')
    pixels = memoryview(raw)[8:]
    rectangles = []
    for ident, item in records.items():
        u0, v0, u1, v1 = item['uv']
        require(0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1, f'{ident}: UV rectangle is empty or out of range')
        edges = (u0 * width, v0 * height, u1 * width, v1 * height)
        require(all(abs(v - round(v)) < 0.0001 for v in edges), f'{ident}: UV edges are not aligned to atlas pixels')
        x0, y0, x1, y1 = map(round, edges)
        sx, sy = item['size_px']
        require(sx > 0 and sy > 0 and abs(x1 - x0 - 2 * sx) < 0.0001
                and abs(y1 - y0 - 2 * sy) < 0.0001,
                f'{ident}: display size disagrees with the half-scale atlas crop')
        for other_id, ax0, ay0, ax1, ay1 in rectangles:
            require(not (max(x0, ax0) < min(x1, ax1) and max(y0, ay0) < min(y1, ay1)),
                    f'{ident}: atlas crop overlaps {other_id}')
        rectangles.append((ident, x0, y0, x1, y1))
        opaque = light = 0
        for y in range(y0, y1):
            for offset in range((y * width + x0) * 4, (y * width + x1) * 4, 4):
                r, g, b, a = pixels[offset:offset + 4]
                opaque += a > 0
                # 字色比边框亮，可发现空白 crop；这不是 OCR，不证明文字正确。
                light += a > 0 and min(r, g, b) > 235
        require(opaque > (x1 - x0) * (y1 - y0) * 0.05 and light >= 8,
                f'{ident}: atlas crop has no visible panel or light text pixels')
    return width, height


def distance_to_segment(point, a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dz * dz
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dz) / length2)) if length2 else 0
    return math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dz)


def ring_location(point, ring):
    """返回 -1=外部、0=边界、1=内部；先检查边界再作射线奇偶判断。"""
    inside = False
    x, z = point
    for a, b in zip(ring, ring[1:]):
        if distance_to_segment(point, a, b) < 0.00001:
            return 0
        if (a[1] > z) != (b[1] > z):
            cross_x = a[0] + (z - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < cross_x:
                inside = not inside
    return 1 if inside else -1


class OsmSnapshot:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.nodes = {e.get('id'): ((float(e.get('lon')) - LON0) * METERS_LON,
                                  (LAT0 - float(e.get('lat'))) * METERS_LAT)
                      for e in root.findall('node')}
        self.ways = {e.get('id'): [n.get('ref') for n in e.findall('nd')]
                     for e in root.findall('way')}
        self.elements = {e.tag + '/' + e.get('id'): e for e in root if e.get('id')}

    def rings(self, chains, key):
        # OSM multipolygon 的边界可能由多条反向的 way 拼成，按节点端点连接。
        remaining = [list(chain) for chain in chains]
        result = []
        while remaining:
            ring = remaining.pop(0)
            require(len(ring) >= 2, f'{key}: empty boundary member')
            while ring[0] != ring[-1]:
                for index, chain in enumerate(remaining):
                    if chain[0] == ring[-1] or chain[-1] == ring[-1]:
                        if chain[-1] == ring[-1]:
                            chain.reverse()
                        ring.extend(chain[1:])
                        remaining.pop(index)
                        break
                else:
                    raise ValueError(f'{key}: boundary ways cannot be joined into a closed ring')
            require(len(set(ring)) >= 3, f'{key}: degenerate polygon ring')
            result.append([self.nodes[node] for node in ring])
        return result

    def area(self, key):
        element = self.elements[key]
        if element.tag == 'way':
            return self.rings([self.ways[element.get('id')]], key), []
        require(element.tag == 'relation', f'{key}: expected a polygon way or relation')
        outer, inner = [], []
        for member in element.findall('member'):
            require(member.get('type') == 'way' and member.get('role', '') in ('', 'outer', 'inner'),
                    f'{key}: unsupported multipolygon member; inspect this OSM snapshot')
            (inner if member.get('role') == 'inner' else outer).append(self.ways[member.get('ref')])
        require(outer, f'{key}: no outer boundary')
        return self.rings(outer, key), self.rings(inner, key)

    def covers(self, key, point):
        outer, inner = self.area(key)
        return (any(ring_location(point, ring) >= 0 for ring in outer)
                and not any(ring_location(point, ring) > 0 for ring in inner))

    def bounds(self, key):
        outer, _ = self.area(key)
        points = [point for ring in outer for point in ring]
        return [min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points)]

    def ground_road_segments(self):
        for key, element in self.elements.items():
            if element.tag != 'way':
                continue
            tags = {t.get('k'): t.get('v') for t in element.findall('tag')}
            if (tags.get('highway') not in ('service', 'residential', 'unclassified', 'living_street')
                    or tags.get('area') == 'yes' or tags.get('bridge') not in (None, 'no')
                    or tags.get('tunnel') not in (None, 'no')):
                continue
            points = [self.nodes[n] for n in self.ways[element.get('id')]]
            for a, b in zip(points, points[1:]):
                if a != b:
                    yield key, a, b


def verify_locations(records, config, osm):
    expected = index_records(config['places'], 'place source')
    require({key for key, value in records.items() if value['kind'] != 'gate'} == expected.keys(),
            'Generated non-gate labels differ from the source catalogue')
    mapped = 0
    for ident, item in expected.items():
        label = records[ident]
        point = (label['position'][0], label['position'][2])
        require(label['name'] == item['name'], f'{ident}: generated name differs from the source catalogue')
        key = item.get('osm') or item.get('area')
        if key:
            require(osm.covers(key, point), f'{ident}: label {point} is outside {key} or inside its inner hole')
            mapped += 1
            if item.get('osm'):
                require(all(abs(a - b) < 0.001 for a, b in zip(label['body_bounds_xz'], osm.bounds(key))),
                        f'{ident}: culling bounds disagree with the OSM building footprint')
        elif item.get('node'):
            node = osm.nodes[str(item['node'])]
            require(math.dist(point, node) < 0.001, f'{ident}: label has drifted from its OSM node')
        else:
            require(math.dist(point, item['position_xz_m']) < 0.001,
                    f'{ident}: generated position differs from the source catalogue')
    # 独立绑定语义对象：即使源 JSON 和生成 JSON 一起误改，也不能把湖标到陆地、
    # 把运动广场标到相邻海边足球场，或把海滨小区标到杜鹃山。
    for ident, key in KEY_AREAS.items():
        label = records[ident]
        require(osm.covers(key, (label['position'][0], label['position'][2])),
                f'{ident}: not inside the independently identified OSM area {key}')
    return mapped


def verify_gates(records, gates):
    gate_index = index_records(gates, 'gate manifest')
    require({key for key, value in records.items() if value['kind'] == 'gate'}
            == {'gate-' + key for key in gate_index}, 'Gate labels are missing, duplicated or stale')
    for ident, gate in gate_index.items():
        label = records['gate-' + ident]
        require(label['name'] == gate['name'], f'{ident}: gate label names the wrong gate')
        require(math.dist((label['position'][0], label['position'][2]), gate['center_xz_m']) < 0.001,
                f'{ident}: label does not match the generated gate centre')
        require(math.isfinite(gate['top_y_m']) and label['position'][1] > gate['top_y_m'],
                f'{ident}: gate label is buried below the generated model top')
    return len(gate_index)


def verify_shops(records, manifest, osm):
    roadside = manifest['road_side']
    previous_z = -math.inf
    points = []
    for item in roadside:
        label = records['shop-' + item['id']]
        x, _, z = label['position']
        require(label['kind'] == 'shop' and item['start_z'] <= z <= item['end_z'],
                f"{label['id']}: label is outside its storefront interval from the user manifest")
        require(z > previous_z, 'Roadside shop labels no longer follow the user manifest order')
        previous_z = z
        points.append((x, z))
    require(max(p[0] for p in points) - min(p[0] for p in points) < 2,
            'Roadside shop labels have crossed to a different frontage')
    repair = records['shop-repair']['position']
    sf_item = manifest['across_from_repair']
    sf = records['shop-' + sf_item['id']]['position']
    require(sf_item['start_z'] <= sf[2] <= sf_item['end_z']
            and abs(sf[2] - repair[2]) <= 10 and 3 < abs(sf[0] - repair[0]) < 40,
            'SF label is no longer near and across from the repair shop')
    midpoint = ((repair[0] + sf[0]) * .5, (repair[2] + sf[2]) * .5)
    road, a, b = min(osm.ground_road_segments(), key=lambda segment: distance_to_segment(midpoint, segment[1], segment[2]))
    require(distance_to_segment(midpoint, a, b) < 6, 'Cannot locate the mapped ground road between SF and repair shop')
    def side(p):
        return (b[0] - a[0]) * (p[2] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    require(side(repair) * side(sf) < 0, f'SF and repair labels are on the same side of {road}')
    aliases = {'szu': 'creative'}
    opposite_ids = {'shop-' + aliases.get(item['id'], item['id']) for item in manifest['opposite_qiaoxiang']}
    expected_ids = {'shop-' + item['id'] for item in roadside} | opposite_ids | {'shop-' + sf_item['id']}
    require({key for key, value in records.items() if value['kind'] == 'shop'} == expected_ids,
            'Shop labels differ from the user manifest (including the four shops opposite Qiaoxiang)')
    return len(expected_ids), road


def verify_assets(assets, data):
    records = index_records(read_json(assets / 'campus-places.json'), 'generated places')
    verify_tsv(records, assets / 'campus-places.tsv')
    dimensions = verify_atlas(records, assets / 'campus-place-labels.rgba')
    osm = OsmSnapshot(data / 'yuehai.osm')
    mapped = verify_locations(records, read_json(data / 'campus-places.json'), osm)
    gates = verify_gates(records, read_json(assets / 'campus-gates.json'))
    shops, shop_road = verify_shops(records, read_json(data / 'qiaoyuan-shops.json'), osm)
    return {'places': len(records), 'atlas': dimensions, 'mapped': mapped,
            'gates': gates, 'shops': shops, 'shop_road': shop_road}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    try:
        result = verify_assets(args.assets, args.data)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        print('FAIL: ' + str(error), file=sys.stderr)
        return 1
    print(f"PASS: {result['places']} labels; JSON / 18-column TSV agree, including bounds and localY.")
    print(f"PASS: {result['atlas'][0]}x{result['atlas'][1]} RGBA payload; UV crops fit, do not overlap, and contain visible text pixels.")
    print(f"PASS: {result['mapped']} OSM-bound labels; buildings/bounds, both lakes (including inner hole), south sports plaza and Haibin area.")
    print(f"PASS: {result['gates']} gate labels; {result['shops']} shops follow manifest; SF and repair straddle {result['shop_road']}.")
    print('LIMITS: no field survey, OCR, GUI/occlusion or appearance check; label heights are navigation aids.')
    print('LIMITS: opposite-Qiaoxiang shop order, exact printer location and current businesses remain unverified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
