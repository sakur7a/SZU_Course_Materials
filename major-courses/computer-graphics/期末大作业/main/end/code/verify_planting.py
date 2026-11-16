"""核验场景绿化的避让，并用隔离夹具验证人工树点的接收与拒绝。"""
import json
import math
import struct
import tempfile
from pathlib import Path

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import campus_planting as planting
import generate_campus as campus
from verify_gate_assets import Terrain


def actual_sample():
    config=json.loads((campus.DATA/'planting-annotations.json').read_text(encoding='utf8'))
    report=json.loads((campus.ASSETS/'campus-planting-report.json').read_text(encoding='utf8'))
    zones={z['id']:z for z in config['regions']}
    geometries=[Polygon(z['polygon_xz_m']) for z in config['regions']]
    study=unary_union(geometries).buffer(2).envelope
    walks=unary_union([planting._mesh_footprint(campus.ASSETS/name,study) for name in (
        'campus-network-sidewalk.mesh','campus-network-paths.mesh',
        'campus-lide-walk.mesh','campus-lide-walk-edge.mesh')])
    booth=Polygon(campus.LIDE_WALK['gate_booth_footprint_m'])
    bridges=unary_union([g.buffer((7.5 if key=='way/574956548' else
                                   3.5 if key=='way/776358616' else 2.0)+5.8)
                          for key,t,g in campus.roadways if t.get('bridge') not in (None,'no')])
    for item in report['shrub_points']:
        zone=zones[item['id']]
        footprint=Point(item['x'],item['z']).buffer(item['radius'])
        if not Polygon(zone['polygon_xz_m']).covers(footprint): raise ValueError('Shrub outside annotated zone')
        if not campus.campus.covers(footprint): raise ValueError('Shrub outside campus')
        if walks.buffer(.30).intersects(footprint): raise ValueError('Shrub overlaps a path')
        if campus.vehicle_surface.buffer(.30).intersects(footprint): raise ValueError('Shrub overlaps a road')
        if booth.buffer(.30).intersects(footprint): raise ValueError('Shrub overlaps the gate booth')
        if bridges.buffer(.30).intersects(footprint): raise ValueError('Shrub overlaps a bridge clearance')
        if any(g.buffer(.30).intersects(footprint) for _,_,g in campus.buildings if g.intersects(study)):
            raise ValueError('Shrub overlaps a building')
        if item['confidence'] != zone['confidence'] or not item['source']:
            raise ValueError('Shrub provenance or confidence missing')
    tree_data=(campus.ASSETS/'campus-annotated-trees.bin').read_bytes()
    count,=struct.unpack_from('<I',tree_data)
    if len(tree_data)!=4+count*16 or count!=report['tree_instances']:
        raise ValueError('Annotated tree instance file does not match report')
    raw=(campus.ASSETS/'campus-annotated-shrubs.mesh').read_bytes()
    if raw[:4]!=b'SZM1': raise ValueError('Invalid annotated shrub mesh')
    vertex_count,=struct.unpack_from('<I',raw,4)
    if len(raw)!=8+vertex_count*32 or vertex_count!=len(report['shrub_points'])*18:
        raise ValueError('Annotated shrub mesh does not match report')
    vertices=list(struct.iter_unpack('<8f',raw[8:]))
    terrain=Terrain(campus.ASSETS)
    for i in range(0,len(vertices),18):
        for offset in (0,1,3,6,7,9,12,13,15):
            x,y,z=vertices[i+offset][:3]
            if abs(y-terrain(x,z)-.03)>.003:
                raise ValueError('Shrub base does not follow rendered terrain')
    print(f"Actual planting verified: {len(report['shrub_points'])} shrubs; {count} tree points.")


def synthetic_tree_cases():
    # 这是单独的虚构夹具，不写入校园项目或冒充实景树位。
    # 将夹具限制在源码目录内；删除前检查临时路径没有越出该目录。
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='planting-check-',dir=parent) as temp:
        assets=Path(temp)
        if not assets.resolve().is_relative_to(parent):
            raise ValueError('Temporary fixture escaped the project directory')
        for name in ('campus-network-sidewalk.mesh','campus-network-paths.mesh',
                     'campus-lide-walk.mesh','campus-lide-walk-edge.mesh'):
            mesh=campus.MeshWriter()
            mesh.quad((2,0,-5),(4,0,-5),(4,0,5),(2,0,5),(0,1,0),((0,0),(1,0),(1,1),(0,1)))
            mesh.save(assets/name)
        (assets/'campus-gates.json').write_text('[]',encoding='utf8')
        (assets/'campus-trees.bin').write_bytes(struct.pack('<I4f',1,15,0,15,1))
        base={'asset':'existing_broadleaf','confidence':'illustrative','source':'synthetic test','clearance_m':1}
        config={'regions':[{'id':'test','kind':'shrub_cluster','confidence':'illustrative','source':'synthetic test',
                            'polygon_xz_m':[[-8,-4],[-3,-4],[-3,4],[-8,4]],'count':4,'seed':4,
                            'min_spacing_m':.8,'radius_m':[.25,.3],'height_m':[.5,.6]}],
                'tree_points':[dict(base,id='valid',center_xz_m=[0,10]),
                               dict(base,id='road',center_xz_m=[7,0]),
                               dict(base,id='existing',center_xz_m=[15,15]),
                               dict(base,id='bridge',center_xz_m=[-10,10])]}
        args=dict(config=config,assets=assets,campus=box(-30,-30,30,30),vehicle_surface=box(6,-5,9,5),
                  buildings=[],waters=[],pitches=[],booth=box(-12,-12,-10,-10),
                  bridge_clearance=box(-11,9,-9,11),height_at=lambda x,z:0,mesh_type=campus.MeshWriter)
        one=planting.generate(**args)
        first=(assets/'campus-annotated-shrubs.mesh').read_bytes()
        two=planting.generate(**args)
        if one['accepted_tree_ids']!=['valid'] or {r['id'] for r in one['rejected']}!={'road','existing','bridge'}:
            raise ValueError('Annotated tree acceptance and rejection failed')
        if one!=two or first!=(assets/'campus-annotated-shrubs.mesh').read_bytes():
            raise ValueError('Planting generation is not deterministic')
        args['config']={'regions':[],'tree_points':[]}
        empty=planting.generate(**args)
        if empty['shrub_vertices']!=0 or (assets/'campus-annotated-shrubs.mesh').read_bytes()!=b'SZM1\0\0\0\0':
            raise ValueError('An empty planting layer should be valid')
    print('Synthetic tree acceptance, road/tree/bridge rejection, repeatability and empty layer verified.')


if __name__=='__main__':
    actual_sample()
    synthetic_tree_cases()
