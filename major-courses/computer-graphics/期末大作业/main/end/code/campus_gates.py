"""统一索引各校门的独立模型与材质，保持生成器和渲染器采用同一套资源。"""
import json
import struct

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

import campus_gates_west
import campus_gates_south
import campus_gates_north


def excluded_buildings(data):
    return {key for file in ('gates-west.json','gates-south.json','gates-north.json')
            for gate in json.loads((data/file).read_text(encoding='utf8'))['gates']
            for key in gate.get('exclude_osm_buildings',())}


def make_gates(**kwargs):
    gates=[]
    for module in (campus_gates_west,campus_gates_south,campus_gates_north):
        gates.extend(module.make_gates(**kwargs)['gates'])
    order=['liye','zhiyuan','zhixin','zhihe','zhicheng','zhiheng','lide','lizhi','ligong','lixin','liyan']
    gates.sort(key=lambda gate:order.index(gate['id']))
    assets=kwargs['assets'];height_at=kwargs['height_at'];reserved=[]
    lines=['# G: gate view; L: independent material layer; W: walkable surface']
    for gate in gates:
        coordinates=[]
        for layer in gate['layers']:
            raw=(assets/layer['mesh']).read_bytes();count,=struct.unpack_from('<I',raw,4)
            vertices=np.frombuffer(raw,dtype='<f4',offset=8,count=count*8).reshape(-1,8)
            coordinates.append(vertices[:,:3])
        vertices=np.vstack(coordinates)
        low=vertices.min(axis=0);high=vertices.max(axis=0)
        gate['bounds_xyz_m']=[low.astype(float).tolist(),high.astype(float).tolist()]
        gate['top_y_m']=float(high[1])
        footprint=box(float(low[0]),float(low[2]),float(high[0]),float(high[2])).buffer(2)
        reserved.append(footprint)
        gate.pop('vegetation_keep_clear',None)
        eye=gate['view_eye'];eye[1]=max(eye[1],height_at(eye[0],eye[2])+1.8)
        row=['G',gate['id'],gate['name'],*eye,*gate['view_target'],*gate['center_xz_m']]
        lines.append('\t'.join(str(v) for v in row))
        for layer in gate['layers']:
            texture=gate['sign_texture'] if layer['material']=='sign' else '-'
            row=['L',gate['id'],layer['material'],layer['mesh'],texture,*layer['color']]
            lines.append('\t'.join(str(v) for v in row))
        if gate.get('walking_mesh'):
            lines.append('\t'.join(('W',gate['id'],gate['walking_mesh'])))
    (assets/'campus-gates.tsv').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (assets/'campus-gates.json').write_text(json.dumps(gates,ensure_ascii=False,indent=2),encoding='utf8')
    return gates,unary_union(reserved)
