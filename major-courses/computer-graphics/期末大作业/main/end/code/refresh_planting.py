"""只用已生成的道路与地形资产刷新标注植被，不重做全校园。"""
import generate_campus as campus
import json
from verify_gate_assets import Terrain

result = campus.make_annotated_planting(height_at=Terrain(campus.ASSETS))
metadata=campus.ASSETS/'campus-data.json'
if metadata.exists():
    data=json.loads(metadata.read_text(encoding='utf8'))
    data['annotated_planting']={'shrubs':len(result['shrub_points']),
                                'tree_instances':result['tree_instances'],
                                'rejected':len(result['rejected'])}
    metadata.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
print(f"Planting generated: {len(result['shrub_points'])} illustrative shrubs, "
      f"{result['tree_instances']} annotated trees, {len(result['rejected'])} warnings.")
