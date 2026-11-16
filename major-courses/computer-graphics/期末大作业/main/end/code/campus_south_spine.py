"""Ground-level teaching courtyard and the recessed ZhiYi podium gallery."""
import argparse
import json
import math
from pathlib import Path

from shapely import get_parts
from shapely.geometry import LineString, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from campus_lide_corridor import edit_surface, read_triangles
from campus_planting import _mesh_footprint


def podium_plan(geom, config):
    opening = config['building_opening_requests'][0]
    frontage = LineString(opening['boundary_polyline_xz_m'])
    strip = geom.intersection(frontage.buffer(opening['inset_estimate_m'],
                                              cap_style='flat', join_style='mitre'))
    return frontage, strip


def make_podium(shell, roof, gallery, *, geom, base, height, config, passage, wall, beam):
    """Open the two courtyard-facing bays, keeping the upper mass and passage."""
    opening = config['building_opening_requests'][0]
    frontage, strip = podium_plan(geom, config)
    deck_y = opening['existing_passage_deck_y_m']
    underside = max(base + opening['first_slab_top_above_base_estimate_m'] - .24,
                    deck_y + opening['clear_height_estimate_m'])
    # The existing through-route keeps its two wall openings at the bridge deck.
    path = ((passage, deck_y),)
    wall(shell, roof, geom, underside, base+height, (), leveled_paths=path)
    wall(shell, roof, geom.difference(strip), base-2, underside, (),
         cap=False, leveled_paths=path)
    cap = type(gallery)()
    cap.horizontal(strip, underside)
    for index in range(0, len(cap.vertices), 24):
        for corner in (0, 16, 8):
            x, y, z, _, _, _, u, v = cap.vertices[index+corner:index+corner+8]
            gallery.v((x,y,z),(0,-1,0),(u,v))

    # Keep the supporting row inside the covered footprint and clear of the bridge route.
    blocked = passage.buffer(2.05)
    for a, b in zip(frontage.coords, list(frontage.coords)[1:]):
        line = LineString((a,b))
        dx, dz = (b[0]-a[0])/line.length, (b[1]-a[1])/line.length
        spans = max(1, round(line.length/6.2))
        for index in range(spans):
            p = line.interpolate(line.length*(index+.5)/spans)
            for sign in (-1,1):
                x,z = p.x - dz*sign*.78, p.y + dx*sign*.78
                column = box(x-.24,z-.24,x+.24,z+.24)
                if strip.covers(column) and not blocked.intersects(column):
                    beam(gallery,(x,base-.65,z),(x,underside,z),.24)
                    break
    return strip


def make_spine(*, config, polygon, line, vehicle, height_at, mesh_type, terrain, assets):
    """Drape the paths and lawns on the exact render triangles and give lawn edges depth."""
    court = polygon(config['work_area_osm'])
    buildings = unary_union([polygon(key) for key in config['building_exclusions']])
    opening = config['building_opening_requests'][0]
    _, gallery = podium_plan(polygon(opening['osm']), config)
    routes = unary_union([line(key) for key in config['relative_lower_paths']])
    crossings = routes.buffer(1.12, cap_style='flat', join_style='mitre')
    # The layer=-1 paths are absent from automatic paving. Connect their approaches
    # around the courtyard, while keeping any portion inside the building shell clear.
    approach = crossings.intersection(court.buffer(18))
    work = (court.union(approach).difference(buildings.buffer(config['building_clearance_m'], join_style='mitre'))
            .union(gallery.buffer(.19, join_style='mitre')).difference(vehicle.buffer(.18)))
    # Existing bridge posts are 8-triangle beams. Cut out their bases before creating
    # a walkable surface, retaining the existing bridge and its clear spans.
    _, supports = read_triangles(terrain.parent/'campus-bridge-support.mesh')
    bases = []
    for post in supports.reshape(-1,24,8):
        x0,_,z0 = post[:, :3].min(axis=0)
        x1,_,z1 = post[:, :3].max(axis=0)
        if x1-x0 <= 2 and z1-z0 <= 2:
            base = box(x0-.16,z0-.16,x1+.16,z1+.16)
            if work.intersects(base):
                bases.append(base)
    work = work.difference(unary_union(bases))
    lawns = unary_union([polygon(key) for key in config['grass_osm']]).intersection(work).difference(crossings)
    width = config['edge_width_estimate_m']
    edge = lawns.boundary.buffer(width/2, join_style='mitre').intersection(work).difference(crossings)
    grass = lawns.difference(edge)
    mapped_paving = _mesh_footprint(terrain.parent/'campus-network-paths.mesh', work.envelope)
    paving = work.difference(lawns.union(edge)).difference(mapped_paving)
    paths = {role:assets/f'campus-south-spine-{role}.mesh' for role in ('paving','grass','edge')}
    for role, shape, lift, uv in (('paving',paving,config['paving_offset_m'],1.6),
                                  ('grass',grass,config['grass_offset_m'],3.1),
                                  ('edge',edge,config['edge_height_estimate_m'],1.6)):
        edit_surface(terrain, paths[role], shape, drape=(lift,uv))

    # The top follows the terrain triangles exactly. Add the visible inner and outer curb walls.
    mesh = mesh_type()
    _, top = read_triangles(paths['edge'])
    mesh.vertices.extend(top.ravel().tolist())
    lip = config['edge_height_estimate_m']
    for part in get_parts(edge):
        if part.geom_type != 'Polygon':
            continue
        part = orient(part, sign=1)
        for ring in (part.exterior, *part.interiors):
            points = list(ring.coords)
            for a,b in zip(points, points[1:]):
                length = math.dist(a,b)
                if length < .005:
                    continue
                nx,nz = (b[1]-a[1])/length, (a[0]-b[0])/length
                count = max(1, math.ceil(length/.8))
                for i in range(count):
                    p,q = (tuple(a[k]+(b[k]-a[k])*j/count for k in range(2)) for j in (i,i+1))
                    py,qy = height_at(*p),height_at(*q)
                    mesh.quad((p[0],py-.02,p[1]),(q[0],qy-.02,q[1]),
                              (q[0],qy+lip,q[1]),(p[0],py+lip,p[1]),(nx,0,nz),
                              ((0,0),(1,0),(1,1),(0,1)))
    mesh.save(paths['edge'])

    # Reuse the existing stone and grass textures. The W entry shares the actual paving mesh.
    index = [
        'L\tsouth-spine\tstone\tcampus-south-spine-paving.mesh\tcampus-sidewalk.ppm\t1\t1\t.99',
        'L\tsouth-spine\tstone\tcampus-south-spine-grass.mesh\tgrass.ppm\t1.03\t1.13\t.89',
        'L\tsouth-spine\tstone\tcampus-south-spine-edge.mesh\tconcrete.ppm\t.90\t.90\t.84',
        'L\tsouth-spine\tstone\tcampus-zhiyi-gallery.mesh\tconcrete.ppm\t.92\t.92\t.89',
        'W\tsouth-spine\tcampus-south-spine-paving.mesh',
    ]
    (assets/'campus-south-spine.tsv').write_text('\n'.join(index)+'\n', encoding='utf-8')
    return {'paving_m2':round(paving.area,1), 'lawn_m2':round(grass.area,1),
            'curb_m2':round(edge.area,1), 'covered_gallery_m2':round(gallery.area,1)}


if __name__ == '__main__':
    import generate_campus as campus
    from verify_gate_assets import Terrain

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=campus.ASSETS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    terrain = Terrain(args.assets)
    print(campus.make_south_courtyard(assets=args.output, terrain=args.assets/'campus-terrain.mesh',
                                     height_at=terrain))
    shell, roof, gallery = (campus.MeshWriter() for _ in range(3))
    geom = campus.way_polygon(campus.ways['776358621'])
    p = geom.representative_point()
    campus.make_zhiyi_podium(shell, roof, gallery, geom, terrain(p.x,p.y),
                           campus.HEIGHTS['way/776358621'][0])
    shell.save(args.output/'campus-zhiyi-podium.mesh')
    gallery.save(args.output/'campus-zhiyi-gallery.mesh')
