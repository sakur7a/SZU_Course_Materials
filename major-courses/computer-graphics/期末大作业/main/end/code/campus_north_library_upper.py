"""Generate the north library's estimated U stairs and second-floor deck.

The official plans show the two turning stairs, but provide no dimensions.
All dimensions here are modelling estimates, recorded in the adjacent JSON.
Outputs use the campus world coordinate system and the renderer's SZM1 format.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import mapbox_earcut
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "data" / "north-library-upper.json"
FRAME = HERE / "data" / "north-library-frame.json"
DEFAULT_TERRAIN = HERE / "assets" / "campus-heights.bin"
DEFAULT_OUTPUT = HERE.parents[2] / "build" / "north-library-upper-candidate" / "assets"
PREFIX = "campus-north-library-upper-"


def _parts(shape):
    if shape.geom_type == "Polygon":
        yield shape
    elif hasattr(shape, "geoms"):
        for part in shape.geoms:
            yield from _parts(part)


def _triangles(shape):
    for part in _parts(shape):
        poly = orient(part, sign=1.0)
        rings = [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
        rings = [r for r in rings if len(r) >= 3]
        points = np.asarray([p for ring in rings for p in ring], dtype=np.float64)
        ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
        for tri in points[mapbox_earcut.triangulate_float64(points, ends)].reshape(-1, 3, 2):
            a,b=tri[1]-tri[0],tri[2]-tri[0]
            if abs(a[0]*b[1]-a[1]*b[0]) > 1e-10:
                yield tri


class WorldFrame:
    def __init__(self, frame, base):
        self.x0, self.z0 = frame["center_xz"]
        self.base = float(base)
        angle = math.radians(frame["rotation_y_degrees"])
        self.c, self.s = math.cos(angle), math.sin(angle)

    def point(self, p):
        x, y, z = p
        return (self.x0 + self.c*x + self.s*z, self.base + y,
                self.z0 - self.s*x + self.c*z)

    def vector(self, n):
        x, y, z = n
        return (self.c*x + self.s*z, y, -self.s*x + self.c*z)


class Mesh:
    def __init__(self, frame):
        self.frame, self.faces = frame, []

    def face(self, a, b, c, normal=None):
        a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
        n = np.cross(b-a, c-a) if normal is None else np.asarray(normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            return
        normal_world = self.frame.vector(n/norm)
        self.faces.append([(*self.frame.point(p), *normal_world, p[0]/12, p[2]/12) for p in (a,b,c)])

    def quad(self, a, b, c, d):
        self.face(a,b,c)
        self.face(a,c,d)

    def horizontal(self, shape, y, up=True, conform=False):
        tris=list(_triangles(shape))
        # Earcut can join two collinear hole boundaries with a long edge.
        # Split that edge at every existing boundary vertex, so the closed
        # deck has no T-junctions while both stairs remain exactly mirrored.
        all_points=[]
        if conform:
            all_points=[p for tri in tris for p in tri]
            for part in _parts(shape):
                all_points.extend(part.exterior.coords[:-1])
                for ring in part.interiors: all_points.extend(ring.coords[:-1])
            all_points=np.unique(np.round(np.asarray(all_points,dtype=float),10),axis=0)
        for tri in tris:
            pts=[np.asarray(p,dtype=float) for p in tri]
            a,b=pts[1]-pts[0],pts[2]-pts[0]
            if (a[0]*b[1]-a[1]*b[0]<0) != up:
                pts[1],pts[2]=pts[2],pts[1]
            boundary=[]
            for a,b in zip(pts,pts[1:]+pts[:1]):
                boundary.append(a)
                if conform:
                    delta=b-a; t=((all_points-a)@delta)/(delta@delta)
                    cross=np.abs((all_points[:,0]-a[0])*delta[1]-(all_points[:,1]-a[1])*delta[0])
                    ids=np.flatnonzero((t>1e-8)&(t<1-1e-8)&(cross<1e-8))
                    boundary.extend(all_points[i] for i in ids[np.argsort(t[ids])])
            if len(boundary)==3:
                self.face(*[(x,y,z) for x,z in boundary],normal=(0,1 if up else -1,0))
            else:
                center=np.mean(pts,axis=0)
                for a,b in zip(boundary,boundary[1:]+boundary[:1]):
                    self.face((center[0],y,center[1]),(a[0],y,a[1]),(b[0],y,b[1]),
                              normal=(0,1 if up else -1,0))

    def extrude(self, shape, y0, y1, conform=False):
        self.horizontal(shape, y1, conform=conform)
        self.horizontal(shape, y0, up=False, conform=conform)
        for part in _parts(shape):
            poly = orient(part, sign=1.0)
            for ring in [poly.exterior, *poly.interiors]:
                coords = list(ring.coords)
                for (x0,z0),(x1,z1) in zip(coords,coords[1:]):
                    self.quad((x0,y0,z0),(x0,y1,z0),(x1,y1,z1),(x1,y0,z1))

    def beam(self, a, b, half=0.035):
        a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        d = b-a
        if np.linalg.norm(d) < 1e-6:
            return
        d /= np.linalg.norm(d)
        ref = np.array((0.,1.,0.)) if abs(d[1]) < .95 else np.array((1.,0.,0.))
        u = np.cross(d,ref); u /= np.linalg.norm(u)
        v = np.cross(d,u)
        aa=[a+half*(i*u+j*v) for i,j in ((-1,-1),(1,-1),(1,1),(-1,1))]
        bb=[b+half*(i*u+j*v) for i,j in ((-1,-1),(1,-1),(1,1),(-1,1))]
        for k in range(4):
            q=(k+1)%4
            self.quad(aa[k],aa[q],bb[q],bb[k])
        self.quad(aa[3],aa[2],aa[1],aa[0]); self.quad(bb[0],bb[1],bb[2],bb[3])

    def save(self, path):
        values=np.asarray(self.faces,dtype="<f4").reshape(-1,3,8)
        if not len(values) or len(values)*3 > 500000 or not np.isfinite(values).all():
            raise ValueError(f"Invalid mesh: {path}")
        Path(path).write_bytes(b"SZM1"+struct.pack("<I",len(values)*3)+values.tobytes())
        return len(values)*3


def _grid(f):
    nx,nz,x0,x1,z0,z1=struct.unpack("<IIffff",f.read(24))
    data=np.frombuffer(f.read(4*nx*nz),dtype="<f4")
    if nx<2 or nz<2 or len(data)!=nx*nz:
        raise ValueError("Invalid terrain height grid")
    return nx,nz,x0,x1,z0,z1,data.reshape(nz,nx)


def _grid_at(grid,x,z):
    nx,nz,x0,x1,z0,z1,data=grid
    u=np.clip((x-x0)/(x1-x0),0,1)*(nx-1); v=np.clip((z-z0)/(z1-z0),0,1)*(nz-1)
    i=min(int(u),nx-2); j=min(int(v),nz-2); a=u-i; b=v-j
    p,q,r,s=data[j,i],data[j,i+1],data[j+1,i],data[j+1,i+1]
    return float((1-b)*p+(b-a)*r+a*s if b>=a else (1-a)*p+b*s+(a-b)*q)


def _height_sampler(path):
    path=Path(path)
    if path.suffix.lower()==".mesh":
        data=path.read_bytes()
        if data[:4]!=b"SZM1":
            raise ValueError("Invalid terrain mesh")
        count,=struct.unpack_from("<I",data,4)
        if len(data)!=8+count*32:
            raise ValueError("Truncated terrain mesh")
        tris=np.frombuffer(data,dtype="<f4",offset=8).reshape(-1,3,8)
        mask=((tris[:,:,0].max(axis=1)>-170)&(tris[:,:,0].min(axis=1)<-80)&
              (tris[:,:,2].max(axis=1)>-150)&(tris[:,:,2].min(axis=1)<-55)&
              (tris[:,:,4].min(axis=1)>.7))
        tris=tris[mask].astype(float)
        def sampled(x,z):
            for t in tris:
                if x<t[:,0].min()-1e-5 or x>t[:,0].max()+1e-5 or z<t[:,2].min()-1e-5 or z>t[:,2].max()+1e-5:
                    continue
                a,b,c=t[:,[0,2]]
                den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
                if abs(den)<1e-10: continue
                u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(z-c[1]))/den
                v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(z-c[1]))/den
                if min(u,v,1-u-v)>=-1e-5: return float(u*t[0,1]+v*t[1,1]+(1-u-v)*t[2,1])
            raise ValueError(f"Terrain mesh missing sample: {x}, {z}")
        return sampled
    with path.open("rb") as f:
        coarse=_grid(f)
    patch_path=path.with_name("campus-terrain-patches.bin")
    with patch_path.open("rb") as f:
        magic,count=struct.unpack("<4sI",f.read(8))
        if magic!=b"SZP1" or count>16: raise ValueError("Invalid terrain patches")
        patches=[_grid(f) for _ in range(count)]
    def sampled(x,z):
        for grid in patches:
            if grid[2]<=x<=grid[3] and grid[4]<=z<=grid[5]: return _grid_at(grid,x,z)
        return _grid_at(coarse,x,z)
    return sampled


def _turn(side, radius, turn_x, turn_z, steps=32):
    return Polygon([(side*turn_x,turn_z-radius)] +
                   [(side*(turn_x+radius*math.cos(t)),turn_z+radius*math.sin(t))
                    for t in np.linspace(-math.pi/2,math.pi/2,steps+1)] +
                   [(side*turn_x,turn_z+radius)])


def _sample_line(points, step=.075):
    result=[]
    for a,b in zip(points,points[1:]):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        n=max(1,math.ceil(np.linalg.norm(b-a)/step))
        for i in range(n):
            p=a+(b-a)*(i/n)
            if not result or np.linalg.norm(p-result[-1])>1e-8: result.append(p)
    result.append(np.asarray(points[-1],dtype=float))
    return result


def _routes(frame, surfaces, ground, spec, door, column_shape, deck_shape, glass_shape):
    n=spec["steps_per_run"]; x0=spec["inner_x"]; xt=spec["turn_x"]
    z0=spec["near_run_z"]; z1=spec["far_run_z"]; zt=spec["turn_z"]
    routes={}
    for side,label in ((-1,"left_u_stairs"),(1,"right_u_stairs")):
        curve=[(side*(xt+1.2*math.cos(t)),zt+1.2*math.sin(t))
               for t in np.linspace(math.pi/2,-math.pi/2,25)]
        points=[(0,z0),(side*x0,z0),(side*xt,z0),*curve[1:],(side*x0,z1),(0,z1),(0,22.5)]
        routes[label]=_sample_line(points)
    routes["first_door_to_gray_space"]=_sample_line([(0,22.5),(0,26.2)])
    routes["upper_door_to_terrace"]=_sample_line([(0,22.5),(1.2,23.4),(1.2,37.0),(0,37.0)])
    first,second=.24,4.35
    radius=.30; min_clearance=math.inf; min_column=math.inf; min_glass=math.inf
    min_flowerbed=math.inf; min_door=math.inf
    flowerbed=box(-17.5,39.11,17.5,39.5)
    threshold=LineString([(-door["clear_width"]/2,door["z"]),(door["clear_width"]/2,door["z"])])
    result={}
    for name,points in routes.items():
        previous=first if name!="upper_door_to_terrace" else second
        rows=[]
        for x,z in points:
            p=Point(x,z)
            candidates=[y for poly,y in surfaces if poly.buffer(1e-7).covers(p) and y<=previous+.8]
            wx,_,wz=frame.point((x,0,z)); base_ground=ground(wx,wz)-frame.base
            candidates=[y for y in candidates if y>=base_ground-.05]
            if not candidates: raise ValueError(f"No walking surface on {name}: {(x,z,previous,base_ground)}")
            y=max(candidates)
            if abs(y-previous)>.22: raise ValueError(f"Walking discontinuity on {name}: {previous} -> {y}")
            dcol=p.distance(column_shape); dglass=p.distance(glass_shape)
            min_column=min(min_column,dcol); min_glass=min(min_glass,dglass)
            if dcol<radius or dglass<radius: raise ValueError(f"Body/obstacle collision on {name}: {(x,z,dcol,dglass)}")
            if name=="upper_door_to_terrace":
                min_flowerbed=min(min_flowerbed,p.distance(flowerbed))
                if p.distance(flowerbed)<radius: raise ValueError("Terrace route crosses the flower bed")
            if p.distance(threshold)<radius+.035:
                h=door["first_clear_height"] if abs(y-first)<.001 else door["second_clear_height"]
                gap=h-.035
                min_door=min(min_door,gap); min_clearance=min(min_clearance,gap)
                if gap<1.8: raise ValueError("Door lintel is below head height")
            # A 0.30 m horizontal body envelope catches a nearby slab edge;
            # the slab top is a floor, not a ceiling, when the feet are at 4.35 m.
            if y<second-1e-5 and p.buffer(radius,resolution=8).intersects(deck_shape):
                gap=3.75-y
                min_clearance=min(min_clearance,gap)
                if gap<1.8-1e-6: raise ValueError(f"Slab headroom on {name}: {(x,z,y,gap)}")
            if abs(x)<29.6-radius and z<29.6-radius:
                min_clearance=min(min_clearance,7.95-y)
            rows.append([float(wx),float(frame.base+y),float(wz)])
            previous=y
        result[name]=rows
    return result,{"minimum_sampled_headroom_m":min_clearance,
                   "minimum_column_centerline_clearance_m":min_column,
                   "minimum_glass_centerline_clearance_m":min_glass,
                   "minimum_door_headroom_m":min_door,
                   "minimum_terrace_to_flowerbed_m":min_flowerbed,
                   "body_radius_m":radius,"max_step_m":(second-first)/(2*n),
                   "two_directions_checked":True}


def make_assets(output_assets, terrain_path=None, height_at=None):
    """Write the five meshes and route files; height_at returns world terrain y.

    terrain_path can be a campus-heights.bin (with its neighbouring patch file)
    or the rendered campus-terrain.mesh.  The source assets are never modified.
    """
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")); frame_cfg=json.loads(FRAME.read_text(encoding="utf-8"))
    ground=height_at or _height_sampler(terrain_path or DEFAULT_TERRAIN)
    frame=WorldFrame(frame_cfg,ground(*frame_cfg["center_xz"]))
    s=cfg["stair"]; x0=s["inner_x"]; xt=s["turn_x"]; zt=s["turn_z"]
    r=s["turn_radius"]; w=s["run_width"]; n=s["steps_per_run"]
    first=cfg["first_floor_level"]; bottom=cfg["second_floor_bottom"]; top=cfg["second_floor_top"]
    half=(first+top)/2; rise=(top-first)/(2*n); dx=(xt-x0)/n
    meshes={k:Mesh(frame) for k in ("deck","stairs","rails","frames","walk")}
    deck,stairs,rails,frames,walk=(meshes[k] for k in ("deck","stairs","rails","frames","walk"))
    holes=[]; bodies=[]; surface_tops=[]
    for side in (-1,1):
        turn=_turn(side,r,xt,zt)
        margin=s["opening_side_allowance"]
        opening_run=box(min(side*x0,side*xt),s["far_run_z"]-w/2-margin,
                        max(side*x0,side*xt),s["near_run_z"]+w/2+margin)
        hole=opening_run.union(_turn(side,r+margin,xt,zt))
        holes.append(hole); bodies.append(turn)
        stairs.extrude(turn,first,half); surface_tops.append((turn,half))
        for k in range(n):
            a=x0+k*dx; b=x0+(k+1)*dx
            low=box(min(side*a,side*b),s["near_run_z"]-w/2,max(side*a,side*b),s["near_run_z"]+w/2)
            high=box(min(side*a,side*b),s["far_run_z"]-w/2,max(side*a,side*b),s["far_run_z"]+w/2)
            low_y=first+(k+1)*rise; high_y=top-k*rise
            stairs.extrude(low,first,low_y); stairs.extrude(high,first,high_y)
            bodies.extend((low,high)); surface_tops.extend(((low,low_y),(high,high_y)))
        # Handrails stay on the flight edges; posts terminate on the treads.
        for z,high in ((s["near_run_z"]+w/2,False),(s["near_run_z"]-w/2,False),
                       (s["far_run_z"]-w/2,True),(s["far_run_z"]+w/2,True)):
            pts=[]
            for k in range(n):
                x=side*(x0+(k+.5)*dx); y=(top-k*rise) if high else (first+(k+1)*rise)
                rails.beam((x,y,z),(x,y+s["handrail_height"],z),.025)
                pts.append((x,y+s["handrail_height"],z))
            edge_start=(side*x0,pts[0][1],z)
            edge_end=(side*xt,half+s["handrail_height"],z)
            line=[edge_start,*pts,edge_end]
            for a,b in zip(line,line[1:]): rails.beam(a,b,.035)
        arc=[(side*(xt+r*math.cos(t)),half+s["handrail_height"],zt+r*math.sin(t))
             for t in np.linspace(-math.pi/2,math.pi/2,33)]
        for a,b in zip(arc,arc[1:]): rails.beam(a,b,.035)
        for p in arc[::4]: rails.beam((p[0],half,p[2]),p,.025)
    original_deck=box(-30.25,-29.7,30.25,29.7).union(box(-17.5,26.5,17.5,39.5))
    deck_shape=original_deck.difference(unary_union(holes))
    if not deck_shape.is_valid: raise ValueError("Invalid upper deck cutouts")
    deck.extrude(deck_shape,bottom,top,conform=True)
    columns=unary_union([box(x-cfg["column_width"]/2,cfg["column_center_z"]-cfg["column_width"]/2,
                             x+cfg["column_width"]/2,cfg["column_center_z"]+cfg["column_width"]/2)
                         for x in cfg["column_centers_x"]])
    # Walk mesh contains exposed tops only. Do not mark the enclosed rooms,
    # stairs' solid footprints, columns or the front flower bed as paths.
    apron=box(-30.2,22.5,30.2,30.2).difference(columns.union(unary_union(bodies)))
    upper=deck_shape.intersection(box(-30.25,22.5,30.25,39.5)).difference(
        columns.union(box(-17.5,39.11,17.5,39.5)))
    walk.horizontal(apron,first); walk.horizontal(upper,top)
    for poly,y in surface_tops: walk.horizontal(poly,y)
    surfaces=[(apron,first),(upper,top),*surface_tops]
    d=cfg["door"]; dz=d["z"]; halfw=d["clear_width"]/2
    for level,h in ((first,d["first_clear_height"]),(top,d["second_clear_height"])):
        frames.beam((-halfw,level,dz),(-halfw,level+h,dz),.035)
        frames.beam((halfw,level,dz),(halfw,level+h,dz),.035)
        frames.beam((-halfw,level+h,dz),(halfw,level+h,dz),.035)
    glass=unary_union([LineString([(-24,dz),(-halfw,dz)]),LineString([(halfw,dz),(24,dz)])]).buffer(.08)
    routes,qa=_routes(frame,surfaces,ground,s,d,columns,deck_shape,glass)
    # The same walking selection must work in both directions, not just uphill.
    for name,rows in routes.items():
        prev=rows[-1][1]-frame.base
        for wx,wy,wz in reversed(rows):
            x=frame.c*(wx-frame.x0)-frame.s*(wz-frame.z0)
            z=frame.s*(wx-frame.x0)+frame.c*(wz-frame.z0)
            p=Point(x,z); available=[y for poly,y in surfaces if poly.buffer(1e-7).covers(p) and y<=prev+.8]
            if not available or abs(max(available)-(wy-frame.base))>.005:
                raise ValueError(f"Reverse walk differs on {name}: {(x,z,prev,wy-frame.base,available)}")
            prev=max(available)
    for hole in holes:
        if deck_shape.intersection(hole).area>1e-8: raise ValueError("Deck intersects a stair opening")
    if abs(2*n*rise-(top-first))>1e-8: raise ValueError("Stair flight does not meet the second-floor height")
    output=Path(output_assets); output.mkdir(parents=True,exist_ok=True)
    counts={name:mesh.save(output/(PREFIX+name+".mesh")) for name,mesh in meshes.items()}
    document={"description":"Estimated geometry; world x, foot y, z. Routes sampled in both directions.",
              "base_y":frame.base,"frame":frame_cfg,"validation":qa,"routes":routes}
    (output/(PREFIX+"routes.json")).write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tsv=[f"P\t{name}\t{x:.5f}\t{y:.5f}\t{z:.5f}" for name,rows in routes.items() for x,y,z in rows]
    (output/(PREFIX+"routes.tsv")).write_text("\n".join(tsv)+"\n",encoding="utf-8")
    return {"vertices":counts,"route_samples":{k:len(v) for k,v in routes.items()},"validation":qa,"base_y":frame.base}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--terrain",type=Path,default=DEFAULT_TERRAIN)
    args=parser.parse_args()
    print(json.dumps(make_assets(args.output,terrain_path=args.terrain),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
