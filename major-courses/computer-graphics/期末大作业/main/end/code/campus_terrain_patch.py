"""在不改变全校粗网格的情况下，细化个别道路与场馆的地形接缝。"""
import math
import struct

import numpy as np
from shapely.geometry import Point
from shapely.ops import substring, unary_union
from shapely.prepared import prep


class TerrainPatch:
    def __init__(self, values, bounds, spacing):
        self.values=values
        self.x0,self.x1,self.z0,self.z1=bounds
        self.spacing=spacing
        self.nz,self.nx=values.shape
        self.normals=None

    def contains(self,x,z):
        return self.x0<=x<=self.x1 and self.z0<=z<=self.z1

    def sample(self,values,x,z):
        u=np.clip((x-self.x0)/self.spacing,0,self.nx-1)
        v=np.clip((z-self.z0)/self.spacing,0,self.nz-1)
        i,j=min(int(u),self.nx-2),min(int(v),self.nz-2)
        a,b=u-i,v-j
        p,q,r,s=values[j,i],values[j,i+1],values[j+1,i],values[j+1,i+1]
        return (1-b)*p+(b-a)*r+a*s if b>=a else (1-a)*p+b*s+(a-b)*q

    def height(self,x,z):
        return float(self.sample(self.values,x,z))

    def make_normals(self,coarse_normal):
        # 接缝边界沿用粗地形法线，避免同一地面出现明显的光照缝。
        gz,gx=np.gradient(self.values,self.spacing,self.spacing)
        normal=np.stack((-gx,np.ones_like(gx),-gz),axis=2)
        normal/=np.linalg.norm(normal,axis=2,keepdims=True)
        for j in range(self.nz):
            for i in range(self.nx):
                if i in (0,self.nx-1) or j in (0,self.nz-1):
                    normal[j,i]=coarse_normal(self.x0+i*self.spacing,self.z0+j*self.spacing)
        self.normals=normal.astype(np.float32)


def smooth_road_patch(coarse, *, coarse_bounds, bounds, spacing, line, center,
                      half_length, target_grade, protected, additional_routes=(),
                      lock_protected_profile=False, keep_endpoint_fit=False,
                      fit_spacing=None, core_weight=5.0, edge_weight=2.0,
                      landscape_smoothing_steps=0):
    """保持场馆和补丁边界不变，拟合道路及两侧；高程仅供视觉展示。"""
    x0,x1,z0,z1=bounds
    X0,X1,Z0,Z1=coarse_bounds
    coarse_nz,coarse_nx=coarse.shape
    coarse_step_x=(X1-X0)/(coarse_nx-1)
    coarse_step_z=(Z1-Z0)/(coarse_nz-1)
    if (abs(coarse_step_x/spacing-round(coarse_step_x/spacing))>1e-6
            or abs(coarse_step_z/spacing-round(coarse_step_z/spacing))>1e-6
            or any(abs((v-o)/s-round((v-o)/s))>1e-6 for v,o,s in
                   ((x0,X0,coarse_step_x),(x1,X0,coarse_step_x),
                    (z0,Z0,coarse_step_z),(z1,Z0,coarse_step_z)))):
        raise ValueError('A detail patch must align with the coarse terrain cells')

    def coarse_height(x,z):
        u=np.clip((x-X0)/coarse_step_x,0,coarse_nx-1)
        v=np.clip((z-Z0)/coarse_step_z,0,coarse_nz-1)
        i,j=min(int(u),coarse_nx-2),min(int(v),coarse_nz-2)
        a,b=u-i,v-j
        p,q,r,s=coarse[j,i],coarse[j,i+1],coarse[j+1,i],coarse[j+1,i+1]
        return (1-b)*p+(b-a)*r+a*s if b>=a else (1-a)*p+b*s+(a-b)*q

    xs=np.arange(x0,x1+spacing*.1,spacing)
    zs=np.arange(z0,z1+spacing*.1,spacing)
    values=np.asarray([[coarse_height(x,z) for x in xs] for z in zs],dtype=np.float64)
    patch=TerrainPatch(values,bounds,spacing)
    locked=prep(protected.buffer(.75))
    requests=[{'line':line,'center':center,'half_length':half_length,'target_grade':target_grade,
               'lock_protected_profile':lock_protected_profile,'keep_endpoint_fit':keep_endpoint_fit,
               'fit_spacing':fit_spacing,'core_weight':core_weight,'edge_weight':edge_weight}]
    requests.extend(additional_routes)
    profiles=[]
    for request in requests:
        path=request['line']
        midpoint=path.project(Point(*request['center']))
        start,end=max(0,midpoint-request['half_length']),min(path.length,midpoint+request['half_length'])
        distances=np.linspace(start,end,max(3,int(round((end-start)/2.5))+1))
        positions=[path.interpolate(d) for d in distances]
        original=np.asarray([patch.height(p.x,p.y) for p in positions])
        second=np.diff(np.eye(len(distances)),n=2,axis=0)
        weights=np.ones(len(distances));weights[:3]=weights[-3:]=15.0
        fixed=np.asarray([locked.covers(p) for p in positions]) if request.get('lock_protected_profile') else np.zeros(len(distances),dtype=bool)
        weights[fixed]=1000.0
        profile=np.linalg.solve(np.diag(weights)+55*(second.T@second),weights*original)
        for _ in range(1400):
            difference=np.diff(profile)
            excess=np.sign(difference)*np.maximum(0,np.abs(difference)-request['target_grade']*np.diff(distances))*.48
            profile[:-1]+=excess;profile[1:]-=excess
            profile[0],profile[-1]=original[0],original[-1]
            profile[fixed]=original[fixed]
        profiles.append({'request':request,'start':start,'end':end,
                         'distances':distances,'original':original,'profile':profile})

    zone=prep(unary_union([substring(p['request']['line'],max(0,p['start']-15),
                                     min(p['request']['line'].length,p['end']+15)).buffer(20)
                           for p in profiles]))
    lookup={}
    for j,z in enumerate(zs):
        for i,x in enumerate(xs):
            p=Point(x,z)
            # 两层边缘网格固定，使细分区域与粗网格无缝相接。
            if 1<i<patch.nx-2 and 1<j<patch.nz-2 and zone.covers(p) and not locked.covers(p):
                lookup[j,i]=len(lookup)
    if not lookup:
        raise ValueError('No adjustable detail grid points near the road')

    row_ids,col_ids,coefficients,rhs=[],[],[],[]
    def add_row(pairs,value):
        index=len(rhs);rhs.append(value)
        for col,coefficient in pairs:
            if abs(coefficient)>1e-12:
                row_ids.append(index);col_ids.append(col);coefficients.append(coefficient)

    def triangle_weights(x,z):
        u=np.clip((x-x0)/spacing,0,patch.nx-1);v=np.clip((z-z0)/spacing,0,patch.nz-1)
        i,j=min(int(u),patch.nx-2),min(int(v),patch.nz-2)
        a,b=u-i,v-j
        if b>=a:
            return [((j,i),1-b),((j+1,i),b-a),((j+1,i+1),a)]
        return [((j,i),1-a),((j,i+1),a-b),((j+1,i+1),b)]

    for item in profiles:
        request=item['request']
        path=request['line']
        fit_step=request.get('fit_spacing')
        # 道路经过多个细分三角面时，加密拟合约束以减少很短距离内的局部尖折。
        fit_distances=(np.linspace(item['start'],item['end'],
                                   max(3,int((item['end']-item['start'])/fit_step)+1))
                       if fit_step else item['distances'])
        fit_levels=np.interp(fit_distances,item['distances'],item['profile'])
        for d,level in zip(fit_distances,fit_levels):
            p=path.interpolate(float(d))
            a=path.interpolate(max(0,d-.5));b=path.interpolate(min(path.length,d+.5))
            dx,dz=b.x-a.x,b.y-a.y
            length=math.hypot(dx,dz)
            if length<.05:
                continue
            tx,tz=-dz/length,dx/length
            end_fade=1.0 if request.get('keep_endpoint_fit') else (item['end']-d)/10
            fade=max(0,min(1,(d-item['start'])/10,end_fade))
            for offset in (-5,-3,0,3,5):
                x,z=p.x+offset*tx,p.y+offset*tz
                if locked.covers(Point(x,z)):
                    continue
                pairs=[(lookup[node],coefficient) for node,coefficient in triangle_weights(x,z) if node in lookup]
                if not pairs:
                    continue
                weight=request.get('edge_weight',2.0) if abs(offset)>3 else request.get('core_weight',5.0)
                add_row([(i,c*weight) for i,c in pairs],(level-patch.height(x,z))*fade*weight)
    for node,index in lookup.items():
        add_row([(index,.25)],0.0)
        for neighbor in ((node[0]+1,node[1]),(node[0],node[1]+1)):
            pairs=[(index,1.5)]
            if neighbor in lookup:
                pairs.append((lookup[neighbor],-1.5))
            add_row(pairs,0.0)

    # 用稀疏行列索引实现共轭梯度，不依赖额外的 SciPy 运行环境。
    rows=np.asarray(row_ids,dtype=np.int32);cols=np.asarray(col_ids,dtype=np.int32)
    coeff=np.asarray(coefficients);rhs=np.asarray(rhs)
    count=len(lookup)
    def transposed(value):
        return np.bincount(cols,weights=coeff*value[rows],minlength=count)
    def multiply(value):
        return transposed(np.bincount(rows,weights=coeff*value[cols],minlength=len(rhs)))
    right=transposed(rhs)
    diagonal=np.bincount(cols,weights=coeff*coeff,minlength=count)
    delta=np.zeros(count);residual=right.copy();z=residual/diagonal;direction=z.copy();rz=residual@z
    threshold=1e-6*max(float(np.linalg.norm(right)),1.0)
    for _ in range(1200):
        q=multiply(direction);denominator=direction@q
        if denominator<1e-30 or np.linalg.norm(residual)<threshold:
            break
        step=rz/denominator
        delta+=step*direction;residual-=step*q
        z=residual/diagonal;new_rz=residual@z
        direction=z+(new_rz/rz)*direction;rz=new_rz
    if np.linalg.norm(residual)>=threshold*2:
        raise ValueError('Road detail fit did not converge')
    for node,index in lookup.items():
        values[node]+=delta[index]
    if landscape_smoothing_steps:
        # 固定车行面、人行道和场馆，只柔化道路外侧可调整坡地的局部尖角。
        mask=np.zeros(values.shape,dtype=bool)
        for j,i in lookup:
            p=Point(xs[i],zs[j])
            if all(item['request']['line'].distance(p)>6.0 for item in profiles):
                mask[j,i]=True
        for _ in range(landscape_smoothing_steps):
            previous=values.copy()
            average=(previous[:-2,1:-1]+previous[2:,1:-1]+
                     previous[1:-1,:-2]+previous[1:-1,2:])*.25
            interior=values[1:-1,1:-1];selected=mask[1:-1,1:-1]
            interior[selected]=previous[1:-1,1:-1][selected]*.45+average[selected]*.55
    results=[]
    for item in profiles:
        path=item['request']['line'];distances=item['distances'];original=item['original']
        final=np.asarray([patch.height(p.x,p.y) for p in (path.interpolate(d) for d in distances)])
        results.append({
            'osm_way':item['request'].get('osm_way'),
            'start_m':round(float(item['start']),2),'end_m':round(float(item['end']),2),
            'before_max_sample_grade':round(float(np.max(np.abs(np.diff(original)/np.diff(distances)))),3),
            'after_max_sample_grade':round(float(np.max(np.abs(np.diff(final)/np.diff(distances)))),3),
            'end_height_difference_m':round(float(final[-1]-final[0]),2)
        })
    patch.values=values.astype(np.float32)
    info={**{k:v for k,v in results[0].items() if k!='osm_way'},
        'max_grid_adjustment_m':round(float(np.max(np.abs(delta))),2),
        'detail_grid_spacing_m':spacing,
        'note':'细分地形与坡度为视觉拟合，非实测数据'
    }
    if len(results)>1:
        info['additional_roads']=results[1:]
    return patch,info


def write_patches(path,patches):
    with path.open('wb') as output:
        output.write(b'SZP1'+struct.pack('<I',len(patches)))
        for patch in patches:
            output.write(struct.pack('<II4f',patch.nx,patch.nz,patch.x0,patch.x1,patch.z0,patch.z1))
            output.write(np.asarray(patch.values,dtype='<f4').tobytes())
