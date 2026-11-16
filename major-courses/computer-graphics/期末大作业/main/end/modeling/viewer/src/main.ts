import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import './style.css';

type Vec3 = [number, number, number];
type ViewName = 'hero' | 'front' | 'top' | 'walk';
interface ManifestNode { id: string; parent: string | null; roles: string[]; local_matrix: number[] }
interface ManifestAnchor { id: string; node: string; local: Vec3; world: Vec3 }
interface View { eye: Vec3; target: Vec3 }
interface Manifest {
  schema_version: number; asset_id: string; asset_version: string; name: string;
  synthetic?: boolean; units: string; coordinate_system: string; model: string;
  nodes: ManifestNode[]; anchors: ManifestAnchor[];
  interaction?: { node: string; axis: Vec3; range_deg: [number, number] } | null;
  hierarchy_probe?: { parent: string; descendant: string; rotation_deg: number } | null;
  views?: Partial<Record<ViewName, View>>;
  preview_context?: {model: string; role: string};
  evidence?: {kind?: string};
}
interface Probe {
  ready: boolean;
  errors: string[];
  inspect: () => Record<string, unknown> | null;
  setAngle: (degrees: number) => boolean;
  setView: (name: string) => boolean;
  testHierarchy: () => Record<string, unknown>;
}
declare global { interface Window { __assetProbe: Probe } }

const element = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing UI element: ${id}`);
  return node as T;
};
const state = element<HTMLDivElement>('state');
const status = element<HTMLSpanElement>('top-status');
const statusDot = element<HTMLSpanElement>('status-dot');
const slider = element<HTMLInputElement>('angle');
const gridToggle = element<HTMLInputElement>('grid-toggle');
const wireToggle = element<HTMLInputElement>('wireframe-toggle');
const contextToggle = element<HTMLInputElement>('context-toggle');
const hierarchyButton = element<HTMLButtonElement>('hierarchy-test');
const viewButtons = [...document.querySelectorAll<HTMLButtonElement>('.view-button')];
const text = (id: string, value: string | number) => { element(id).textContent = String(value); };
const vector = (v: Vec3) => new THREE.Vector3(...v);
const finiteVec = (value: unknown): value is Vec3 => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const number = (v: number) => Number(v.toFixed(3)).toLocaleString('zh-CN', { maximumFractionDigits: 3 });
const showState = (title: string, detail: string, kind: 'empty' | 'loading' | 'error') => {
  state.hidden = false;
  state.classList.toggle('error', kind === 'error');
  text('state-symbol', kind === 'error' ? '!' : kind === 'loading' ? '◌' : '＋');
  text('state-title', title); text('state-detail', detail);
  statusDot.className = `status-dot${kind === 'error' ? ' error' : ''}`;
  status.textContent = title; text('footer-status', title);
};
const probe: Probe = {
  ready: false, errors: [], inspect: () => null,
  setAngle: () => false, setView: () => false,
  testHierarchy: () => ({ ok: false, error: '资产尚未载入' }),
};
window.__assetProbe = probe;

let manifest: Manifest | undefined;
let root: THREE.Object3D | undefined;
let surroundings: THREE.Object3D | undefined;
let renderer: THREE.WebGLRenderer;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let scene: THREE.Scene;
let grid: THREE.GridHelper;
let interactionBase: THREE.Matrix4 | undefined;
let currentAngle = 0;
let currentView = 'hero';
const originalWireframe = new WeakMap<THREE.Material, boolean>();

function fail(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  probe.errors.push(message); probe.ready = false;
  showState('资产载入失败', message, 'error');
}

function setupRenderer() {
  const host = element<HTMLDivElement>('canvas-host');
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#e7e5dd');
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
  camera.position.set(9, 7, 11);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 1, 0);
  controls.update();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xa7a99b, 2.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.1);
  key.position.set(8, 15, 10); scene.add(key);
  const fill = new THREE.DirectionalLight(0xe4e6da, 0.8);
  fill.position.set(-8, 6, -5); scene.add(fill);
  grid = new THREE.GridHelper(40, 40, 0x929b8f, 0xbac0b5);
  grid.position.y = 0.002;
  const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
  gridMaterials.forEach(material => { material.transparent = true; material.opacity = 0.3; material.depthWrite = false; });
  scene.add(grid);
  const resize = () => {
    const width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
    camera.aspect = width / height; camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    if (root && window.innerWidth <= 800 && currentView !== 'walk') {
      fitNarrowViewport();
      updateCameraClipping();
      controls.update();
    }
  };
  new ResizeObserver(resize).observe(host); resize();
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
}

function readManifest(data: unknown): Manifest {
  if (!data || typeof data !== 'object') throw new Error('清单不是 JSON 对象。');
  const value = data as Partial<Manifest>;
  if (value.schema_version !== 1 || typeof value.asset_id !== 'string' || typeof value.name !== 'string' || typeof value.model !== 'string') throw new Error('清单缺少有效的身份或模型路径。');
  if (value.units !== 'm' || value.coordinate_system !== 'RH_Y_UP') throw new Error('当前查看器需要米制、右手 Y-up 资产。');
  if (!Array.isArray(value.nodes) || !Array.isArray(value.anchors)) throw new Error('清单缺少节点或锚点列表。');
  for (const node of value.nodes) {
    if (!node || typeof node.id !== 'string' || !(node.parent === null || typeof node.parent === 'string')) throw new Error('清单节点格式无效。');
  }
  for (const anchor of value.anchors) {
    if (!anchor || typeof anchor.id !== 'string' || typeof anchor.node !== 'string' || !finiteVec(anchor.local) || !finiteVec(anchor.world)) throw new Error('清单锚点格式无效。');
  }
  if (value.interaction && (!finiteVec(value.interaction.axis) || value.interaction.axis.every(v => v === 0) || !Array.isArray(value.interaction.range_deg) || value.interaction.range_deg.length !== 2 || !value.interaction.range_deg.every(Number.isFinite) || value.interaction.range_deg[0] > value.interaction.range_deg[1])) throw new Error('动件轴或角度范围无效。');
  return value as Manifest;
}

function manifestUrl(path: string): URL {
  const url = new URL(path, window.location.href);
  if (url.origin !== window.location.origin || !/^\/assets\/[^/]+\/manifest\.json$/.test(url.pathname) || url.search || url.hash) throw new Error('资产路径须为 /assets/<资产 ID>/manifest.json。');
  return url;
}

function modelUrl(model: string, source: URL): URL {
  if (!model || model.startsWith('/') || model.startsWith('\\') || model.includes('\\') || model.includes('%') || model.includes('?') || model.includes('#') || model.split('/').some(part => part === '..' || part === '.') || /^[a-z]+:/i.test(model)) throw new Error('模型路径必须位于资产包内。');
  const url = new URL(model, source);
  if (url.origin !== source.origin || !url.pathname.startsWith(source.pathname.slice(0, source.pathname.lastIndexOf('/') + 1))) throw new Error('模型路径必须位于资产包内。');
  return url;
}

function objectFor(id: string): THREE.Object3D | undefined { return root?.getObjectByName(id); }
function renderNow() { if (root) root.updateMatrixWorld(true); renderer.render(scene, camera); }

function modelStatistics() {
  let nodes = 0, meshes = 0, triangles = 0;
  const materials = new Set<THREE.Material>();
  root?.traverse(object => {
    // GLTFLoader adds a scene wrapper that is not an asset node.
    if (object !== root) nodes++;
    if (object instanceof THREE.Mesh) {
      meshes++;
      const geometry = object.geometry;
      const elements = geometry.index?.count ?? geometry.getAttribute('position')?.count ?? 0;
      const instances = object instanceof THREE.InstancedMesh ? object.count : 1;
      triangles += Math.floor(elements / 3) * instances;
      (Array.isArray(object.material) ? object.material : [object.material]).forEach(material => materials.add(material));
    }
  });
  return { nodes, meshes, triangles, materials: materials.size };
}

function inspect() {
  if (!root || !manifest || !probe.ready) return null;
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root, true);
  const contextBox = surroundings ? new THREE.Box3().setFromObject(surroundings, true) : undefined;
  const size = box.getSize(new THREE.Vector3());
  const anchors = manifest.anchors.map(anchor => {
    const object = objectFor(anchor.node);
    const actual = object ? object.localToWorld(vector(anchor.local)).toArray() : null;
    return { id: anchor.id, node: anchor.node, local: anchor.local, expected: anchor.world, world: actual, actual, error: actual ? Math.hypot(...actual.map((v, i) => v - anchor.world[i])) : null };
  });
  const stats = modelStatistics();
  return {
    assetId: manifest.asset_id, units: manifest.units, coordinateSystem: manifest.coordinate_system,
    bounds: { min: box.min.toArray(), max: box.max.toArray(), size: size.toArray() },
    anchors, nodeCount: stats.nodes, nodes: manifest.nodes.map(n => ({ id: n.id, found: !!objectFor(n.id), parent: n.parent })),
    statistics: stats, angle: currentAngle, view: currentView,
    context: surroundings && contextBox ? {visible: surroundings.visible, bounds: {min: contextBox.min.toArray(), max: contextBox.max.toArray()}} : null,
    renderer: { render: { ...renderer.info.render }, memory: { ...renderer.info.memory }, pixelRatio: renderer.getPixelRatio() },
  };
}

function fitNarrowViewport() {
  if (!root || window.innerWidth > 800) return;
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root, true);
  if (box.isEmpty()) return;
  const offset = camera.position.clone().sub(controls.target);
  const distance = offset.length();
  if (distance < 1e-8) return;
  const outward = offset.normalize();
  const right = new THREE.Vector3().crossVectors(camera.up, outward);
  if (right.lengthSq() < 1e-12) right.crossVectors(new THREE.Vector3(0, 0, 1), outward);
  right.normalize();
  const up = new THREE.Vector3().crossVectors(outward, right).normalize();
  const tangentY = Math.tan(THREE.MathUtils.degToRad(camera.getEffectiveFOV()) / 2);
  const tangentX = tangentY * camera.aspect;
  let needed = distance;
  // Fit all world-space bounding-box corners, with room for the viewport controls.
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
    const corner = new THREE.Vector3(x, y, z).sub(controls.target);
    const depth = corner.dot(outward);
    needed = Math.max(needed, depth + Math.abs(corner.dot(right)) * 1.12 / tangentX, depth + Math.abs(corner.dot(up)) * 1.27 / tangentY);
  }
  camera.position.copy(controls.target).addScaledVector(outward, needed);
}

function updateCameraClipping() {
  const distance = camera.position.distanceTo(controls.target);
  camera.near = Math.max(0.001, distance / 10000); camera.far = Math.max(1000, distance * 100);
  camera.updateProjectionMatrix();
}

function setView(name: string): boolean {
  if (!root || !manifest || !['hero', 'front', 'top', 'walk'].includes(name)) return false;
  const view = manifest.views?.[name as ViewName];
  if (view && finiteVec(view.eye) && finiteVec(view.target)) {
    camera.position.copy(vector(view.eye)); controls.target.copy(vector(view.target));
  } else {
    const box = new THREE.Box3().setFromObject(root, true);
    const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const distance = Math.max(size.x, size.y, size.z, 1) * 2.2;
    const direction = name === 'front' ? new THREE.Vector3(0, 0.18, 1) : name === 'top' ? new THREE.Vector3(0, 1, 0.001) : new THREE.Vector3(1, 0.8, 1);
    camera.position.copy(center).add(direction.normalize().multiplyScalar(distance)); controls.target.copy(center);
  }
  if (name !== 'walk') fitNarrowViewport();
  updateCameraClipping(); controls.update();
  currentView = name;
  viewButtons.forEach(button => { const selected = button.dataset.view === name; button.classList.toggle('active', selected); button.setAttribute('aria-pressed', String(selected)); });
  renderNow(); return true;
}

function setAngle(degrees: number): boolean {
  const control = manifest?.interaction;
  const node = control && objectFor(control.node);
  if (!node || !control || !interactionBase || !Number.isFinite(degrees)) return false;
  currentAngle = Math.max(control.range_deg[0], Math.min(control.range_deg[1], degrees));
  const rotation = new THREE.Matrix4().makeRotationAxis(vector(control.axis).normalize(), THREE.MathUtils.degToRad(currentAngle));
  node.matrixAutoUpdate = false;
  node.matrix.copy(interactionBase).multiply(rotation);
  node.matrixWorldNeedsUpdate = true;
  slider.value = String(currentAngle); text('angle-value', `${number(currentAngle)}°`);
  renderNow(); return true;
}

function testHierarchy(): Record<string, unknown> {
  const spec = manifest?.hierarchy_probe;
  if (!root || !spec) return { ok: false, error: '清单未声明层级测试。' };
  const parent = objectFor(spec.parent), child = objectFor(spec.descendant);
  if (!parent || !child) return { ok: false, error: '测试节点未在模型中找到。' };
  let ancestor = child.parent, isDescendant = false;
  while (ancestor) { if (ancestor === parent) { isDescendant = true; break; } ancestor = ancestor.parent; }
  if (!isDescendant) return { ok: false, isDescendant, error: '指定节点不是父节点的后代。' };
  root.updateMatrixWorld(true);
  const before = child.matrixWorld.clone();
  const savedAutoUpdate = parent.matrixAutoUpdate;
  if (savedAutoUpdate) parent.updateMatrix();
  const savedLocal = parent.matrix.clone();
  let after = before.clone();
  try {
    parent.matrixAutoUpdate = false;
    parent.matrix.copy(savedLocal).multiply(new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(spec.rotation_deg)));
    parent.matrixWorldNeedsUpdate = true; root.updateMatrixWorld(true);
    after = child.matrixWorld.clone();
  } finally {
    parent.matrix.copy(savedLocal); parent.matrixAutoUpdate = savedAutoUpdate;
    parent.matrixWorldNeedsUpdate = true; root.updateMatrixWorld(true);
  }
  const restored = child.matrixWorld.clone();
  const delta = Math.max(...before.elements.map((value, i) => Math.abs(value - after.elements[i])));
  const restoreDelta = Math.max(...before.elements.map((value, i) => Math.abs(value - restored.elements[i])));
  renderNow();
  return { ok: isDescendant && delta > 1e-8 && restoreDelta < 1e-8, isDescendant, changed: delta > 1e-8, restored: restoreDelta < 1e-8, maxDelta: delta, restoreMaxDelta: restoreDelta, parent: spec.parent, descendant: spec.descendant, before: before.toArray(), after: after.toArray(), restoredMatrix: restored.toArray() };
}

function applyWireframe(enabled: boolean) {
  root?.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    (Array.isArray(object.material) ? object.material : [object.material]).forEach(material => {
      const candidate = material as THREE.Material & { wireframe?: boolean };
      if (typeof candidate.wireframe !== 'boolean') return;
      if (!originalWireframe.has(material)) originalWireframe.set(material, candidate.wireframe);
      candidate.wireframe = enabled ? true : originalWireframe.get(material)!;
    });
  });
  renderNow();
}

function buildHierarchy() {
  if (!manifest) return;
  const target = element<HTMLDivElement>('hierarchy'); target.replaceChildren();
  const ordered = manifest.nodes, ids = new Set(ordered.map(n => n.id));
  const seen = new Set<string>();
  const add = (node: ManifestNode, depth: number) => {
    if (seen.has(node.id)) return; seen.add(node.id);
    const button = document.createElement('button'); button.type = 'button'; button.className = 'tree-row';
    if (!objectFor(node.id)) button.classList.add('missing');
    button.style.paddingLeft = `${8 + Math.min(depth, 12) * 13}px`;
    const glyph = document.createElement('span'); glyph.className = 'tree-glyph'; glyph.textContent = ordered.some(n => n.parent === node.id) ? '▾' : '·';
    const label = document.createElement('span'); label.className = 'tree-name'; label.textContent = node.id;
    button.title = `${node.id}${node.roles?.length ? ` · ${node.roles.join(', ')}` : ''}`;
    button.append(glyph, label);
    button.addEventListener('click', () => { target.querySelectorAll('.tree-row').forEach(row => row.classList.remove('selected')); button.classList.add('selected'); });
    target.appendChild(button);
    ordered.filter(n => n.parent === node.id).forEach(child => add(child, depth + 1));
  };
  ordered.filter(node => node.parent === null || !ids.has(node.parent)).forEach(node => add(node, 0));
  ordered.forEach(node => add(node, 0));
  text('hierarchy-count', `${ordered.length} NODES`);
}

async function loadAsset() {
  const asset = new URLSearchParams(location.search).get('asset');
  if (!asset) { showState('等待资产', '在地址中提供 /assets/<资产 ID>/manifest.json。', 'empty'); return; }
  showState('正在载入', '读取资产清单与三维模型…', 'loading');
  const url = manifestUrl(asset);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`无法读取清单（HTTP ${response.status}）：${url.pathname}`);
  let data: unknown;
  try { data = await response.json(); } catch { throw new Error(`清单不是有效 JSON：${url.pathname}`); }
  manifest = readManifest(data);
  const gltf = await new GLTFLoader().loadAsync(modelUrl(manifest.model, url).href);
  root = gltf.scene; scene.add(root); root.updateMatrixWorld(true);
  // The read-only environment is a sibling, never a child of the editable asset.
  if (manifest.preview_context) {
    const context = await new GLTFLoader().loadAsync(modelUrl(manifest.preview_context.model, url).href);
    surroundings = context.scene; scene.add(surroundings); surroundings.updateMatrixWorld(true);
    element('context-display').hidden = false;
    grid.visible = false; gridToggle.checked = false;
  }
  const missing = manifest.nodes.filter(node => !objectFor(node.id)).map(node => node.id);
  if (missing.length) throw new Error(`模型缺少清单节点：${missing.join('、')}`);
  if (manifest.anchors.some(anchor => !objectFor(anchor.node))) throw new Error('模型缺少锚点所属节点。');
  if (manifest.interaction) {
    const movable = objectFor(manifest.interaction.node);
    if (!movable) throw new Error(`模型缺少动件节点：${manifest.interaction.node}`);
    movable.updateMatrix(); interactionBase = movable.matrix.clone();
    slider.min = String(manifest.interaction.range_deg[0]); slider.max = String(manifest.interaction.range_deg[1]);
    slider.step = '0.1'; slider.disabled = false;
    text('angle-min', `${number(manifest.interaction.range_deg[0])}°`); text('angle-max', `${number(manifest.interaction.range_deg[1])}°`); text('interaction-node', manifest.interaction.node);
    setAngle(Math.max(manifest.interaction.range_deg[0], Math.min(manifest.interaction.range_deg[1], 0)));
  }
  if (manifest.hierarchy_probe) hierarchyButton.disabled = false;
  const box = new THREE.Box3().setFromObject(root, true);
  if (box.isEmpty() || !box.min.toArray().every(Number.isFinite) || !box.max.toArray().every(Number.isFinite)) throw new Error('模型没有有效的几何包围盒。');
  const size = box.getSize(new THREE.Vector3()), stats = modelStatistics();
  text('asset-name', manifest.name); text('asset-id', `${manifest.asset_id}  /  v${manifest.asset_version || '—'}`);
  element('synthetic-tag').hidden = !manifest.synthetic && manifest.evidence?.kind !== 'photo_based_approximation';
  text('synthetic-tag', manifest.synthetic ? '测试夹具' : '照片近似');
  text('metric-size', size.toArray().map(number).join(' × ')); text('metric-triangles', stats.triangles.toLocaleString('zh-CN')); text('metric-meshes', stats.meshes); text('metric-nodes', stats.nodes);
  element('walk-view').hidden = !manifest.views?.walk;
  buildHierarchy(); setView('hero');
  probe.inspect = inspect; probe.setAngle = setAngle; probe.setView = setView; probe.testHierarchy = testHierarchy;
  probe.ready = true; renderNow();
  state.hidden = true; statusDot.className = 'status-dot ready'; status.textContent = '资产已载入'; text('footer-status', '已就绪');
}

slider.addEventListener('input', () => { setAngle(Number(slider.value)); });
gridToggle.addEventListener('change', () => { if (grid) grid.visible = gridToggle.checked; });
contextToggle.addEventListener('change', () => { if (surroundings) surroundings.visible = contextToggle.checked; });
wireToggle.addEventListener('change', () => { if (root) applyWireframe(wireToggle.checked); });
viewButtons.forEach(button => button.addEventListener('click', () => { if (button.dataset.view) setView(button.dataset.view); }));
hierarchyButton.addEventListener('click', () => {
  const result = testHierarchy();
  text('hierarchy-result', result.ok ? `测试通过 · 后代随父节点旋转，已恢复` : `测试未通过 · ${result.error || '变换未传递或恢复'}`);
});

try { setupRenderer(); void loadAsset().catch(fail); } catch (error) { fail(error); }
