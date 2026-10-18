// RoomGen.Agent 3D scene viewer: loads <build>/scene.glb with three.js and
// offers an orbit camera plus first-person Fly/Walk navigation, with jumps to
// the panoramas' camera poses recorded in scene3d_manifest.json.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  addTriangle, createWalkIndex, findStandingSpot, stepWalk, supportBelow, WALK_DEFAULTS,
} from './walk.mjs';

const GLB_URL = '/scene.glb';
const MANIFEST_URL = '/manifest.json';

const DEFAULT_EYE_HEIGHT = 1.5;
const WALK_SPEED = 2.8;   // m/s
const FLY_SPEED = 4.0;    // m/s
const SPRINT_FACTOR = 2.2;
const LOOK_SPEED = 0.0022;
const PITCH_LIMIT = Math.PI / 2 - 0.02;
const SECTION_CUT_M = 1.45;  // section plane above the active floor surface

const ui = {
  canvas: document.getElementById('scene-canvas'),
  title: document.getElementById('scene-title'),
  stats: document.getElementById('scene-stats'),
  viewpoint: document.getElementById('viewpoint-select'),
  floor: document.getElementById('floor-select'),
  section: document.getElementById('section-toggle'),
  modeOrbit: document.getElementById('mode-orbit'),
  modeFly: document.getElementById('mode-fly'),
  modeWalk: document.getElementById('mode-walk'),
  reset: document.getElementById('reset-view'),
  loading: document.getElementById('loading'),
  loadingLabel: document.getElementById('loading-label'),
  progress: document.getElementById('progress-bar'),
  pointerHint: document.getElementById('pointer-hint'),
  error: document.getElementById('error-box'),
  hint: document.getElementById('hint-text'),
  pose: document.getElementById('pose-readout'),
};

const state = {
  mode: 'orbit',
  manifest: null,
  bounds: null,
  center: new THREE.Vector3(),
  radius: 5,
  eyeHeight: DEFAULT_EYE_HEIGHT,
  yaw: 0,
  pitch: 0,
  locked: false,
  keys: new Set(),
  floors: [],
  rooms: [],
  roomById: new Map(),
  walkIndex: null,
  feetY: null,
  activeFloor: null,
  sectionCut: false,
  clipPlanes: [],
  modelMaterials: [],
};

const renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.localClippingEnabled = true;

const scene = new THREE.Scene();
scene.background = skyGradient();

const camera = new THREE.PerspectiveCamera(62, 1, 0.05, 4000);
camera.rotation.order = 'YXZ';

const controls = new OrbitControls(camera, ui.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.2;
controls.maxDistance = 300;
controls.screenSpacePanning = false;

const hemisphere = new THREE.HemisphereLight(0xdfe9ff, 0x4a453d, 0.75);
scene.add(hemisphere);

const sun = new THREE.DirectionalLight(0xfff3e0, 2.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0006;
sun.shadow.normalBias = 0.03;
scene.add(sun);
scene.add(sun.target);

const fill = new THREE.DirectionalLight(0xdfe8ff, 0.55);
scene.add(fill);

const clock = new THREE.Clock();

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //
function skyGradient() {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, '#9fc4e8');
  gradient.addColorStop(0.55, '#d7e6f2');
  gradient.addColorStop(1, '#efe9df');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function showError(message) {
  ui.error.hidden = false;
  ui.error.textContent = message;
  ui.loading.classList.add('is-hidden');
}

function setProgress(fraction, label) {
  if (typeof fraction === 'number' && Number.isFinite(fraction)) {
    ui.progress.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
  }
  if (label) {
    ui.loadingLabel.textContent = label;
  }
}

function formatStats(manifest) {
  const stats = (manifest && manifest.stats) || {};
  const parts = [];
  const objects = stats.mesh_nodes ?? stats.instances;
  const triangles = stats.triangles_rendered ?? stats.triangles;
  if (objects) parts.push(`${objects} objects`);
  if (triangles) parts.push(`${triangles.toLocaleString()} tris`);
  if (stats.materials) parts.push(`${stats.materials} materials`);
  if (stats.bytes) parts.push(`${(stats.bytes / 1e6).toFixed(1)} MB`);
  return parts.join(' · ');
}

function vectorFrom(array, fallback) {
  if (Array.isArray(array) && array.length === 3 && array.every((v) => Number.isFinite(v))) {
    return new THREE.Vector3(array[0], array[1], array[2]);
  }
  return fallback.clone();
}

function boundsFromBox(box) {
  return {
    min: [box.min.x, box.min.y, box.min.z],
    max: [box.max.x, box.max.y, box.max.z],
  };
}

// --------------------------------------------------------------------------- //
// floors, walk surfaces and section clipping
// --------------------------------------------------------------------------- //
function pointInPolygon(x, z, polygon) {
  if (!Array.isArray(polygon) || polygon.length < 3) {
    return false;
  }
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i][0];
    const zi = polygon[i][1];
    const xj = polygon[j][0];
    const zj = polygon[j][1];
    if (((zi > z) !== (zj > z))
        && (x < ((xj - xi) * (z - zi)) / (zj - zi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function roomAt(x, z, floorId = undefined) {
  return state.rooms.find((room) => {
    if (floorId !== undefined && room.floor_id !== floorId) {
      return false;
    }
    return pointInPolygon(x, z, room.polygon);
  }) || null;
}

function floorById(id) {
  return state.floors.find((floor) => floor.id === id) || null;
}

function floorForPoint(x, z, referenceY) {
  const candidates = state.floors.filter((floor) => (floor.rooms || []).some((roomId) => {
    const room = state.roomById.get(roomId);
    return room && pointInPolygon(x, z, room.polygon);
  }));
  if (!candidates.length) {
    return null;
  }
  candidates.sort((a, b) => a.elevation - b.elevation);
  const below = candidates.filter((floor) => floor.elevation <= referenceY + 0.05);
  return below.length ? below[below.length - 1] : candidates[0];
}

function firstRoomOnFloor(floor) {
  if (!floor) {
    return null;
  }
  for (const roomId of floor.rooms || []) {
    const room = state.roomById.get(roomId);
    if (room) {
      return room;
    }
  }
  return null;
}

function roomFloor(room) {
  return room && room.floor_id ? floorById(room.floor_id) : null;
}

function walkOptions() {
  const options = {
    maxStepUp: WALK_DEFAULTS.maxStepUp,
    maxDrop: WALK_DEFAULTS.maxDrop,
    bodyRadius: WALK_DEFAULTS.bodyRadius,
  };
  if (state.activeFloor) {
    const floor = state.activeFloor;
    // An isolated storey must not be climbed invisibly: the upper flight is
    // clipped away, so the walker stays on (or just above) that floor surface.
    options.acceptSupport = (y) => y >= floor.elevation - 0.08
      && y <= floor.elevation + Math.max(0.5, WALK_DEFAULTS.maxStepUp + 0.2);
  }
  return options;
}

function buildWalkIndex(root) {
  const index = createWalkIndex();
  root.updateMatrixWorld(true);
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  root.traverse((node) => {
    if (!node.isMesh || !node.geometry) {
      return;
    }
    const position = node.geometry.getAttribute('position');
    if (!position) {
      return;
    }
    const element = node.geometry.getIndex();
    const count = element ? element.count : position.count;
    for (let corner = 0; corner + 2 < count; corner += 3) {
      const i0 = element ? element.getX(corner) : corner;
      const i1 = element ? element.getX(corner + 1) : corner + 1;
      const i2 = element ? element.getX(corner + 2) : corner + 2;
      a.fromBufferAttribute(position, i0).applyMatrix4(node.matrixWorld);
      b.fromBufferAttribute(position, i1).applyMatrix4(node.matrixWorld);
      c.fromBufferAttribute(position, i2).applyMatrix4(node.matrixWorld);
      addTriangle(index, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    }
  });
  return index;
}

function collectMaterials(root) {
  const materials = new Set();
  root.traverse((node) => {
    if (!node.isMesh) {
      return;
    }
    const list = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of list) {
      if (material) {
        materials.add(material);
      }
    }
  });
  return [...materials];
}

function applyClipping() {
  const planes = [];
  if (state.activeFloor) {
    const floor = state.activeFloor;
    // Keep the storey body: from the underside of its slab to its ceiling.
    // Hand-written/older manifests may omit the derived bounds, so fall back
    // to elevation - thickness / elevation + height.
    const thickness = Number.isFinite(floor.slab_thickness) ? floor.slab_thickness : 0.15;
    const height = Number.isFinite(floor.height) ? floor.height : 2.7;
    const bottom = Number.isFinite(floor.slab_bottom) ? floor.slab_bottom : floor.elevation - thickness;
    const ceiling = Number.isFinite(floor.top) ? floor.top : floor.elevation + height;
    planes.push(new THREE.Plane(new THREE.Vector3(0, 1, 0), -bottom));
    const top = state.sectionCut ? floor.elevation + SECTION_CUT_M : ceiling;
    planes.push(new THREE.Plane(new THREE.Vector3(0, -1, 0), top));
  }
  state.clipPlanes = planes;
  for (const material of state.modelMaterials) {
    material.clippingPlanes = planes.length ? planes : null;
    material.clipShadows = planes.length > 0;
    material.needsUpdate = true;
  }
}

function snapWalkerAt(x, z, { floor = null, referenceY = null } = {}) {
  const eye = state.eyeHeight;
  let targetFloor = floor || state.activeFloor || floorForPoint(
    x, z, referenceY === null ? camera.position.y - eye : referenceY);
  let feet = null;

  if (state.walkIndex && targetFloor) {
    // Prefer the storey surface itself: room centres are bounding-box centres
    // and may sit under a wardrobe or a bed, whose top is not a floor.
    const surface = supportBelow(state.walkIndex, x, z,
      targetFloor.elevation + 0.45, targetFloor.elevation - 0.08);
    feet = surface ? surface.y : targetFloor.elevation;
  } else if (state.walkIndex) {
    // Legacy single-storey manifest: probe around standing height so an orbit
    // camera above the roof does not land the walker on top of the house.
    const fromY = state.rooms.length
      ? state.eyeHeight + WALK_DEFAULTS.maxStepUp
      : camera.position.y + 0.5;
    const surface = supportBelow(state.walkIndex, x, z,
      fromY, state.bounds ? state.bounds.min[1] : -10);
    feet = surface ? surface.y : camera.position.y - eye;
  } else {
    feet = targetFloor ? targetFloor.elevation : camera.position.y - eye;
  }

  if (state.walkIndex) {
    // Room centres and camera poses can sit inside a wardrobe or over a slab
    // opening; nudge to the nearest free standing spot on the same storey.
    const options = walkOptions();
    const room = roomAt(x, z, targetFloor ? targetFloor.id : undefined)
      || (targetFloor ? firstRoomOnFloor(targetFloor) : null);
    if (room && Array.isArray(room.polygon) && room.polygon.length >= 3) {
      options.acceptPoint = (px, pz) => pointInPolygon(px, pz, room.polygon);
    }
    const spot = findStandingSpot(state.walkIndex, x, z, feet, options);
    if (spot) {
      x = spot.x;
      z = spot.z;
      feet = spot.feetY;
    }
  }

  state.feetY = feet;
  camera.position.set(x, feet + eye, z);
}

function enterWalkFromCamera() {
  let x = camera.position.x;
  let z = camera.position.z;
  let floor = state.activeFloor
    || floorForPoint(x, z, camera.position.y - state.eyeHeight);

  if (!floor && state.floors.length) {
    floor = state.floors[0];
  }
  const room = roomAt(x, z, floor ? floor.id : undefined)
    || firstRoomOnFloor(floor)
    || state.rooms[0];
  if (room && !roomAt(x, z, floor ? floor.id : undefined)) {
    x = room.center[0];
    z = room.center[2];
    floor = roomFloor(room) || floor;
  }
  snapWalkerAt(x, z, { floor, referenceY: camera.position.y - state.eyeHeight });
}

// --------------------------------------------------------------------------- //
// scene setup
// --------------------------------------------------------------------------- //
function applyBounds(bounds) {
  state.bounds = bounds;
  const min = new THREE.Vector3(bounds.min[0], bounds.min[1], bounds.min[2]);
  const max = new THREE.Vector3(bounds.max[0], bounds.max[1], bounds.max[2]);
  const size = new THREE.Vector3().subVectors(max, min);
  state.center.copy(min).addScaledVector(size, 0.5);
  state.radius = Math.max(1, 0.5 * size.length());

  const fit = state.radius * 1.35;
  sun.position.copy(state.center).add(new THREE.Vector3(fit * 0.75, fit * 1.15, fit * 0.65));
  sun.target.position.copy(state.center);
  const shadowCam = sun.shadow.camera;
  shadowCam.left = -fit;
  shadowCam.right = fit;
  shadowCam.top = fit;
  shadowCam.bottom = -fit;
  shadowCam.near = 0.1;
  shadowCam.far = fit * 6;
  shadowCam.updateProjectionMatrix();

  fill.position.copy(state.center).add(new THREE.Vector3(-fit * 0.8, fit * 0.9, -fit * 0.7));
  fill.target = sun.target;
  scene.add(fill.target);
}

function overviewPose() {
  const offset = new THREE.Vector3(1.0, 0.62, 1.0).normalize().multiplyScalar(state.radius * 1.5);
  return { position: state.center.clone().add(offset), target: state.center.clone() };
}

function setOrbitPose(position, target) {
  camera.position.copy(position);
  controls.target.copy(target);
  camera.lookAt(target);
  controls.update();
}

// --------------------------------------------------------------------------- //
// camera modes
// --------------------------------------------------------------------------- //
function setMode(mode) {
  if (!['orbit', 'fly', 'walk'].includes(mode)) {
    return;
  }
  const previous = state.mode;
  state.mode = mode;
  const firstPerson = mode !== 'orbit';
  controls.enabled = !firstPerson;
  ui.modeOrbit.classList.toggle('is-active', mode === 'orbit');
  ui.modeFly.classList.toggle('is-active', mode === 'fly');
  ui.modeWalk.classList.toggle('is-active', mode === 'walk');
  ui.canvas.classList.toggle('is-first-person', firstPerson);

  if (firstPerson) {
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    state.yaw = Math.atan2(-direction.x, -direction.z);
    state.pitch = Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1));
    if (mode === 'walk') {
      state.pitch = 0;
      if (previous !== 'walk' || state.feetY === null) {
        enterWalkFromCamera();
      } else {
        camera.position.y = state.feetY + state.eyeHeight;
      }
    }
    applyLook();
    ui.pointerHint.hidden = state.locked;
    ui.hint.textContent = mode === 'walk'
      ? 'Walk: WASD follows the GLB floors and stairs, walls/furniture block, mouse looks, Shift sprints, Esc releases'
      : 'Fly: WASD moves, mouse looks, Shift sprints, Space/C up-down, Esc releases the mouse';
  } else {
    if (document.pointerLockElement === ui.canvas) {
      document.exitPointerLock();
    }
    ui.pointerHint.hidden = true;
    ui.hint.textContent = 'Orbit: drag to rotate, wheel to zoom, right-drag to pan';
    setOrbitPose(camera.position.clone(), state.center.clone());
  }
  updatePoseReadout();
}

function applyLook() {
  camera.rotation.set(state.pitch, state.yaw, 0, 'YXZ');
}

function updateFirstPerson(delta) {
  const sprinting = state.keys.has('ShiftLeft') || state.keys.has('ShiftRight');
  const baseSpeed = state.mode === 'fly' ? FLY_SPEED : WALK_SPEED;
  const speed = baseSpeed * (sprinting ? SPRINT_FACTOR : 1) * delta;

  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const flat = new THREE.Vector3(forward.x, 0, forward.z);
  if (flat.lengthSq() < 1e-8) {
    flat.set(-Math.sin(state.yaw), 0, -Math.cos(state.yaw));
  }
  flat.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  right.y = 0;
  right.normalize();

  const move = new THREE.Vector3();
  if (state.keys.has('KeyW')) move.add(state.mode === 'walk' ? flat : forward);
  if (state.keys.has('KeyS')) move.sub(state.mode === 'walk' ? flat : forward);
  if (state.keys.has('KeyD')) move.add(right);
  if (state.keys.has('KeyA')) move.sub(right);
  if (state.mode === 'fly') {
    if (state.keys.has('Space') || state.keys.has('KeyE')) move.y += 1;
    if (state.keys.has('KeyC') || state.keys.has('KeyQ')) move.y -= 1;
  }

  if (state.mode === 'walk') {
    if (move.lengthSq() > 0) {
      move.y = 0;
      move.normalize();
      const pose = {
        x: camera.position.x,
        z: camera.position.z,
        feetY: state.feetY === null ? camera.position.y - state.eyeHeight : state.feetY,
      };
      if (state.walkIndex) {
        const next = stepWalk(state.walkIndex, pose, move.x * speed, move.z * speed, walkOptions());
        camera.position.x = next.x;
        camera.position.z = next.z;
        state.feetY = next.feetY;
      } else {
        camera.position.x += move.x * speed;
        camera.position.z += move.z * speed;
        const floor = state.activeFloor;
        state.feetY = floor ? floor.elevation : state.feetY;
      }
    }
    // The eye rides on the surface the GLB reports underneath the walker; the
    // short interpolation only smooths tread-to-tread snaps, it is not a height.
    const targetY = state.feetY + state.eyeHeight;
    camera.position.y += (targetY - camera.position.y) * Math.min(1, delta * 14);
  } else if (move.lengthSq() > 0) {
    camera.position.addScaledVector(move.normalize(), speed);
  }
}

function poseFromCamera(cameraEntry) {
  const position = vectorFrom(cameraEntry.position, state.center);
  if (cameraEntry.forward) {
    const forward = vectorFrom(cameraEntry.forward, new THREE.Vector3(1, 0, 0)).normalize();
    return { position, forward };
  }
  const yaw = THREE.MathUtils.degToRad(cameraEntry.yaw_deg || 0);
  return { position, forward: new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw)) };
}

function jumpToCamera(cameraEntry) {
  const { position, forward } = poseFromCamera(cameraEntry);
  const floor = cameraEntry.floor_id ? floorById(cameraEntry.floor_id) : null;
  if (state.activeFloor && floor && state.activeFloor.id !== floor.id) {
    setActiveFloor(floor, { teleport: false });
  }
  if (state.mode === 'orbit') {
    const target = position.clone().addScaledVector(forward, Math.max(1.5, state.radius * 0.35));
    setOrbitPose(position, target);
  } else {
    if (state.mode === 'walk') {
      snapWalkerAt(position.x, position.z, {
        floor,
        referenceY: position.y - state.eyeHeight,
      });
    } else {
      camera.position.copy(position);
    }
    state.yaw = Math.atan2(-forward.x, -forward.z);
    state.pitch = 0;
    applyLook();
  }
  updatePoseReadout();
}

function jumpToRoom(room) {
  const position = vectorFrom(room.center, state.center);
  const target = position.clone();
  const floor = roomFloor(room);
  if (state.activeFloor && floor && state.activeFloor.id !== floor.id) {
    setActiveFloor(floor, { teleport: false });
  }
  if (state.mode === 'orbit') {
    setOrbitPose(position.clone().add(new THREE.Vector3(state.radius * 0.5, state.radius * 0.4, state.radius * 0.5)), target);
  } else {
    if (state.mode === 'walk') {
      snapWalkerAt(position.x, position.z, {
        floor,
        referenceY: position.y - state.eyeHeight,
      });
    } else {
      camera.position.copy(position);
    }
    state.yaw = 0;
    state.pitch = 0;
    applyLook();
  }
  updatePoseReadout();
}

function resetView() {
  if (state.mode === 'orbit') {
    const pose = overviewPose();
    setOrbitPose(pose.position, pose.target);
  } else if (state.mode === 'walk') {
    const floor = state.activeFloor || state.floors[0] || null;
    const room = firstRoomOnFloor(floor) || state.rooms[0] || null;
    const x = room ? room.center[0] : state.center.x;
    const z = room ? room.center[2] : state.center.z;
    snapWalkerAt(x, z, { floor: roomFloor(room) || floor });
    state.yaw = 0;
    state.pitch = 0;
    applyLook();
  } else {
    const pose = overviewPose();
    camera.position.copy(pose.position);
    const direction = new THREE.Vector3().subVectors(pose.target, pose.position).normalize();
    state.yaw = Math.atan2(-direction.x, -direction.z);
    state.pitch = Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1));
    applyLook();
  }
  updatePoseReadout();
}

// --------------------------------------------------------------------------- //
// UI wiring
// --------------------------------------------------------------------------- //
function populateViewpoints(manifest) {
  const select = ui.viewpoint;
  select.textContent = '';
  const overview = document.createElement('option');
  overview.value = 'overview';
  overview.textContent = 'Overview';
  select.append(overview);

  const floorLabel = (entry) => (entry.floor_id ? `${entry.floor_id}: ` : '');

  const cameras = manifest.cameras || [];
  if (cameras.length) {
    const group = document.createElement('optgroup');
    group.label = 'Panorama viewpoints';
    for (const entry of cameras) {
      const option = document.createElement('option');
      option.value = `camera:${entry.name}`;
      option.textContent = entry.room ? `${entry.name} — ${floorLabel(entry)}${entry.room}` : entry.name;
      group.append(option);
    }
    select.append(group);
  }

  const cameraRooms = new Set(cameras.map((entry) => entry.room));
  const rooms = (manifest.rooms || []).filter((room) => room && !cameraRooms.has(room.id));
  if (rooms.length) {
    const group = document.createElement('optgroup');
    group.label = 'Rooms';
    for (const room of rooms) {
      const option = document.createElement('option');
      option.value = `room:${room.id}`;
      const label = room.type ? `${room.id} — ${room.type}` : room.id;
      option.textContent = `${floorLabel(room)}${label}`;
      group.append(option);
    }
    select.append(group);
  }
  select.disabled = false;
}

function populateFloors(manifest) {
  const select = ui.floor;
  select.textContent = '';
  const all = document.createElement('option');
  all.value = '';
  all.textContent = 'All floors';
  select.append(all);
  for (const floor of manifest.floors || []) {
    const option = document.createElement('option');
    option.value = floor.id;
    const elevation = Number(floor.elevation);
    option.textContent = `Floor ${floor.id} · ${Number.isFinite(elevation) ? elevation.toFixed(2) : '?'} m`;
    select.append(option);
  }
  select.disabled = !(manifest.floors || []).length;
  if (ui.section) {
    ui.section.disabled = !(manifest.floors || []).length;
  }
}

function setActiveFloor(floor, { teleport = true } = {}) {
  state.activeFloor = floor || null;
  if (ui.floor) {
    ui.floor.value = floor ? floor.id : '';
  }
  if (ui.section) {
    ui.section.disabled = !floor;
    if (!floor) {
      state.sectionCut = false;
      ui.section.classList.remove('is-active');
      ui.section.setAttribute('aria-pressed', 'false');
    }
  }
  applyClipping();
  if (floor && teleport) {
    const here = roomAt(camera.position.x, camera.position.z, floor.id);
    const room = here || firstRoomOnFloor(floor);
    if (room) {
      const x = here ? camera.position.x : room.center[0];
      const z = here ? camera.position.z : room.center[2];
      if (state.mode === 'walk') {
        snapWalkerAt(x, z, { floor });
        state.pitch = 0;
        applyLook();
      } else if (state.mode === 'fly') {
        camera.position.set(x, floor.elevation + state.eyeHeight, z);
      } else {
        const target = new THREE.Vector3(x, floor.elevation + 1.0, z);
        setOrbitPose(target.clone().add(new THREE.Vector3(state.radius * 0.45,
                                                           state.radius * 0.4,
                                                           state.radius * 0.45)), target);
      }
    }
  }
  updatePoseReadout();
}

function onFloorChange() {
  const id = ui.floor.value;
  setActiveFloor(id ? floorById(id) : null);
}

function onSectionToggle() {
  state.sectionCut = !state.sectionCut;
  ui.section.classList.toggle('is-active', state.sectionCut);
  ui.section.setAttribute('aria-pressed', String(state.sectionCut));
  applyClipping();
}

function onViewpointChange() {
  const value = ui.viewpoint.value;
  if (!value || value === 'overview') {
    resetView();
    return;
  }
  const [kind, id] = value.split(':');
  if (kind === 'camera') {
    const entry = (state.manifest.cameras || []).find((camera_) => camera_.name === id);
    if (entry) jumpToCamera(entry);
    return;
  }
  if (kind === 'room') {
    const room = (state.manifest.rooms || []).find((room_) => room_.id === id);
    if (room) jumpToRoom(room);
  }
}

function updatePoseReadout() {
  const p = camera.position;
  if (state.mode === 'walk') {
    const floor = state.activeFloor ? ` · ${state.activeFloor.id}` : '';
    const ground = state.feetY === null ? '' : ` · ground ${state.feetY.toFixed(2)}`;
    ui.pose.textContent = `walk${floor} · x ${p.x.toFixed(2)} y ${p.y.toFixed(2)} z ${p.z.toFixed(2)} m${ground}`;
    return;
  }
  ui.pose.textContent = `${state.mode} · x ${p.x.toFixed(2)} y ${p.y.toFixed(2)} z ${p.z.toFixed(2)} m`;
}

function onKeyDown(event) {
  if (event.target instanceof HTMLSelectElement) {
    return;
  }
  if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.code)) {
    event.preventDefault();
  }
  if (event.code === 'Digit1') setMode('orbit');
  if (event.code === 'Digit2') setMode('fly');
  if (event.code === 'Digit3') setMode('walk');
  if (event.code === 'KeyR') resetView();
  state.keys.add(event.code);
}

function onPointerLockChange() {
  state.locked = document.pointerLockElement === ui.canvas;
  ui.pointerHint.hidden = state.locked || state.mode === 'orbit';
  if (!state.locked) {
    state.keys.clear();
  }
}

function onMouseMove(event) {
  if (!state.locked || state.mode === 'orbit') {
    return;
  }
  state.yaw -= event.movementX * LOOK_SPEED;
  state.pitch -= event.movementY * LOOK_SPEED;
  state.pitch = THREE.MathUtils.clamp(state.pitch, -PITCH_LIMIT, PITCH_LIMIT);
  applyLook();
}

function onResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

// --------------------------------------------------------------------------- //
// load
// --------------------------------------------------------------------------- //
async function fetchManifest() {
  const response = await fetch(MANIFEST_URL, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`manifest.json: HTTP ${response.status}`);
  }
  return response.json();
}

function prepareModel(root) {
  root.traverse((node) => {
    if (node.isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        if (material && 'envMapIntensity' in material) {
          material.envMapIntensity = 1.0;
        }
      }
    }
  });
  return root;
}

function loadModel() {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      GLB_URL,
      (gltf) => resolve(gltf),
      (event) => {
        if (event && event.lengthComputable && event.total > 0) {
          const loaded = event.loaded / 1e6;
          setProgress(event.loaded / event.total, `Loading scene.glb — ${loaded.toFixed(1)} / ${(event.total / 1e6).toFixed(1)} MB`);
        } else if (event && event.loaded) {
          setProgress(null, `Loading scene.glb — ${(event.loaded / 1e6).toFixed(1)} MB`);
        }
      },
      (error) => reject(new Error(`scene.glb failed to load: ${error && error.message ? error.message : error}`)),
    );
  });
}

async function main() {
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();

  onResize();
  window.addEventListener('resize', onResize);
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', (event) => state.keys.delete(event.code));
  window.addEventListener('blur', () => state.keys.clear());
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('pointerlockchange', onPointerLockChange);
  ui.canvas.addEventListener('click', () => {
    if (state.mode !== 'orbit') {
      ui.canvas.requestPointerLock();
    }
  });
  ui.modeOrbit.addEventListener('click', () => setMode('orbit'));
  ui.modeFly.addEventListener('click', () => setMode('fly'));
  ui.modeWalk.addEventListener('click', () => setMode('walk'));
  ui.reset.addEventListener('click', resetView);
  ui.viewpoint.addEventListener('change', onViewpointChange);
  if (ui.floor) {
    ui.floor.addEventListener('change', onFloorChange);
  }
  if (ui.section) {
    ui.section.addEventListener('click', onSectionToggle);
  }

  try {
    const manifest = await fetchManifest();
    state.manifest = manifest;
    state.eyeHeight = Number.isFinite(manifest.eye_height) ? manifest.eye_height : DEFAULT_EYE_HEIGHT;
    state.floors = manifest.floors || [];
    state.rooms = manifest.rooms || [];
    state.roomById = new Map(state.rooms.map((room) => [room.id, room]));
    const gltf = await loadModel();

    const model = prepareModel(gltf.scene);
    scene.add(model);
    state.modelMaterials = collectMaterials(model);

    const box = new THREE.Box3().setFromObject(model);
    const bounds = state.bounds || boundsFromBox(box);
    applyBounds(bounds);

    setProgress(null, 'Building walk surfaces from the GLB…');
    await new Promise((resolve) => setTimeout(resolve, 0));
    try {
      state.walkIndex = buildWalkIndex(model);
    } catch (walkError) {
      console.error('[viewer] walk surface build failed; Walk uses the floor table only', walkError);
      state.walkIndex = null;
    }

    ui.title.textContent = manifest.title || 'RoomGen.Agent scene';
    ui.stats.textContent = formatStats(manifest) || 'GLB loaded';
    populateViewpoints(manifest);
    populateFloors(manifest);
    resetView();
    setMode('orbit');

    state.loaded = true;
    ui.loading.classList.add('is-hidden');
  } catch (error) {
    showError(`${error.message}\n\nServe a build directory that contains scene.glb:\n  bash scripts/scene3d_view.sh outputs/blender_agent/p1_settled`);
    return;
  }

  const animate = () => {
    requestAnimationFrame(animate);
    const delta = Math.min(clock.getDelta(), 0.1);
    if (state.mode === 'orbit') {
      controls.update();
    } else if (state.locked || state.keys.size) {
      updateFirstPerson(delta);
      updatePoseReadout();
    }
    renderer.render(scene, camera);
  };
  animate();
}

main();
