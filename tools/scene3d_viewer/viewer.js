// RoomGen.Agent 3D scene viewer: loads <build>/scene.glb with three.js and
// offers an orbit camera plus first-person Fly/Walk navigation, with jumps to
// the panoramas' camera poses recorded in scene3d_manifest.json.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const GLB_URL = '/scene.glb';
const MANIFEST_URL = '/manifest.json';

const DEFAULT_EYE_HEIGHT = 1.5;
const WALK_SPEED = 2.8;   // m/s
const FLY_SPEED = 4.0;    // m/s
const SPRINT_FACTOR = 2.2;
const LOOK_SPEED = 0.0022;
const PITCH_LIMIT = Math.PI / 2 - 0.02;

const ui = {
  canvas: document.getElementById('scene-canvas'),
  title: document.getElementById('scene-title'),
  stats: document.getElementById('scene-stats'),
  viewpoint: document.getElementById('viewpoint-select'),
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
};

const renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

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
      camera.position.y = state.eyeHeight;
    }
    applyLook();
    ui.pointerHint.hidden = state.locked;
    ui.hint.textContent = 'Fly/Walk: WASD moves, mouse looks, Shift sprints, Space/C up-down (fly), Esc releases the mouse';
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
  if (move.lengthSq() > 0) {
    camera.position.addScaledVector(move.normalize(), speed);
  }
  if (state.mode === 'walk') {
    camera.position.y = state.eyeHeight;
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
  if (state.mode === 'orbit') {
    const target = position.clone().addScaledVector(forward, Math.max(1.5, state.radius * 0.35));
    setOrbitPose(position, target);
  } else {
    camera.position.copy(position);
    if (state.mode === 'walk') {
      camera.position.y = state.eyeHeight;
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
  if (state.mode === 'orbit') {
    setOrbitPose(position.clone().add(new THREE.Vector3(state.radius * 0.5, state.radius * 0.4, state.radius * 0.5)), target);
  } else {
    camera.position.copy(position);
    if (state.mode === 'walk') {
      camera.position.y = state.eyeHeight;
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
  } else {
    const pose = overviewPose();
    camera.position.copy(pose.position);
    const direction = new THREE.Vector3().subVectors(pose.target, pose.position).normalize();
    state.yaw = Math.atan2(-direction.x, -direction.z);
    state.pitch = Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1));
    if (state.mode === 'walk') {
      camera.position.y = state.eyeHeight;
      state.pitch = 0;
    }
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

  const cameras = manifest.cameras || [];
  if (cameras.length) {
    const group = document.createElement('optgroup');
    group.label = 'Panorama viewpoints';
    for (const entry of cameras) {
      const option = document.createElement('option');
      option.value = `camera:${entry.name}`;
      option.textContent = entry.room ? `${entry.name} — ${entry.room}` : entry.name;
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
      option.textContent = room.type ? `${room.id} — ${room.type}` : room.id;
      group.append(option);
    }
    select.append(group);
  }
  select.disabled = false;
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

  try {
    const manifest = await fetchManifest();
    state.manifest = manifest;
    state.eyeHeight = Number.isFinite(manifest.eye_height) ? manifest.eye_height : DEFAULT_EYE_HEIGHT;
    const gltf = await loadModel();

    const model = prepareModel(gltf.scene);
    scene.add(model);

    const box = new THREE.Box3().setFromObject(model);
    const bounds = state.bounds || boundsFromBox(box);
    applyBounds(bounds);

    ui.title.textContent = manifest.title || 'RoomGen.Agent scene';
    ui.stats.textContent = formatStats(manifest) || 'GLB loaded';
    populateViewpoints(manifest);
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
