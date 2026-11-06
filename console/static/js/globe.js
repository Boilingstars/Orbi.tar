import * as THREE from '../vendor/three.module.js';
import { fetchOrbits, onTrack, primeTracks, trackSnapshot } from './api.js?v=whatif1';

const EARTH_M = 6371000;
const COLORS = {
  job: new THREE.Color(0x3ddc6a),
  idle: new THREE.Color(0xe8ebef),
  calibrate: new THREE.Color(0x8b7fd4),
};

let view = null;

export function syncGlobe(host, state, actions) {
  if (!view || view.host !== host) view = createGlobe(host);
  const step = state.previewStep ?? state.step;
  view.pending = { state, actions, step };
  paintHud(view, state, step, view.error || '');
  bindStream(state);
  applyStep(state, step);
  resize(view);
}

function createGlobe(host) {
  host.replaceChildren();
  const canvas = document.createElement('canvas');
  canvas.className = 'globe-canvas';
  const hud = document.createElement('div');
  hud.className = 'hud';
  host.append(canvas, hud);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setClearColor(0x05070c, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 80);
  scene.add(new THREE.AmbientLight(0x8aa0c0, 0.18));
  const sun = new THREE.DirectionalLight(0xfff4e0, 2.4);
  sun.target.position.set(0, 0, 0);
  scene.add(sun);
  scene.add(sun.target);

  const earthMat = new THREE.MeshPhongMaterial({
    color: 0x1a4f86,
    emissive: 0x061018,
    shininess: 8,
    specular: new THREE.Color(0x223344),
  });
  const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 48), earthMat);
  scene.add(earth);
  const earthTex = new THREE.TextureLoader().load(
    new URL('../vendor/earth.jpg', import.meta.url).href,
    (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
      earthMat.map = tex;
      earthMat.emissiveMap = tex;
      earthMat.emissive = new THREE.Color(0xffffff);
      earthMat.emissiveIntensity = 0.18;
      earthMat.color.set(0xffffff);
      earthMat.needsUpdate = true;
    },
  );
  earthTex.colorSpace = THREE.SRGBColorSpace;
  scene.add(starField());

  const orbits = new THREE.Group();
  const points = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 11, sizeAttenuation: false, vertexColors: true, depthTest: true }),
  );
  scene.add(orbits);
  scene.add(points);

  const next = {
    host,
    canvas,
    hud,
    renderer,
    scene,
    camera,
    sun,
    orbits,
    points,
    ids: [],
    yaw: 0.6,
    pitch: 0.35,
    dragging: false,
    moved: 0,
    error: '',
    frame: null,
    pending: null,
  };
  bindPointer(next);
  const observer = new ResizeObserver(() => resize(next));
  observer.observe(host);
  resize(next);
  const loop = () => {
    next.frame = requestAnimationFrame(loop);
    placeCamera(next);
    next.renderer.render(scene, camera);
  };
  loop();
  return next;
}

function starField() {
  const n = 900;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    const r = 18;
    pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.cos(phi);
    pos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(
    geo,
    new THREE.PointsMaterial({ color: 0xffffff, size: 1.2, sizeAttenuation: false }),
  );
}

function bindPointer(next) {
  let lastX = 0;
  let lastY = 0;
  next.canvas.addEventListener('pointerdown', (event) => {
    next.dragging = true;
    next.moved = 0;
    lastX = event.clientX;
    lastY = event.clientY;
    next.canvas.classList.add('is-drag');
    next.canvas.setPointerCapture(event.pointerId);
  });
  next.canvas.addEventListener('pointermove', (event) => {
    if (!next.dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    next.moved += Math.abs(dx) + Math.abs(dy);
    next.yaw -= dx * 0.005;
    next.pitch = Math.max(-1.2, Math.min(1.2, next.pitch + dy * 0.005));
  });
  const end = (event) => {
    if (!next.dragging) return;
    next.dragging = false;
    next.canvas.classList.remove('is-drag');
    if (next.moved < 5) pick(next, event);
  };
  next.canvas.addEventListener('pointerup', end);
  next.canvas.addEventListener('pointercancel', end);
}

function pick(next, event) {
  if (!next.pending || next.ids.length === 0) return;
  const rect = next.canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const ray = new THREE.Raycaster();
  ray.params.Points.threshold = 0.08;
  ray.setFromCamera(ndc, next.camera);
  const hits = ray.intersectObject(next.points);
  if (!hits.length) return;
  const id = next.ids[hits[0].index];
  if (id) next.pending.actions.selectSatellite(id);
}

function placeCamera(next) {
  const dist = 3.15;
  const cp = Math.cos(next.pitch);
  next.camera.position.set(
    dist * cp * Math.sin(next.yaw),
    dist * Math.sin(next.pitch),
    dist * cp * Math.cos(next.yaw),
  );
  next.camera.lookAt(0, 0, 0);
}

function resize(next) {
  const w = next.host.clientWidth || 1;
  const h = next.host.clientHeight || 1;
  next.renderer.setSize(w, h, false);
  next.camera.aspect = w / h;
  next.camera.updateProjectionMatrix();
}

function paintHud(next, state, step, error) {
  const selected = state.selection.kind === 'satellite' ? state.selection.id : '—';
  const note = error ? ` · ${error}` : '';
  next.hud.innerHTML = `<strong>Глобус</strong> · шаг <span class="num">${step}</span> · выбран <span class="sel">${selected}</span>${note}<br>зелёный — задание, белый — ожидание, фиолетовый — калибровка · тусклые — сейчас нельзя взять`;
}

function bindStream(state) {
  const snap = trackSnapshot() || (primeTracks(), trackSnapshot());
  if (view.track === snap && view.boundScenario === state.scenario.id) return;
  if (view.unsub) view.unsub();
  view.boundScenario = state.scenario.id;
  view.track = snap;
  view.linesReady = false;
  clearGroup(view.orbits);
  view.unsub = onTrack((msg) => {
    if (!view || trackSnapshot() !== view.track || !view.pending) return;
    if (view.pending.state.scenario.id !== view.boundScenario) return;
    if (msg.type === 'orbits') buildLines(view, msg);
    if (msg.type === 'error') {
      loadStepOrbits(view.pending.state, view.pending.state.previewStep ?? view.pending.state.step);
      return;
    }
    const step = view.pending.state.previewStep ?? view.pending.state.step;
    if (msg.type === 'frame' && msg.step === step) {
      view.error = '';
      paintFrame(view, view.pending.state, msg, step);
    }
  });
}

function applyStep(state, step) {
  const snap = trackSnapshot();
  if (view && snap) {
    if (snap.orbits && !view.linesReady) buildLines(view, snap.orbits);
    const frame = snap.frames.get(step);
    if (frame) {
      view.error = '';
      paintFrame(view, state, frame, step);
      return;
    }
  }
  loadStepOrbits(state, step);
}

function loadStepOrbits(state, step) {
  if (!view) return;
  if (view.orbitReq === step) return;
  view.orbitReq = step;
  fetchOrbits(step).then((data) => {
    if (!view?.pending) return;
    const now = view.pending.state.previewStep ?? view.pending.state.step;
    if (now !== step) return;
    view.error = '';
    paint(view, view.pending.state, data, step);
  }).catch(() => {
    if (!view?.pending) return;
    view.error = 'орбиты недоступны';
    paintHud(view, view.pending.state, step, view.error);
  });
}

function buildLines(next, msg) {
  clearGroup(next.orbits);
  (msg.path || []).forEach((line) => {
    const linePos = [];
    line.forEach((p) => {
      const v = toThree(p[0], p[1], p[2]);
      linePos.push(v.x, v.y, v.z);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
    next.orbits.add(new THREE.Line(
      geo,
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }),
    ));
  });
  next.linesReady = true;
}

function paintFrame(next, state, frame, step) {
  const ids = (trackSnapshot() && trackSnapshot().ids) || [];
  const byId = new Map(state.satellites.map((sat) => [sat.id, sat]));
  const positions = [];
  const colors = [];
  frame.pos.forEach((p, i) => {
    const here = toThree(p[0], p[1], p[2]);
    positions.push(here.x, here.y, here.z);
    const color = colorFor(byId.get(ids[i]), step, state.step);
    colors.push(color.r, color.g, color.b);
  });
  setColoredPoints(next.points, positions, colors);
  next.ids = ids;
  const lit = ecefToThree(frame.sun[0], frame.sun[1], frame.sun[2]).normalize().multiplyScalar(8);
  next.sun.position.copy(lit);
}

function paint(next, state, data, step) {
  const byId = new Map(state.satellites.map((sat) => [sat.id, sat]));
  clearGroup(next.orbits);
  const positions = [];
  const colors = [];
  const ids = [];
  data.satellites.forEach((sat) => {
    const linePos = [];
    sat.orbit.forEach((p) => {
      const v = toThree(p.lat, p.lon, p.alt_m);
      linePos.push(v.x, v.y, v.z);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
    next.orbits.add(new THREE.Line(
      geo,
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }),
    ));
    const here = toThree(sat.lat, sat.lon, sat.alt_m);
    positions.push(here.x, here.y, here.z);
    const color = colorFor(byId.get(sat.id), step, state.step);
    colors.push(color.r, color.g, color.b);
    ids.push(sat.id);
  });
  setColoredPoints(next.points, positions, colors);
  next.ids = ids;
  const sun = data.sun_ecef;
  const lit = ecefToThree(sun[0], sun[1], sun[2]).normalize().multiplyScalar(8);
  next.sun.position.copy(lit);
}

function colorFor(sat, step, currentStep) {
  const action = actionAt(sat, step, currentStep);
  const base = COLORS[action] || COLORS.idle;
  const busy = action === 'job' || action === 'calibrate';
  if (sat && sat.assignable === false && !busy) {
    return base.clone().lerp(new THREE.Color(0x3a404c), 0.72);
  }
  return base;
}

function actionAt(sat, step, currentStep) {
  if (!sat) return 'idle';
  const row = (sat.action_tape || []).find((item) => item.step === step);
  if (row) return row.action;
  if (step === currentStep) return sat.action || 'idle';
  return 'idle';
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry.dispose();
    child.material.dispose();
  }
}

function setColoredPoints(points, positions, colors) {
  const count = positions.length / 3;
  const geo = points.geometry;
  const pos = geo.getAttribute('position');
  if (!pos || pos.count !== count) {
    geo.dispose();
    const fresh = new THREE.BufferGeometry();
    fresh.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    fresh.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    points.geometry = fresh;
    return;
  }
  pos.array.set(positions);
  pos.needsUpdate = true;
  const col = geo.getAttribute('color');
  col.array.set(colors);
  col.needsUpdate = true;
  geo.computeBoundingSphere();
}

function toThree(latDeg, lonDeg, altM) {
  const lat = (latDeg * Math.PI) / 180;
  const lon = (lonDeg * Math.PI) / 180;
  const r = 1.025 + altM / EARTH_M;
  const clat = Math.cos(lat);
  return ecefToThree(r * clat * Math.cos(lon), r * clat * Math.sin(lon), r * Math.sin(lat));
}

function ecefToThree(x, y, z) {
  return new THREE.Vector3(x, z, -y);
}
