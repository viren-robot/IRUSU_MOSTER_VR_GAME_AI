import * as THREE from "three";
import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const FOV = 50;          // vertical field of view (deg)
const CAM_DIST = 9;      // distance from camera to the monster plane
const HAND_LOST_MS = 400;

// MediaPipe's 21-landmark hand skeleton (for drawing)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const video = $("cam");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const stageEl = $("stage");
const trackPill = $("trackPill");
const gesturePill = $("gesturePill");
const intro = $("intro");
const help = $("help");
const startBtn = $("startBtn");
const introStatus = $("introStatus");
const flipBtn = $("flipBtn");
const resetBtn = $("resetBtn");
const headsetBtn = $("headsetBtn");
const gameBtn = $("gameBtn");
const gamePill = $("gamePill");
const gameOver = $("gameover");
const gameOverText = $("gameOverText");
const playAgainBtn = $("playAgainBtn");
const dodgeBtn = $("dodgeBtn");

// ---------------------------------------------------------------------------
// Three.js setup
// ---------------------------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
stageEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 100);
camera.position.set(0, 0, 0);       // viewer sits at the centre of a 360° world

scene.add(new THREE.HemisphereLight(0xdbeafe, 0x1e1b4b, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
keyLight.position.set(3, 5, 6);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x7dd3fc, 8, 25, 2);
rimLight.position.set(-4, 2, -3);
scene.add(rimLight);

// ---------------------------------------------------------------------------
// 360° world: sky sphere + look-around (phone gyroscope, or drag fallback)
// ---------------------------------------------------------------------------
function buildSky() {
  const w = 2048;
  const h = 1024;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d");

  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0.0, "#05060f");
  grad.addColorStop(0.45, "#1b1240");
  grad.addColorStop(0.55, "#2a1b5c");
  grad.addColorStop(1.0, "#0a0a16");
  g.fillStyle = grad;
  g.fillRect(0, 0, w, h);

  // Soft aurora glows along the horizon
  for (const [x, rgb] of [[0.2, "125,211,252"], [0.55, "167,139,250"], [0.85, "244,114,182"]]) {
    const rg = g.createRadialGradient(x * w, h * 0.55, 0, x * w, h * 0.55, w * 0.18);
    rg.addColorStop(0, `rgba(${rgb},0.35)`);
    rg.addColorStop(1, `rgba(${rgb},0)`);
    g.fillStyle = rg;
    g.fillRect(0, 0, w, h);
  }

  // Stars (seeded so every visit looks the same)
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 1400; i++) {
    g.fillStyle = `rgba(255,255,255,${0.3 + rnd() * 0.7})`;
    g.fillRect(rnd() * w, rnd() * h * 0.9, rnd() < 0.1 ? 2 : 1, rnd() < 0.1 ? 2 : 1);
  }

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(60, 64, 32),
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, depthWrite: false })
  );
  scene.add(sky);
}
buildSky();

// Look-around state
const deviceQ = new THREE.Quaternion();
const screenQ = new THREE.Quaternion();
const euler = new THREE.Euler();
const zAxis = new THREE.Vector3(0, 0, 1);
const xToY = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5));
let deviceOn = false;
let dev = { alpha: 0, beta: 0, gamma: 0 };
let dragYaw = 0;
let dragPitch = 0;

window.addEventListener("deviceorientation", (e) => {
  if (e.alpha == null && e.beta == null) return;
  deviceOn = true;
  const d = THREE.MathUtils.degToRad;
  dev = { alpha: d(e.alpha || 0), beta: d(e.beta || 0), gamma: d(e.gamma || 0) };
});

async function requestOrientationPermission() {
  // iOS needs an explicit permission prompt; Android Chrome does not.
  const DOE = window.DeviceOrientationEvent;
  if (DOE && typeof DOE.requestPermission === "function") {
    try {
      await DOE.requestPermission();
    } catch (err) {
      console.warn("Motion permission not granted", err);
    }
  }
}

function updateCameraLook(dt) {
  if (deviceOn) {
    // Standard device-orientation to camera-quaternion mapping
    const angle = THREE.MathUtils.degToRad(window.screen?.orientation?.angle ?? 0);
    euler.set(dev.beta, dev.alpha, -dev.gamma, "YXZ");
    deviceQ.setFromEuler(euler);
    deviceQ.multiply(xToY);
    deviceQ.multiply(screenQ.setFromAxisAngle(zAxis, -angle));
  } else {
    euler.set(dragPitch, dragYaw, 0, "YXZ");
    deviceQ.setFromEuler(euler);
  }
  camera.quaternion.slerp(deviceQ, 1 - Math.exp(-12 * dt));
  camera.updateMatrixWorld();
}

function radialShadowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(0,0,0,0.6)");
  grd.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

// ---------------------------------------------------------------------------
// Monster (procedural, original design)
// ---------------------------------------------------------------------------
const mats = {
  skin: new THREE.MeshStandardMaterial({ color: 0x7c5cff, roughness: 0.4, metalness: 0.05, emissive: 0x1a0f40 }),
  horn: new THREE.MeshStandardMaterial({ color: 0xf5f3ff, roughness: 0.3 }),
  white: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x0b0b16, roughness: 0.6 }),
  glow: new THREE.MeshStandardMaterial({ color: 0x67e8f9, emissive: 0x22d3ee, emissiveIntensity: 1.2 }),
};

// Point on the front of a sphere of radius r, at (x, y).
function frontPos(x, y, r = 1.2) {
  const z = Math.sqrt(Math.max(0, r * r - x * x - y * y));
  return new THREE.Vector3(x, y, z);
}

// Orient an object so its local +Z faces outward from the body centre.
function faceOutward(obj, pos) {
  obj.position.copy(pos);
  obj.lookAt(pos.clone().multiplyScalar(2));
}

function buildMonster() {
  const root = new THREE.Group();   // position / scale / yaw driven by hand
  const body = new THREE.Group();   // idle bob, squash, roar
  root.add(body);

  // Lumpy torso
  const bodyGeo = new THREE.SphereGeometry(1.2, 64, 48);
  const p = bodyGeo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = 1 + 0.06 * Math.sin(v.x * 3.1) * Math.cos(v.y * 2.7) + 0.04 * Math.sin(v.z * 4.3 + v.x * 2.0);
    v.multiplyScalar(n);
    v.y *= 1.1;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  bodyGeo.computeVertexNormals();
  body.add(new THREE.Mesh(bodyGeo, mats.skin));

  // Eyes (pupils look toward the pointing finger)
  const pupils = [];
  const glowRing = [];
  for (const ex of [-0.42, 0.42]) {
    const eye = new THREE.Group();
    faceOutward(eye, frontPos(ex, 0.32));
    const white = new THREE.Mesh(new THREE.SphereGeometry(0.3, 32, 24), mats.white);
    white.scale.set(1, 1, 0.5);
    eye.add(white);
    const pupilPivot = new THREE.Group();
    pupilPivot.position.z = 0.12;
    const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.14, 24, 16), mats.dark);
    pupil.position.z = 0.04;
    pupilPivot.add(pupil);
    eye.add(pupilPivot);
    pupils.push(pupilPivot);
    // Thin glowing rim around each eye (intensity pulses on roar)
    const rim = new THREE.Mesh(new THREE.TorusGeometry(0.31, 0.03, 12, 40), mats.glow);
    rim.position.z = 0.02;
    eye.add(rim);
    glowRing.push(rim);
    body.add(eye);
  }

  // Mouth + teeth
  const mouth = new THREE.Group();
  faceOutward(mouth, frontPos(0, -0.42));
  const mouthMesh = new THREE.Mesh(new THREE.SphereGeometry(0.45, 32, 16), mats.dark);
  mouthMesh.scale.set(1, 0.25, 0.3);
  mouth.add(mouthMesh);
  const teeth = new THREE.Group();
  teeth.position.set(0, 0.1, 0.05);
  for (let i = -3; i <= 3; i++) {
    const tooth = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.16, 8), mats.horn);
    tooth.rotation.x = Math.PI;
    tooth.position.set(i * 0.1, 0, 0);
    teeth.add(tooth);
  }
  mouth.add(teeth);
  body.add(mouth);

  // Horns
  for (const side of [-1, 1]) {
    const horn = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.75, 16), mats.horn);
    horn.position.set(side * 0.8, 1.0, 0.2);
    horn.rotation.z = -side * 0.6;
    body.add(horn);
  }

  // Arms (pivot at the shoulder)
  const arms = [];
  for (const side of [-1, 1]) {
    const pivot = new THREE.Group();
    pivot.position.set(side * 1.05, -0.1, 0.1);
    const limb = new THREE.Mesh(new THREE.CapsuleGeometry(0.16, 0.9, 8, 16), mats.skin);
    limb.position.y = -0.6;
    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.24, 24, 16), mats.skin);
    hand.position.y = -1.25;
    pivot.add(limb, hand);
    pivot.userData.side = side;
    arms.push(pivot);
    body.add(pivot);
  }

  // Feet
  for (const side of [-1, 1]) {
    const foot = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mats.skin);
    foot.scale.set(0.4, 0.28, 0.45);
    foot.position.set(side * 0.55, -1.3, 0.4);
    body.add(foot);
  }

  // Ground shadow
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 3.2),
    new THREE.MeshBasicMaterial({ map: radialShadowTexture(), transparent: true, depthWrite: false })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = -1.85;
  root.add(shadow);

  return { root, body, mouth, mouthMesh, teeth, pupils, glowRing, arms };
}

const monster = buildMonster();
scene.add(monster.root);

// ---------------------------------------------------------------------------
// Target / current state (smoothed)
// ---------------------------------------------------------------------------
const makeState = () => ({ x: 0, y: 0, scale: 1, yaw: 0, pitch: 0, roar: 0, wave: 0, lookX: 0, lookY: 0 });
const target = makeState();
const cur = makeState();

function damp(a, b, lambda, dt) {
  return a + (b - a) * (1 - Math.exp(-lambda * dt));
}

// ---------------------------------------------------------------------------
// Hand tracking helpers
// ---------------------------------------------------------------------------
let mirror = true;               // front camera is mirrored
let lastHandAt = 0;
let handLandmarks = null;

const dist2d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const mapRange = (v, a, b, c, d) => c + ((clamp(v, a, b) - a) / (b - a)) * (d - c);

// Video is shown with object-fit: cover. Map normalized video coords to screen px.
function videoToScreen(nx, ny) {
  const sw = window.innerWidth;
  const sh = window.innerHeight;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { x: sw / 2, y: sh / 2 };
  const s = Math.max(sw / vw, sh / vh);
  const ox = (sw - vw * s) / 2;
  const oy = (sh - vh * s) / 2;
  return { x: ox + nx * vw * s, y: oy + ny * vh * s };
}

// Screen px -> world point on the plane the monster lives on (z = -CAM_DIST).
// Raycasting keeps this correct while the viewer turns around in 360°.
const raycaster = new THREE.Raycaster();
const ndcVec = new THREE.Vector2();
const monsterPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), CAM_DIST);
const hitPoint = new THREE.Vector3();
function screenToWorld(sx, sy) {
  ndcVec.set((sx / window.innerWidth) * 2 - 1, 1 - (sy / window.innerHeight) * 2);
  raycaster.setFromCamera(ndcVec, camera);
  if (!raycaster.ray.intersectPlane(monsterPlane, hitPoint)) return { x: 0, y: 0 };
  return { x: hitPoint.x, y: hitPoint.y };
}

// Convert a MediaPipe landmark (normalized, unmirrored) to screen px.
function landmarkToScreen(p) {
  const nx = mirror ? 1 - p.x : p.x;
  return videoToScreen(nx, p.y);
}

// Process one hand's 21 landmarks into control targets.
function handleHand(lm) {
  // Palm centre: average of wrist and the four knuckles (0, 5, 9, 13, 17)
  const palmIdx = [0, 5, 9, 13, 17];
  const palm = palmIdx.reduce(
    (acc, i) => ({ x: acc.x + lm[i].x / palmIdx.length, y: acc.y + lm[i].y / palmIdx.length }),
    { x: 0, y: 0 }
  );

  // Hand size reference (wrist -> middle knuckle) makes gestures scale-invariant
  const ref = dist2d(lm[0], lm[9]) || 1e-3;
  const pinchRatio = dist2d(lm[4], lm[8]) / ref;
  const tips = [8, 12, 16, 20];
  const openness = tips.reduce((s, i) => s + dist2d(lm[0], lm[i]), 0) / tips.length / ref;

  const fist = openness < 1.5;
  const openPalm = openness > 1.9;

  // Position: palm drives the monster's place in the scene
  const palmScreen = landmarkToScreen(palm);
  const world = screenToWorld(palmScreen.x, palmScreen.y);
  target.x = world.x;
  target.y = world.y;

  // Size: thumb-index spread (only when the hand is open enough to read it)
  if (!fist) {
    target.scale = mapRange(pinchRatio, 0.15, 1.2, 0.6, 1.8);
  }

  // Roar when fisted
  target.roar = fist ? 1 : 0;

  // Wave when palm is open
  target.wave = openPalm ? 1 : 0;

  // Look direction from palm -> index fingertip, normalized by hand size
  const dxN = (mirror ? -1 : 1) * (lm[8].x - palm.x) / (ref * 1.6);
  const dyN = (lm[8].y - palm.y) / (ref * 1.6);
  target.lookX = clamp(dxN, -1, 1);
  target.lookY = clamp(-dyN, -1, 1);
  target.yaw = target.lookX * 0.6;
  target.pitch = target.lookY * 0.3;

  // Status labels
  let gesture = "Hand";
  if (fist) gesture = "Fist · roar";
  else if (openPalm) gesture = "Open palm · wave";
  else if (pinchRatio < 0.3) gesture = "Pinch · small";
  else gesture = `Size ${target.scale.toFixed(2)}×`;
  gesturePill.textContent = gesture;
}

function setIdleTarget(t) {
  // Drift gently when no hand is tracked
  target.x = Math.sin(t * 0.3) * 0.6;
  target.y = 0.3 + Math.sin(t * 0.5) * 0.25;
  target.scale = 1;
  target.roar = 0;
  target.wave = 0;
  target.lookX = Math.sin(t * 0.7) * 0.4;
  target.lookY = Math.sin(t * 0.45) * 0.2;
  target.yaw = Math.sin(t * 0.4) * 0.15;
  target.pitch = 0;
}

// ---------------------------------------------------------------------------
// Game mode: "Orb Hunt". Steer the monster over glowing orbs before time runs out.
// Gold orbs are worth 3 points, normal orbs 1. Best score is saved on the device.
// ---------------------------------------------------------------------------
const GAME_TIME = 60;   // seconds per round
const ORB_COUNT = 10;
const orbGeo = new THREE.SphereGeometry(0.28, 24, 16);
const orbMat = new THREE.MeshStandardMaterial({ color: 0xfde68a, emissive: 0xf59e0b, emissiveIntensity: 1.4, roughness: 0.2 });
const goldMat = new THREE.MeshStandardMaterial({ color: 0xfef08a, emissive: 0xfacc15, emissiveIntensity: 2.2, roughness: 0.2 });
const orbs = [];
const game = { active: false, score: 0, timeLeft: GAME_TIME };
let bestScore = 0;
let lastHudText = "";
try {
  bestScore = Number(localStorage.getItem("irusu-best") || 0);
} catch (_) {
  bestScore = 0;
}

function spawnOrb(o) {
  o.gold = Math.random() < 0.2;
  o.x = (Math.random() * 2 - 1) * 4;   // spread across the visible arc
  o.y = (Math.random() * 2 - 1) * 2;
  o.alive = true;
  o.respawnAt = 0;
  o.mesh.material = o.gold ? goldMat : orbMat;
  o.mesh.position.set(o.x, o.y, -CAM_DIST);
  o.mesh.visible = true;
}

for (let i = 0; i < ORB_COUNT; i++) {
  const mesh = new THREE.Mesh(orbGeo, orbMat);
  mesh.visible = false;
  scene.add(mesh);
  orbs.push({ mesh, x: 0, y: 0, alive: false, gold: false, respawnAt: 0 });
}

function updateGameHud() {
  const text = `${Math.max(0, Math.ceil(game.timeLeft))}s · ${game.score} pts · best ${bestScore}`;
  if (text !== lastHudText) {
    gamePill.textContent = text;
    lastHudText = text;
  }
}

function startGame() {
  lastMode = "orb";
  dodge.active = false;
  monster.root.visible = true;
  hideRocks();
  game.active = true;
  game.score = 0;
  game.timeLeft = GAME_TIME;
  for (const o of orbs) spawnOrb(o);
  gameOver.classList.add("hidden");
  gamePill.classList.remove("hidden");
  lastHudText = "";
  updateGameHud();
}

function endGame() {
  game.active = false;
  game.timeLeft = 0;
  for (const o of orbs) {
    o.alive = false;
    o.mesh.visible = false;
  }
  if (game.score > bestScore) {
    bestScore = game.score;
    try {
      localStorage.setItem("irusu-best", String(bestScore));
    } catch (_) {}
  }
  updateGameHud();
  gameOverText.textContent = `You scored ${game.score}. Best: ${bestScore}.`;
  gameOver.classList.remove("hidden");
}

function updateGame(dt, t) {
  if (!game.active) return;
  game.timeLeft -= dt;
  if (game.timeLeft <= 0) {
    endGame();
    return;
  }

  const reach = 1.3 * cur.scale;
  for (const o of orbs) {
    if (!o.alive) {
      if (t >= o.respawnAt) spawnOrb(o);
      continue;
    }
    o.mesh.rotation.y += dt * 2;
    o.mesh.scale.setScalar(1 + Math.sin(t * 4 + o.x) * 0.12);
    if (Math.hypot(o.x - cur.x, o.y - cur.y) < reach) {
      o.alive = false;
      o.mesh.visible = false;
      o.respawnAt = t + 0.8;
      game.score += o.gold ? 3 : 1;
    }
  }
  updateGameHud();
}

function toggleGame() {
  if (game.active) endGame();
  startGame();
}

// ---------------------------------------------------------------------------
// Game 2: "Dodge Storm". Rocks fall toward you; steer the monster out of their path.
// You have 3 lives and a short invulnerable blink after each hit.
// Score = seconds survived. Best score is saved on the device.
// ---------------------------------------------------------------------------
const ROCK_COUNT = 14;
const rockGeo = new THREE.DodecahedronGeometry(0.45, 0);
const rockMat = new THREE.MeshStandardMaterial({ color: 0x3f3f58, emissive: 0x7f1d1d, emissiveIntensity: 0.6, roughness: 0.7 });
const rocks = [];
const dodge = { active: false, lives: 3, survived: 0, invuln: 0, spawnTimer: 0 };
let bestDodge = 0;
let lastMode = "orb";
try {
  bestDodge = Number(localStorage.getItem("irusu-dodge-best") || 0);
} catch (_) {
  bestDodge = 0;
}
for (let i = 0; i < ROCK_COUNT; i++) {
  const mesh = new THREE.Mesh(rockGeo, rockMat);
  mesh.visible = false;
  scene.add(mesh);
  rocks.push({ mesh, x: 0, y: 0, vy: 0, alive: false });
}

function hideRocks() {
  for (const r of rocks) {
    r.alive = false;
    r.mesh.visible = false;
  }
}

function dodgeHud() {
  const text = `${dodge.lives} lives · ${Math.floor(dodge.survived)}s · best ${bestDodge}`;
  if (text !== lastHudText) {
    gamePill.textContent = text;
    lastHudText = text;
  }
}

function startDodge() {
  lastMode = "dodge";
  game.active = false;
  for (const o of orbs) {
    o.alive = false;
    o.mesh.visible = false;
  }
  dodge.active = true;
  dodge.lives = 3;
  dodge.survived = 0;
  dodge.invuln = 0;
  dodge.spawnTimer = 0;
  hideRocks();
  gameOver.classList.add("hidden");
  gamePill.classList.remove("hidden");
  lastHudText = "";
  dodgeHud();
}

function endDodge() {
  dodge.active = false;
  monster.root.visible = true;
  hideRocks();
  const score = Math.floor(dodge.survived);
  if (score > bestDodge) {
    bestDodge = score;
    try {
      localStorage.setItem("irusu-dodge-best", String(bestDodge));
    } catch (_) {}
  }
  lastHudText = "";
  gamePill.textContent = `Dodge over · ${score}s`;
  gameOverText.textContent = `You survived ${score}s. Best: ${bestDodge}s.`;
  gameOver.classList.remove("hidden");
}

function updateDodge(dt, t) {
  if (!dodge.active) return;
  dodge.survived += dt;
  dodge.invuln = Math.max(0, dodge.invuln - dt);
  // Monster blinks while invulnerable after a hit
  monster.root.visible = dodge.invuln <= 0 || Math.floor(t * 10) % 2 === 0;

  // Rocks arrive faster the longer you survive
  dodge.spawnTimer -= dt;
  if (dodge.spawnTimer <= 0) {
    const r = rocks.find((rock) => !rock.alive);
    if (r) {
      r.alive = true;
      r.x = (Math.random() * 2 - 1) * 4.5;
      r.y = 4.5;
      r.vy = -(2.5 + dodge.survived * 0.05);
      r.mesh.position.set(r.x, r.y, -CAM_DIST);
      r.mesh.visible = true;
    }
    dodge.spawnTimer = Math.max(0.35, 1.1 - dodge.survived * 0.01);
  }

  const hitRadius = 0.9 * cur.scale + 0.45;
  for (const r of rocks) {
    if (!r.alive) continue;
    r.y += r.vy * dt;
    r.mesh.position.y = r.y;
    r.mesh.rotation.x += dt * 2;
    r.mesh.rotation.y += dt * 1.3;
    if (r.y < -4) {
      r.alive = false;
      r.mesh.visible = false;
      continue;
    }
    if (dodge.invuln <= 0 && Math.hypot(r.x - cur.x, r.y - cur.y) < hitRadius) {
      dodge.lives -= 1;
      dodge.invuln = 1.2;
      r.alive = false;
      r.mesh.visible = false;
      if (dodge.lives <= 0) {
        endDodge();
        return;
      }
    }
  }
  dodgeHud();
}

function toggleDodge() {
  if (dodge.active) endDodge();
  startDodge();
}

// ---------------------------------------------------------------------------
// Headset mode: side-by-side stereo for a phone VR viewer (Cardboard-style)
// ---------------------------------------------------------------------------
const stereo = new THREE.StereoCamera();
stereo.eyeSep = 0.064; // distance between eyes in scene units (about 64 mm)
let headsetOn = false;

async function setHeadset(on) {
  headsetOn = on;
  document.body.classList.toggle("headset", on);
  if (on) {
    try {
      await document.documentElement.requestFullscreen?.();
      await screen.orientation.lock("landscape");
    } catch (err) {
      console.warn("Fullscreen or landscape lock unavailable", err);
    }
    // The back camera faces forward, so it can see hands held in front of the viewer
    if (facingMode === "user") {
      facingMode = "environment";
      try {
        await startCamera();
      } catch (err) {
        console.warn("Back camera unavailable", err);
      }
    }
  } else {
    try {
      screen.orientation.unlock();
      await document.exitFullscreen?.();
    } catch (_) {}
  }
}

// Draw the scene once (normal) or twice (left and right eye halves)
function renderFrame() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (!headsetOn) {
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    renderer.render(scene, camera);
    return;
  }
  stereo.update(camera);
  renderer.setScissorTest(true);
  renderer.setViewport(0, 0, w / 2, h);
  renderer.setScissor(0, 0, w / 2, h);
  renderer.render(scene, stereo.cameraL);
  renderer.setViewport(w / 2, 0, w / 2, h);
  renderer.setScissor(w / 2, 0, w / 2, h);
  renderer.render(scene, stereo.cameraR);
}

// Drag fallback: drag the screen to look around when the phone has no gyroscope
let pointerActive = false;
let lastPX = 0;
let lastPY = 0;
window.addEventListener("pointerdown", (e) => {
  if (e.target.closest("button")) return;
  pointerActive = true;
  lastPX = e.clientX;
  lastPY = e.clientY;
});
window.addEventListener("pointerup", () => (pointerActive = false));
window.addEventListener("pointermove", (e) => {
  if (!pointerActive) return;
  dragYaw -= (e.clientX - lastPX) * 0.005;
  dragPitch = clamp(dragPitch - (e.clientY - lastPY) * 0.005, -1.2, 1.2);
  lastPX = e.clientX;
  lastPY = e.clientY;
});

// ---------------------------------------------------------------------------
// Monster animation
// ---------------------------------------------------------------------------
function updateMonster(dt, t) {
  const k = 8; // smoothing speed
  for (const key of ["x", "y", "scale", "yaw", "pitch", "roar", "wave", "lookX", "lookY"]) {
    cur[key] = damp(cur[key], target[key], k, dt);
  }

  monster.root.position.set(cur.x, cur.y, -CAM_DIST);
  monster.root.scale.setScalar(cur.scale);
  monster.root.rotation.set(cur.pitch, cur.yaw, 0);

  // Idle bob + squash; roar pumps the body
  const bob = Math.sin(t * 2.0) * 0.06;
  const roar = cur.roar;
  monster.body.position.y = bob * (1 - roar * 0.5);
  monster.body.scale.set(
    1 + roar * 0.08 + Math.sin(t * 2) * 0.012,
    1 - roar * 0.06 + Math.sin(t * 2) * 0.012,
    1 + roar * 0.08
  );

  // Mouth opens on roar, with a small idle chomp
  const chomp = Math.max(0, Math.sin(t * 1.3)) * 0.08;
  monster.mouthMesh.scale.y = 0.25 + roar * 0.9 + chomp;
  monster.teeth.position.y = 0.1 + roar * 0.22 + chomp * 0.5;

  // Eyes glow harder when roaring
  for (const rim of monster.glowRing) {
    rim.material.emissiveIntensity = 1.2 + roar * 3;
  }

  // Pupils look toward the pointing finger
  for (const pupil of monster.pupils) {
    pupil.position.x = cur.lookX * 0.07;
    pupil.position.y = cur.lookY * 0.07;
  }

  // Arms: raise on roar, wave with an open palm
  for (const arm of monster.arms) {
    const side = arm.userData.side;
    const base = side * (0.35 + roar * 2.2);
    const wave = Math.sin(t * 6 + side) * 0.35 * cur.wave;
    arm.rotation.z = base + wave;
  }
}

// ---------------------------------------------------------------------------
// Overlay drawing (21-point skeleton)
// ---------------------------------------------------------------------------
function resizeOverlay() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  overlay.width = Math.floor(window.innerWidth * dpr);
  overlay.height = Math.floor(window.innerHeight * dpr);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawHand(lm) {
  octx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  if (!lm) return;
  const pts = lm.map(landmarkToScreen);

  octx.lineWidth = 3;
  octx.strokeStyle = "rgba(125, 211, 252, 0.85)";
  octx.lineCap = "round";
  octx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    octx.moveTo(pts[a].x, pts[a].y);
    octx.lineTo(pts[b].x, pts[b].y);
  }
  octx.stroke();

  for (let i = 0; i < pts.length; i++) {
    const tip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
    octx.beginPath();
    octx.arc(pts[i].x, pts[i].y, tip ? 6 : 4, 0, Math.PI * 2);
    octx.fillStyle = tip ? "#a78bfa" : "#ffffff";
    octx.fill();
  }
}

// ---------------------------------------------------------------------------
// Camera + model
// ---------------------------------------------------------------------------
let stream = null;
let facingMode = "user";
let handLandmarker = null;

async function startCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  mirror = facingMode === "user";
  video.classList.toggle("mirror", mirror);
  video.classList.add("live");
}

async function loadHandModel() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU", err);
    return await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
let lastFrame = performance.now();
let lastVideoTime = -1;
let running = false;

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const t = now / 1000;

  if (handLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = handLandmarker.detectForVideo(video, now);
    const lm = res.landmarks && res.landmarks[0];
    if (lm && lm.length === 21) {
      handLandmarks = lm;
      lastHandAt = now;
      handleHand(lm);
      drawHand(lm);
      trackPill.textContent = "Hand: 21/21";
      trackPill.className = "pill ok";
    }
  }

  if (handLandmarks && now - lastHandAt > HAND_LOST_MS) {
    handLandmarks = null;
    drawHand(null);
    gesturePill.textContent = "Idle";
    trackPill.textContent = "Hand: lost";
    trackPill.className = "pill warn";
  }

  if (!handLandmarks && !pointerActive) {
    setIdleTarget(t);
    if (game.active || dodge.active) {
      // No hand: the monster follows the centre of your view (turn the phone to steer)
      const g = screenToWorld(window.innerWidth / 2, window.innerHeight / 2);
      target.x = g.x;
      target.y = g.y;
    }
  }

  updateGame(dt, t);
  updateDodge(dt, t);
  updateCameraLook(dt);
  updateMonster(dt, t);
  renderFrame();
}

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  resizeOverlay();
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  introStatus.textContent = "Starting camera…";
  try {
    await requestOrientationPermission();
    await startCamera();
    introStatus.textContent = "Loading hand model…";
    handLandmarker = await loadHandModel();
    introStatus.textContent = "Ready";
    intro.classList.add("hidden");
    help.classList.remove("hidden");
    trackPill.textContent = "Hand: searching";
    trackPill.className = "pill";
  } catch (err) {
    console.error(err);
    introStatus.textContent =
      "Camera or hand model unavailable (" + (err.name || err.message) + "). You can still drag the monster.";
    startBtn.textContent = "Continue without camera";
    startBtn.disabled = false;
    intro.classList.add("hidden");
    help.classList.remove("hidden");
    trackPill.textContent = "Hand: off";
  }
});

flipBtn.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  try {
    await startCamera();
  } catch (err) {
    console.error(err);
    trackPill.textContent = "Camera switch failed";
    trackPill.className = "pill warn";
  }
});

resetBtn.addEventListener("click", () => {
  target.x = 0;
  target.y = 0;
  target.scale = 1;
});

headsetBtn.addEventListener("click", () => setHeadset(!headsetOn));
gameBtn.addEventListener("click", toggleGame);
dodgeBtn.addEventListener("click", toggleDodge);
playAgainBtn.addEventListener("click", () => (lastMode === "dodge" ? startDodge() : startGame()));

window.addEventListener("resize", resize);
window.addEventListener("orientationchange", resize);

resize();
requestAnimationFrame(frame);
