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
camera.position.set(0, 0.4, CAM_DIST);

scene.add(new THREE.HemisphereLight(0xdbeafe, 0x1e1b4b, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
keyLight.position.set(3, 5, 6);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x7dd3fc, 8, 25, 2);
rimLight.position.set(-4, 2, -3);
scene.add(rimLight);

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

// Screen px -> world coords on the z = 0 plane the monster lives on.
function screenToWorld(sx, sy) {
  const sw = window.innerWidth;
  const sh = window.innerHeight;
  const halfH = Math.tan((FOV * Math.PI) / 360) * CAM_DIST;
  const halfW = halfH * (sw / sh);
  return {
    x: ((sx / sw) * 2 - 1) * halfW,
    y: (1 - (sy / sh) * 2) * halfH,
  };
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

// Pointer fallback (no camera): drag the monster around
let pointerActive = false;
window.addEventListener("pointerdown", (e) => {
  if (e.target.closest("button")) return;
  pointerActive = true;
});
window.addEventListener("pointerup", () => (pointerActive = false));
window.addEventListener("pointermove", (e) => {
  if (!pointerActive || handLandmarks) return;
  const w = screenToWorld(e.clientX, e.clientY);
  target.x = w.x;
  target.y = w.y;
});

// ---------------------------------------------------------------------------
// Monster animation
// ---------------------------------------------------------------------------
function updateMonster(dt, t) {
  const k = 8; // smoothing speed
  for (const key of ["x", "y", "scale", "yaw", "pitch", "roar", "wave", "lookX", "lookY"]) {
    cur[key] = damp(cur[key], target[key], k, dt);
  }

  monster.root.position.set(cur.x, cur.y, 0);
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
  }

  updateMonster(dt, t);
  renderer.render(scene, camera);
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

window.addEventListener("resize", resize);
window.addEventListener("orientationchange", resize);

resize();
requestAnimationFrame(frame);
