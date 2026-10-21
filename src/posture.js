// Desk posture buddy: webcam upper-body pose -> VRM that slumps with you and reminds you to sit up.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { FilesetResolver, PoseLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";

const $ = (id) => document.getElementById(id);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const deg = THREE.MathUtils.degToRad;

// --- scene ------------------------------------------------------------------

const canvas = $("three");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
const controls = new OrbitControls(camera, canvas);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666677, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(1, 3, 2);
scene.add(sun);

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// --- VRM --------------------------------------------------------------------

const loader = new GLTFLoader();
loader.register((p) => new VRMLoaderPlugin(p));
let vrm = null;
let rest = null;

const node = (name) => vrm.humanoid.getNormalizedBoneNode(name);
const worldPos = (o) => o.getWorldPosition(new THREE.Vector3());
const worldQuat = (o) => o.getWorldQuaternion(new THREE.Quaternion());

async function loadVRM(url) {
  const gltf = await loader.loadAsync(url);
  if (vrm) {
    scene.remove(vrm.scene);
    VRMUtils.deepDispose(vrm.scene);
  }
  vrm = gltf.userData.vrm;
  VRMUtils.rotateVRM0(vrm);
  scene.add(vrm.scene);
  vrm.humanoid.resetNormalizedPose();
  vrm.scene.updateMatrixWorld(true);
  const left = worldPos(node("leftUpperLeg")).sub(worldPos(node("rightUpperLeg"))).setY(0).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const forward = new THREE.Vector3().crossVectors(left, up).normalize();
  const quats = {};
  for (const b of ["spine", "neck", "head", "leftUpperArm", "rightUpperArm", "leftShoulder", "rightShoulder"]) {
    if (node(b)) quats[b] = worldQuat(node(b));
  }
  rest = { left, up, forward, quats, head: worldPos(node("head")) };
  // Frame the upper body, like a buddy sitting across the desk.
  const target = rest.head.clone().setY(rest.head.y - 0.3);
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(forward, 3.2 * Math.max(1, 1 / camera.aspect));
  controls.update();
  applyPosture({ slouch: 0, lean: 0, tilt: 0, yaw: 0 });
}

function setWorldRotation(bone, q) {
  const o = node(bone);
  o.quaternion.copy(worldQuat(o.parent).invert().multiply(q));
  o.updateMatrixWorld(true);
}

// Rotations about the avatar's own axes; a positive angle about `left` bends toward `forward`.
const about = (axis, angle) => new THREE.Quaternion().setFromAxisAngle(axis, angle);

// Also used for the stretch demonstrations: headRoll > 0 tips the head toward the screen's right
// (like a mirror), shrug 0..1 lifts both shoulders.
function applyPosture({ slouch, lean, tilt, yaw, headRoll = 0, shrug = 0 }) {
  if (!vrm) return;
  const { left, forward, up, quats } = rest;
  for (const [bone, sign] of [["leftShoulder", 1], ["rightShoulder", -1]]) {
    if (node(bone)) setWorldRotation(bone, about(forward, deg(sign * 22 * shrug)).multiply(quats[bone]));
  }
  // Arms resting down instead of the T-pose.
  setWorldRotation("leftUpperArm", about(forward, deg(-72)).multiply(quats.leftUpperArm));
  setWorldRotation("rightUpperArm", about(forward, deg(72)).multiply(quats.rightUpperArm));
  // Spine rounds forward with slouching and leaning in, and rolls with uneven shoulders.
  const spineTurn = about(left, deg(Math.min(65, 28 * slouch + 15 * lean))).premultiply(about(forward, deg(-tilt)));
  setWorldRotation("spine", spineTurn.clone().multiply(quats.spine));
  // Head drops further and turns toward where the face points.
  const headTurn = about(up, deg(yaw)).multiply(about(forward, deg(-headRoll))).multiply(about(left, deg(Math.min(45, 25 * slouch)))).multiply(spineTurn);
  setWorldRotation("head", headTurn.multiply(quats.head));
}

// --- posture metrics from image landmarks -----------------------------------

const NOSE = 0, LEAR = 7, REAR = 8, LS = 11, RS = 12;
let aspect = 16 / 9; // video width / height, so x and y distances are comparable
let baseline = null;
let metrics = null;

function measure(lm) {
  const pt = (i) => new THREE.Vector2(lm[i].x * aspect, lm[i].y);
  const ls = pt(LS), rs = pt(RS);
  const shoulderMid = ls.clone().add(rs).multiplyScalar(0.5);
  const earMid = pt(LEAR).add(pt(REAR)).multiplyScalar(0.5);
  const shoulderW = ls.distanceTo(rs);
  return {
    shoulderW,
    // Height of the ears above the shoulder line, in shoulder widths: drops when the head juts forward or down.
    neck: (shoulderMid.y - earMid.y) / shoulderW,
    // Shoulder line angle; image y grows downward.
    tilt: THREE.MathUtils.radToDeg(Math.atan2(rs.y - ls.y, Math.abs(rs.x - ls.x))),
    // Ear line angle; positive when the head tips toward the person's right shoulder.
    earTilt: THREE.MathUtils.radToDeg(Math.atan2(pt(REAR).y - pt(LEAR).y, Math.abs(pt(REAR).x - pt(LEAR).x))),
    // Nose offset from the ear midpoint, in shoulder widths: turning the head.
    yaw: (pt(NOSE).x - earMid.x) / shoulderW,
    // Nose depth relative to the ears: clearly negative only when the face points at the camera.
    // Photo test: front views -0.28 to -0.39, side views about 0, a back view +0.33.
    facing: lm[NOSE].z - (lm[LEAR].z + lm[REAR].z) / 2,
  };
}

function smoothMetrics(m) {
  metrics = metrics ? Object.fromEntries(Object.entries(m).map(([k, v]) => [k, metrics[k] + (v - metrics[k]) * 0.3])) : m;
  return metrics;
}

function calibrate() {
  if (!metrics) return;
  baseline = { ...metrics };
  badSeconds = 0;
  $("status").textContent = "已校準。這就是你的「坐正」。";
  say("校準好了");
}

// 0 = like the calibrated posture, 1 = clearly off.
function deviations(m) {
  return {
    // Photo test: one person's slouched shot was 9% lower than her upright one, so 15% counts as fully off.
    slouch: clamp01((1 - m.neck / baseline.neck) / 0.15),
    lean: clamp01((m.shoulderW / baseline.shoulderW - 1) / 0.25),
    tilt: THREE.MathUtils.clamp(m.tilt - baseline.tilt, -15, 15),
    yaw: THREE.MathUtils.clamp((m.yaw - baseline.yaw) * -90, -40, 40),
  };
}

// --- reminders --------------------------------------------------------------

let voiceOn = true;
let zhVoice = null;
const pickVoice = () => {
  const voices = speechSynthesis.getVoices();
  zhVoice = voices.find((v) => v.lang === "zh-TW") ?? voices.find((v) => v.lang.startsWith("zh")) ?? null;
};
speechSynthesis.onvoiceschanged = pickVoice;
pickVoice();
function say(text) {
  if (!voiceOn) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  if (zhVoice) u.voice = zhVoice;
  u.lang = "zh-TW";
  speechSynthesis.speak(u);
}

let badSeconds = 0;       // total time spent off posture
let badStreak = 0;        // current continuous stretch
let reminded = false;
let demoUntil = 0;        // the buddy sits up straight to show you, until this time
let lastTick = performance.now();

function update(lm, now = performance.now()) {
  const dt = Math.min(0.5, (now - lastTick) / 1000);
  lastTick = now;
  const m = smoothMetrics(measure(lm));
  if (!baseline) {
    applyPosture({ slouch: 0, lean: 0, tilt: 0, yaw: 0 });
    return;
  }
  if (m.facing > -0.12) {
    // Side or back views give nonsense ratios, so don't judge them.
    $("score").textContent = "–";
    $("cue").textContent = "請正對鏡頭";
    $("cue").className = "warn";
    return;
  }
  if (stretch) {
    runStretch(m, now, dt);
    return;
  }
  const d = deviations(m);
  const off = Math.max(d.slouch, d.lean, Math.abs(d.tilt) / 12);
  const score = Math.round(100 * (1 - 0.8 * Math.min(1, off)));
  $("score").textContent = score;
  $("neck").textContent = Math.round(d.slouch * 100) + "%";
  $("lean").textContent = Math.round(d.lean * 100) + "%";
  $("tilt").textContent = d.tilt.toFixed(0);

  const cue = now < summaryUntil ? document.createElement("div") : $("cue");
  const grace = Number($("grace").value);
  if (off > 0.5) {
    badSeconds += dt;
    badStreak += dt;
    if (badStreak >= grace && !reminded) {
      reminded = true;
      demoUntil = now + 3000;
      say(d.lean >= d.slouch ? "太靠近螢幕了，往後坐一點" : "坐直一點喔，看我");
    }
    cue.textContent = reminded ? "跟著它坐直" : "有點駝背了…";
    cue.className = "warn";
  } else {
    if (reminded) say("很好，就是這樣");
    badStreak = 0;
    reminded = false;
    cue.textContent = "姿勢很好";
    cue.className = "good";
  }
  $("badtime").textContent = Math.round(badSeconds);
  // Mirror the user, except while demonstrating the upright posture after a reminder.
  applyPosture(now < demoUntil ? { slouch: 0, lean: 0, tilt: 0, yaw: 0 } : exaggerate(d));
}

// The buddy over-acts your posture so a small slump is obvious at a glance.
function exaggerate(d) {
  const g = Number($("gain").value);
  return { slouch: d.slouch * g, lean: d.lean * g, tilt: THREE.MathUtils.clamp(d.tilt * g, -35, 35), yaw: d.yaw };
}

// --- guided neck and shoulder stretch ------------------------------------------

// Each step: what to say, how the buddy demonstrates it, and when the user is doing it.
// Tipping the head also lowers the ear midpoint, so the shrug only counts with the head level.
// "Right" means the person's own right; the buddy faces you, so it tips to the screen's right like a mirror.
const STRETCH = [
  { say: "頭慢慢往右倒，右耳靠近右肩", pose: { headRoll: 30 }, done: (m) => m.earTilt - baseline.earTilt > 12 },
  { say: "換邊，頭往左倒，左耳靠近左肩", pose: { headRoll: -30 }, done: (m) => m.earTilt - baseline.earTilt < -12 },
  { say: "肩膀往上聳，停住", pose: { shrug: 1 }, done: (m) => m.neck / baseline.neck < 0.88 && Math.abs(m.earTilt - baseline.earTilt) < 8 },
  { say: "頭轉向右邊", pose: { yaw: -35 }, done: (m) => (m.yaw - baseline.yaw) < -0.18 },
  { say: "頭轉向左邊", pose: { yaw: 35 }, done: (m) => (m.yaw - baseline.yaw) > 0.18 },
];
// Video test: people in stretch clips held each side about 3 s, sometimes 2.8 s.
const HOLD_SECONDS = 2;
const STEP_TIMEOUT = 15;
let stretch = null; // { step, held, elapsed }
let summaryUntil = 0; // keep the stretch summary on screen until this time

function startStretch() {
  if (!baseline) {
    $("status").textContent = "先坐正校準，再開始伸展。";
    return;
  }
  stretch = { step: 0, held: 0, elapsed: 0 };
  announceStep();
}

function announceStep() {
  const s = STRETCH[stretch.step];
  $("cue").textContent = `伸展 ${stretch.step + 1}/${STRETCH.length}：${s.say}`;
  $("cue").className = "";
  say(s.say);
}

function runStretch(m, now, dt) {
  const s = STRETCH[stretch.step];
  applyPosture({ slouch: 0, lean: 0, tilt: 0, yaw: 0, ...s.pose });
  stretch.elapsed += dt;
  stretch.held = s.done(m) ? stretch.held + dt : 0;
  if (stretch.held > 0) {
    $("cue").textContent = `伸展 ${stretch.step + 1}/${STRETCH.length}：停住 ${Math.max(0, HOLD_SECONDS - stretch.held).toFixed(0)} 秒`;
    $("cue").className = "good";
  }
  const finished = stretch.held >= HOLD_SECONDS;
  if (finished || stretch.elapsed >= STEP_TIMEOUT) {
    stretch.results = [...(stretch.results ?? []), finished];
    stretch.step += 1;
    stretch.held = 0;
    stretch.elapsed = 0;
    if (stretch.step >= STRETCH.length) {
      const doneCount = stretch.results.filter(Boolean).length;
      stretch = null;
      summaryUntil = now + 4000;
      $("cue").textContent = `伸展完成：${doneCount}/${STRETCH.length} 個動作有做到`;
      $("cue").className = "good";
      say("伸展完成，做得很好");
      return;
    }
    if (finished) say("很好");
    setTimeout(announceStep, finished ? 700 : 0);
  }
}

// --- webcam -----------------------------------------------------------------

let landmarker = null;
let drawer = null;
let running = false;

async function start() {
  $("status").textContent = "載入姿勢模型…";
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm");
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  const video = $("cam");
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 } });
  await video.play();
  aspect = video.videoWidth / video.videoHeight;
  const overlay = $("overlay");
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  drawer = new DrawingUtils(overlay.getContext("2d"));
  running = true;
  $("start").disabled = true;
  $("status").textContent = "坐正後按「校準」。3 秒內沒按會自動校準。";
  setTimeout(() => { if (!baseline) calibrate(); }, 3000);
}

function track() {
  const video = $("cam");
  if (!running || video.readyState < 2) return;
  const res = landmarker.detectForVideo(video, performance.now());
  const ctx = $("overlay").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!res.landmarks?.length) return;
  drawer.drawConnectors(res.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: "#7aa2ff", lineWidth: 3 });
  update(res.landmarks[0]);
}

// For testing without a camera: feed image landmarks with an explicit timestamp.
window.__posture = { update, calibrate: () => calibrate(), reset: () => { metrics = null; }, startStretch, state: () => stretch };

$("start").onclick = () => start().catch((err) => ($("status").textContent = "無法開始：" + err.message));
$("calibrate").onclick = calibrate;
$("stretch").onclick = startStretch;
$("gain").oninput = () => { $("gainval").textContent = Number($("gain").value).toFixed(1) + "×"; };
$("voice").onchange = () => { voiceOn = $("voice").checked; };
$("vrmfile").onchange = (e) => {
  const file = e.target.files[0];
  if (file) loadVRM(URL.createObjectURL(file)).catch((err) => ($("status").textContent = "VRM 載入失敗：" + err.message));
};

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  track();
  vrm?.update(clock.getDelta());
  renderer.render(scene, camera);
});

loadVRM("/buddy.vrm").catch((err) => ($("status").textContent = "VRM 載入失敗：" + err.message));
