// Desk posture buddy: webcam upper-body pose + face -> VRM that slumps and looks around with you,
// and speaks up when you slouch, look away or doze off.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { FilesetResolver, PoseLandmarker, FaceLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";

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
  const { clientWidth: w, clientHeight: h } = canvas; // its own box, so it also fits the PiP window
  if (!w || !h) return;
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
  frameCamera();
  applyPosture({ slouch: 0, lean: 0, tilt: 0, yaw: 0 });
}

// Frame the upper body, like a buddy sitting across the desk; closer in on head and shoulders in the small PiP window.
function frameCamera(close = false) {
  if (!rest) return;
  const target = rest.head.clone().setY(rest.head.y - (close ? 0.15 : 0.3));
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(rest.forward, (close ? 1.5 : 3.2) * Math.max(1, 1 / camera.aspect));
  controls.update();
}

function setWorldRotation(bone, q) {
  const o = node(bone);
  o.quaternion.copy(worldQuat(o.parent).invert().multiply(q));
  o.updateMatrixWorld(true);
}

// Rotations about the avatar's own axes; a positive angle about `left` bends toward `forward`.
const about = (axis, angle) => new THREE.Quaternion().setFromAxisAngle(axis, angle);

// Also used for the stretch demonstrations: headRoll > 0 tips the head toward the screen's right
// (like a mirror), shrug 0..1 lifts both shoulders. pitch > 0 nods the head down, on top of the slouch.
function applyPosture({ slouch, lean, tilt, yaw, pitch = 0, headRoll = 0, shrug = 0 }) {
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
  const headTurn = about(up, deg(yaw)).multiply(about(forward, deg(-headRoll))).multiply(about(left, deg(Math.min(45, 25 * slouch) + pitch))).multiply(spineTurn);
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
  faceBaseline = faceMetrics && { ...faceMetrics };
  badSeconds = 0;
  awaySeconds = 0;
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
let speaking = false;      // drives the mouth while a line is spoken
let lastLine = "";
let bubbleUntil = 0;
function say(text) {
  lastLine = text;
  $("bubble").textContent = text;
  $("bubble").hidden = false;
  bubbleUntil = performance.now() + 4000;
  if (!voiceOn) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  if (zhVoice) u.voice = zhVoice;
  u.lang = "zh-TW";
  u.onstart = () => { speaking = true; };
  u.onend = u.onerror = () => { speaking = false; };
  speechSynthesis.speak(u);
}

// What the buddy says, by problem. n = how many times today, s = seconds this stretch has lasted,
// repeat = still the same stretch after an earlier reminder.
const LINES = {
  slouch: ({ n, s, repeat }) => repeat ? [`已經駝背 ${s} 秒了，肩膀往後打開`, `還是彎著喔，${s} 秒了，看我坐直`]
    : n >= 3 ? [`今天第 ${n} 次駝背了，要不要一起做個肩頸伸展？`, `第 ${n} 次囉，是不是椅子太低了？`]
    : ["坐直一點喔，看我", "背有點彎了，跟我一起坐正"],
  lean: ({ n, s, repeat }) => repeat ? [`離螢幕太近 ${s} 秒了，眼睛會累喔`, `還是太近了，往後靠到椅背`]
    : n >= 3 ? [`今天第 ${n} 次貼近螢幕了，字太小的話可以放大`, `又靠過來了，這是今天第 ${n} 次`]
    : ["太靠近螢幕了，往後坐一點", "臉快貼到螢幕了，往後一點"],
  away: ({ n, s, repeat }) => repeat ? [`已經 ${s} 秒沒看螢幕了，要不要先把手機放下？`, `還在看旁邊喔，${s} 秒了`]
    : n >= 3 ? [`今天第 ${n} 次分心了，是不是累了？休息五分鐘也可以`, `第 ${n} 次跑掉囉，我們把這段做完再休息`]
    : ["回來喔，你分心了", "嘿，看這邊，螢幕在這裡"],
  noface: ({ n, s, repeat }) => repeat ? [`${s} 秒看不到你的臉了，在低頭滑手機嗎？`]
    : n >= 3 ? [`今天第 ${n} 次低頭不見了，手機先蓋起來吧`]
    : ["我看不到你的臉，在低頭滑手機嗎？", "臉轉回來一下好嗎"],
  drowsy: ({ n, s, repeat, night }) => night ? ["很晚了，眼睛都閉起來了，早點睡吧"]
    : repeat ? [`眼睛閉了 ${s} 秒了，醒醒！`, "還在打瞌睡喔，站起來動一動"]
    : n >= 3 ? [`今天第 ${n} 次想睡了，建議站起來走一走`]
    : ["眼睛快閉起來了，起來動一動吧", "想睡了嗎？去喝口水"],
  back: ({ s }) => s > 30 ? ["回來了，辛苦了，繼續加油", "好，我們繼續"] : ["很好，就是這樣", "歡迎回來", "對，這樣就好"],
};

const today = new Date().toLocaleDateString("sv"); // YYYY-MM-DD in local time
let counts = {};
try { counts = JSON.parse(localStorage.getItem("bodybuddy.counts." + today)) ?? {}; } catch {}
const totalCount = () => Object.values(counts).reduce((a, b) => a + b, 0);

function compose(kind, seconds, repeat) {
  if (!repeat && kind !== "back") {
    counts[kind] = (counts[kind] ?? 0) + 1;
    try { localStorage.setItem("bodybuddy.counts." + today, JSON.stringify(counts)); } catch {}
  }
  const hour = new Date().getHours();
  const pool = LINES[kind]({ n: counts[kind] ?? 0, s: Math.round(seconds), repeat, night: hour >= 23 || hour < 5 });
  const fresh = pool.filter((l) => l !== lastLine);
  return (fresh.length ? fresh : pool)[Math.floor(Math.random() * (fresh.length || pool.length))];
}

// Posture and gaze reminders share the voice, so one waits while the other was just spoken.
let lastRemindAt = -Infinity;
function remind(kind, seconds, repeat, now) {
  if (now - lastRemindAt < 5000) return false;
  lastRemindAt = now;
  say(compose(kind, seconds, repeat));
  $("nudges").textContent = totalCount();
  return true;
}

let badSeconds = 0;       // total time spent off posture
let badStreak = 0;        // current continuous stretch
let reminded = false;
let postureNext = 0;      // when the next reminder may come during the same stretch
let demoUntil = 0;        // the buddy sits up straight to show you, until this time
let lastTick = performance.now();

function update(lm, now = performance.now()) {
  const dt = Math.min(0.5, (now - lastTick) / 1000);
  lastTick = now;
  lastPoseAt = now;
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
    if (badStreak >= grace && now >= postureNext && remind(d.lean >= d.slouch ? "lean" : "slouch", badStreak, reminded, now)) {
      reminded = true;
      postureNext = now + Math.max(20, 2 * grace) * 1000;
      demoUntil = now + 3000;
    }
    cue.textContent = reminded ? "跟著它坐直" : "有點駝背了…";
    cue.className = "warn";
  } else {
    if (reminded) say(compose("back", badStreak));
    badStreak = 0;
    reminded = false;
    postureNext = 0;
    cue.textContent = "姿勢很好";
    cue.className = "good";
  }
  $("badtime").textContent = Math.round(badSeconds);
  // Mirror the user, except while demonstrating the upright posture after a reminder.
  applyPosture(now < demoUntil ? { slouch: 0, lean: 0, tilt: 0, yaw: 0 } : { ...exaggerate(d), ...headGaze(now) });
}

// The buddy over-acts your posture so a small slump is obvious at a glance.
function exaggerate(d) {
  const g = Number($("gain").value);
  return { slouch: d.slouch * g, lean: d.lean * g, tilt: THREE.MathUtils.clamp(d.tilt * g, -35, 35), yaw: d.yaw };
}

// --- gaze: where the face points, eye direction, closed eyes ------------------

let faceBaseline = null;
let faceMetrics = null;
let lastFaceAt = -Infinity;
let lastPoseAt = -Infinity;
let lastFaceTick = performance.now();
let gazeState = "–";
let awaySeconds = 0;      // total time distracted
let awayStreak = 0;
let awayReminded = false;
let awayNext = 0;
let attendUntil = 0;      // the buddy turns to look at you after a gaze reminder, until this time

const GAZE_LABEL = { focus: "專注", away: "看旁邊", noface: "看不到臉", drowsy: "想睡", absent: "不在位子" };

function measureFace(res) {
  const shapes = res?.faceBlendshapes?.[0]?.categories;
  const mat = res?.facialTransformationMatrixes?.[0]?.data;
  if (!shapes || !mat) return null;
  const b = Object.fromEntries(shapes.map((c) => [c.categoryName, c.score]));
  // Column-major 4x4; the third column is where the face's forward axis points in camera space
  // (x toward image right, y up, z toward the camera). Turning to your own right gives yaw < 0.
  const f = new THREE.Vector3(mat[8], mat[9], mat[10]).normalize();
  return {
    yaw: THREE.MathUtils.radToDeg(Math.atan2(f.x, f.z)),
    pitch: THREE.MathUtils.radToDeg(Math.atan2(f.y, Math.hypot(f.x, f.z))),
    // Eyes turned sideways (either way) and up minus down, 0..1 blendshape scores.
    lookSide: (b.eyeLookOutLeft + b.eyeLookInLeft + b.eyeLookInRight + b.eyeLookOutRight) / 2,
    lookUp: (b.eyeLookUpLeft + b.eyeLookUpRight - b.eyeLookDownLeft - b.eyeLookDownRight) / 2,
    blink: (b.eyeBlinkLeft + b.eyeBlinkRight) / 2,
  };
}

function classifyGaze(f, now) {
  if (!f) return now - lastPoseAt < 500 ? "noface" : "absent";
  const yaw = f.yaw - faceBaseline.yaw;
  const pitch = f.pitch - faceBaseline.pitch;
  if (f.blink > Math.max(0.5, faceBaseline.blink + 0.35)) return "drowsy";
  // The eyes add to where the head points: a 20 degree head turn plus eyes to the corner is away.
  const side = Math.max(0, f.lookSide - faceBaseline.lookSide);
  const up = f.lookUp - faceBaseline.lookUp;
  if (Math.abs(yaw) + 30 * side > 28 || pitch + 25 * up < -25 || pitch + 25 * up > 22) return "away";
  return "focus";
}

function updateFace(res, now = performance.now()) {
  const dt = Math.min(0.5, (now - lastFaceTick) / 1000);
  lastFaceTick = now;
  const raw = measureFace(res);
  if (raw) {
    faceMetrics = faceMetrics && now - lastFaceAt < 1000
      ? Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, faceMetrics[k] + (v - faceMetrics[k]) * 0.3]))
      : raw;
    lastFaceAt = now;
  }
  if (!baseline) return;
  if (!faceBaseline && raw) faceBaseline = { ...faceMetrics }; // calibrated before the face model was ready
  if (!faceBaseline) return;
  gazeState = classifyGaze(raw && faceMetrics, now);
  $("gaze").textContent = GAZE_LABEL[gazeState];
  $("gaze").className = "pill " + (gazeState === "focus" ? "good" : gazeState === "absent" ? "" : "warn");
  const distracted = gazeState === "away" || gazeState === "drowsy" || gazeState === "noface";
  if (distracted && !stretch) {
    awaySeconds += dt;
    awayStreak += dt;
    if (awayStreak >= Number($("gazegrace").value) && now >= awayNext && remind(gazeState, awayStreak, awayReminded, now)) {
      awayReminded = true;
      awayNext = now + 20000;
      attendUntil = now + 3000;
    }
  } else if (gazeState !== "absent") {
    if (awayReminded) say(compose("back", awayStreak));
    awayStreak = 0;
    awayReminded = false;
    awayNext = 0;
  }
  $("awaytime").textContent = Math.round(awaySeconds);
}

// Head angles for the avatar from the face, mirrored like the pose yaw; nothing if the face is stale.
function headGaze(now) {
  if (now < attendUntil) return { yaw: 0, pitch: 0 };
  if (!faceBaseline || now - lastFaceAt > 300) return {};
  return {
    yaw: THREE.MathUtils.clamp(-(faceMetrics.yaw - faceBaseline.yaw), -40, 40),
    pitch: THREE.MathUtils.clamp(-(faceMetrics.pitch - faceBaseline.pitch) - 15 * (faceMetrics.lookUp - faceBaseline.lookUp), -25, 30),
  };
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
let faceLandmarker = null;
let drawer = null;
let running = false;

async function start() {
  $("status").textContent = "載入姿勢與臉部模型…";
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm");
  const face = FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  }).catch((err) => {
    console.warn("face model failed, gaze is off:", err);
    return null;
  });
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  faceLandmarker = await face;
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
  $("status").textContent = (faceLandmarker ? "" : "臉部模型載入失敗，只看姿勢。") + "坐正、看著螢幕後按「校準」。3 秒內沒按會自動校準。";
  setTimeout(() => { if (!baseline) calibrate(); }, 3000);
}

function track() {
  const video = $("cam");
  if (!running || video.readyState < 2) return;
  const now = performance.now();
  const res = landmarker.detectForVideo(video, now);
  const ctx = $("overlay").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (res.landmarks?.length) {
    drawer.drawConnectors(res.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: "#7aa2ff", lineWidth: 3 });
    update(res.landmarks[0], now);
  }
  // After the pose, so a missing face can be told apart from an empty chair.
  if (faceLandmarker) updateFace(faceLandmarker.detectForVideo(video, now), now);
}

// For testing without a camera: feed image landmarks with an explicit timestamp.
window.__posture = { update, calibrate: () => calibrate(), reset: () => { metrics = null; }, startStretch, state: () => stretch };
// Same for the face: pass a FaceLandmarker-shaped result ({ faceBlendshapes, facialTransformationMatrixes }) or null.
window.__face = { update: updateFace, state: () => ({ gaze: gazeState, awayStreak, awaySeconds, counts, lastLine, head: vrm && node("head").quaternion.toArray() }) };

$("start").onclick = () => start().catch((err) => ($("status").textContent = "無法開始：" + err.message));
$("calibrate").onclick = calibrate;
$("stretch").onclick = startStretch;
$("pip").onclick = () => togglePip().catch((err) => ($("status").textContent = "無法開浮動視窗：" + err.message));
$("gain").oninput = () => { $("gainval").textContent = Number($("gain").value).toFixed(1) + "×"; };
$("voice").onchange = () => { voiceOn = $("voice").checked; };
$("vrmfile").onchange = (e) => {
  const file = e.target.files[0];
  if (file) loadVRM(URL.createObjectURL(file)).catch((err) => ($("status").textContent = "VRM 載入失敗：" + err.message));
};

// --- floating desktop window (Document Picture-in-Picture) ---------------------

// The avatar canvas moves into an always-on-top window; the camera and tracking stay in this tab.
let pip = null;

async function togglePip() {
  if (pip) {
    pip.close();
    return;
  }
  if (!("documentPictureInPicture" in window)) {
    $("status").textContent = "這個瀏覽器不支援浮動視窗，請用桌面版 Chrome（116 以上）。";
    return;
  }
  pip = await documentPictureInPicture.requestWindow({ width: 300, height: 380 });
  for (const sheet of document.styleSheets) {
    try {
      const style = pip.document.createElement("style");
      style.textContent = [...sheet.cssRules].map((r) => r.cssText).join("\n");
      pip.document.head.append(style);
    } catch {
      const link = pip.document.createElement("link");
      link.rel = "stylesheet";
      link.href = sheet.href;
      pip.document.head.append(link);
    }
  }
  pip.document.title = "Desk Buddy";
  pip.document.body.className = "pip";
  const wrap = pip.document.createElement("div");
  wrap.id = "pipstage";
  const bar = pip.document.createElement("div");
  bar.id = "pipbar";
  wrap.append(canvas, bar);
  pip.document.body.append(wrap);
  pip.addEventListener("pagehide", () => {
    $("stage").prepend(canvas);
    pip = null;
    $("pip").textContent = "浮到桌面";
    resize();
    frameCamera();
    schedule();
  });
  $("pip").textContent = "收回分頁";
  resize();
  frameCamera(true);
  schedule(); // keep animating from the PiP window, which stays visible while this tab is in the background
}

function updatePipBar() {
  const bar = pip?.document.getElementById("pipbar");
  if (!bar) return;
  const text = `${$("score").textContent} 分｜${$("gaze").textContent}｜${performance.now() < bubbleUntil ? lastLine : $("cue").textContent}`;
  if (bar.textContent !== text) bar.textContent = text;
  bar.className = $("cue").className;
}

// --- loop ---------------------------------------------------------------------

const clock = new THREE.Clock();
let loopId = 0;
function schedule() {
  const id = ++loopId;
  (pip ?? window).requestAnimationFrame(() => { if (id === loopId) frame(); });
}
function frame() {
  track();
  const now = performance.now();
  if (now >= bubbleUntil) $("bubble").hidden = true;
  const em = vrm?.expressionManager;
  if (em) {
    em.setValue("aa", speaking ? 0.25 + 0.25 * Math.sin(now / 60) : 0);
    em.setValue("surprised", now < attendUntil ? 0.5 : 0);
  }
  vrm?.update(clock.getDelta());
  const size = renderer.getSize(new THREE.Vector2());
  if (size.x !== canvas.clientWidth || size.y !== canvas.clientHeight) {
    // Also catches the PiP window being opened or resized.
    resize();
    if (pip) frameCamera(true);
  }
  renderer.render(scene, camera);
  updatePipBar();
  schedule();
}
schedule();

loadVRM("/template.vrm").catch((err) => ($("status").textContent = "VRM 載入失敗：" + err.message));
