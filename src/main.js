// Webcam body pose (MediaPipe) -> your VRM, beside a coach VRM that demonstrates the squat.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { FilesetResolver, PoseLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";

const $ = (id) => document.getElementById(id);

// --- scene ------------------------------------------------------------------

const canvas = $("three");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 50);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.9, 0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666677, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(1, 3, 2);
scene.add(sun);
scene.add(new THREE.GridHelper(6, 12, 0x888888, 0xcccccc));

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// --- avatars and their rest poses -------------------------------------------

const loader = new GLTFLoader();
loader.register((p) => new VRMLoaderPlugin(p));

const LIMBS = [
  ["leftUpperArm", "leftLowerArm"], ["leftLowerArm", "leftHand"],
  ["rightUpperArm", "rightLowerArm"], ["rightLowerArm", "rightHand"],
  ["leftUpperLeg", "leftLowerLeg"], ["leftLowerLeg", "leftFoot"],
  ["rightUpperLeg", "rightLowerLeg"], ["rightLowerLeg", "rightFoot"],
];

const worldPos = (o) => o.getWorldPosition(new THREE.Vector3());
const worldQuat = (o) => o.getWorldQuaternion(new THREE.Quaternion());

// An avatar is { vrm, rest, mirror }; `you` follows the webcam, `coach` demonstrates.
const you = { vrm: null, rest: null, mirror: true };
const coach = { vrm: null, rest: null, mirror: false };
let vrmUrl = "/template.vrm";

const node = (av, name) => av.vrm.humanoid.getNormalizedBoneNode(name);

async function loadAvatar(av, url) {
  const gltf = await loader.loadAsync(url);
  if (av.vrm) {
    scene.remove(av.vrm.scene);
    VRMUtils.deepDispose(av.vrm.scene);
  }
  av.vrm = gltf.userData.vrm;
  VRMUtils.rotateVRM0(av.vrm);
  scene.add(av.vrm.scene);
  captureRest(av);
}

async function loadBoth(url) {
  vrmUrl = url;
  await Promise.all([loadAvatar(you, url), loadAvatar(coach, url)]);
  styleCoach();
  placeCoach();
  setView(currentView);
}

// The coach is a translucent blue figure, so it reads as a guide when overlaid on you.
function styleCoach() {
  const ghost = new THREE.MeshStandardMaterial({ color: 0x5b6cff, transparent: true, opacity: 0.45, depthWrite: false });
  coach.vrm.scene.traverse((o) => {
    if (o.isMesh) {
      o.material = ghost;
      o.renderOrder = 1;
    }
  });
}

function captureRest(av) {
  av.vrm.humanoid.resetNormalizedPose();
  av.vrm.scene.updateMatrixWorld(true);
  // Avatar axes in world space: its left, up, and forward (toward the viewer).
  const left = worldPos(node(av, "leftUpperLeg")).sub(worldPos(node(av, "rightUpperLeg"))).setY(0).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const forward = new THREE.Vector3().crossVectors(left, up).normalize();
  const bones = {};
  for (const [bone, child] of LIMBS) {
    bones[bone] = {
      quat: worldQuat(node(av, bone)),
      dir: worldPos(node(av, child)).sub(worldPos(node(av, bone))).normalize(),
    };
  }
  for (const bone of ["hips", "spine"]) bones[bone] = { quat: worldQuat(node(av, bone)) };
  const footY = Math.min(worldPos(node(av, "leftFoot")).y, worldPos(node(av, "rightFoot")).y);
  av.rest = {
    left, up, forward, bones, footY,
    hipsPos: node(av, "hips").position.clone(),
    basis: new THREE.Matrix4().makeBasis(left, up, forward),
  };
}

// --- pose landmarks -> bone rotations ---------------------------------------

// MediaPipe pose indices; "L"/"R" are the person's own left and right.
const P = { LS: 11, RS: 12, LE: 13, RE: 14, LW: 15, RW: 16, LH: 23, RH: 24, LK: 25, RK: 26, LA: 27, RA: 28 };

// World landmarks: meters, origin between the hips, x toward the person's left, y down, z away from camera.
function toAvatar(av, lm, i) {
  const p = lm[i];
  const { left, up, forward } = av.rest;
  return left.clone().multiplyScalar(av.mirror ? -p.x : p.x).addScaledVector(up, -p.y).addScaledVector(forward, -p.z);
}

function setWorldRotation(av, bone, q) {
  const o = node(av, bone);
  o.quaternion.copy(worldQuat(o.parent).invert().multiply(q));
  o.updateMatrixWorld(true);
}

function applyPose(av, lm) {
  const { rest } = av;
  const pt = (i) => toAvatar(av, lm, i);
  // Landmark index that drives the avatar's own left or right side.
  const L = (a, b) => (av.mirror ? b : a);
  const R = (a, b) => (av.mirror ? a : b);

  // Torso: hips from the hip line, spine from the shoulder line, both with the hip->shoulder axis as up.
  const hipL = pt(L(P.LH, P.RH)), hipR = pt(R(P.LH, P.RH));
  const shL = pt(L(P.LS, P.RS)), shR = pt(R(P.LS, P.RS));
  const torsoUp = shL.clone().add(shR).multiplyScalar(0.5).sub(hipL.clone().add(hipR).multiplyScalar(0.5));
  const basisFrom = (x, upHint) => {
    x.normalize();
    const y = upHint.clone().addScaledVector(x, -upHint.dot(x)).normalize();
    const z = new THREE.Vector3().crossVectors(x, y);
    const m = new THREE.Matrix4().makeBasis(x, y, z).multiply(rest.basis.clone().transpose());
    return new THREE.Quaternion().setFromRotationMatrix(m);
  };
  setWorldRotation(av, "hips", basisFrom(hipL.clone().sub(hipR), torsoUp).multiply(rest.bones.hips.quat));
  setWorldRotation(av, "spine", basisFrom(shL.clone().sub(shR), torsoUp).multiply(rest.bones.spine.quat));

  // Limbs: rotate each bone so its rest direction points at the next joint.
  const segments = {
    leftUpperArm: [L(P.LS, P.RS), L(P.LE, P.RE)], leftLowerArm: [L(P.LE, P.RE), L(P.LW, P.RW)],
    rightUpperArm: [R(P.LS, P.RS), R(P.LE, P.RE)], rightLowerArm: [R(P.LE, P.RE), R(P.LW, P.RW)],
    leftUpperLeg: [L(P.LH, P.RH), L(P.LK, P.RK)], leftLowerLeg: [L(P.LK, P.RK), L(P.LA, P.RA)],
    rightUpperLeg: [R(P.LH, P.RH), R(P.LK, P.RK)], rightLowerLeg: [R(P.LK, P.RK), R(P.LA, P.RA)],
  };
  for (const [bone] of LIMBS) {
    const [a, b] = segments[bone];
    if ((lm[a].visibility ?? 1) < 0.3 || (lm[b].visibility ?? 1) < 0.3) continue;
    const target = pt(b).sub(pt(a)).normalize();
    const turn = new THREE.Quaternion().setFromUnitVectors(rest.bones[bone].dir, target);
    setWorldRotation(av, bone, turn.multiply(rest.bones[bone].quat));
  }

  // Keep the lower foot on the floor, so bending the knees lowers the hips instead of lifting the feet.
  const hips = node(av, "hips");
  hips.position.copy(rest.hipsPos);
  av.vrm.scene.updateMatrixWorld(true);
  const footY = Math.min(worldPos(node(av, "leftFoot")).y, worldPos(node(av, "rightFoot")).y);
  hips.position.y += rest.footY - footY;
}

// --- coach demonstration ----------------------------------------------------

// Key poses in MediaPipe world coordinates; the squat reaches a knee angle of about 80°.
const pose = (pts) => {
  const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));
  for (const [i, [x, y, z]] of Object.entries(pts)) lm[i] = { x, y, z, visibility: 1 };
  return lm;
};
const STAND = pose({
  11: [0.18, -0.5, 0], 12: [-0.18, -0.5, 0], 13: [0.2, -0.25, 0], 14: [-0.2, -0.25, 0], 15: [0.2, 0, 0], 16: [-0.2, 0, 0],
  23: [0.1, 0, 0], 24: [-0.1, 0, 0], 25: [0.1, 0.45, 0], 26: [-0.1, 0.45, 0], 27: [0.1, 0.9, 0], 28: [-0.1, 0.9, 0],
});
const SQUAT = pose({
  11: [0.18, -0.4, -0.25], 12: [-0.18, -0.4, -0.25], 13: [0.18, -0.4, -0.5], 14: [-0.18, -0.4, -0.5], 15: [0.18, -0.4, -0.75], 16: [-0.18, -0.4, -0.75],
  23: [0.1, 0, 0], 24: [-0.1, 0, 0], 25: [0.15, 0.05, -0.42], 26: [-0.15, 0.05, -0.42], 27: [0.15, 0.45, -0.3], 28: [-0.15, 0.45, -0.3],
});
const blend = (a, b, t) => a.map((p, i) => ({ x: p.x + (b[i].x - p.x) * t, y: p.y + (b[i].y - p.y) * t, z: p.z + (b[i].z - p.z) * t, visibility: 1 }));

const REP_SECONDS = 4; // 2 s down, 2 s up
let coachOn = true;
let coachLayout = "side"; // "side": next to you, "overlay": on top of you

function placeCoach() {
  if (!coach.vrm) return;
  coach.vrm.scene.visible = coachOn;
  const offset = coachLayout === "side" ? you.rest.left.clone().multiplyScalar(-0.9) : new THREE.Vector3();
  coach.vrm.scene.position.copy(offset);
  you.vrm.scene.position.copy(coachLayout === "side" ? offset.clone().negate() : new THREE.Vector3());
}

function animateCoach(time) {
  if (!coach.vrm || !coachOn) return;
  const phase = (time / REP_SECONDS) % 1;
  const t = (1 - Math.cos(phase * 2 * Math.PI)) / 2; // 0 standing -> 1 bottom -> 0
  applyPose(coach, blend(STAND, SQUAT, t));
}

// --- squat metrics and spoken cues ------------------------------------------

function kneeAngle(lm, hip, knee, ankle) {
  const a = new THREE.Vector3(lm[hip].x - lm[knee].x, lm[hip].y - lm[knee].y, lm[hip].z - lm[knee].z);
  const b = new THREE.Vector3(lm[ankle].x - lm[knee].x, lm[ankle].y - lm[knee].y, lm[ankle].z - lm[knee].z);
  return THREE.MathUtils.radToDeg(a.angleTo(b));
}

const ZH_COUNT = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
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
  u.rate = 1.1;
  speechSynthesis.speak(u);
}

const DEEP_ENOUGH = 95;
let reps = 0;
let down = false;
let deepest = 180;
let praised = false;

function updateSquat(lm) {
  const angle = (kneeAngle(lm, P.LH, P.LK, P.LA) + kneeAngle(lm, P.RH, P.RK, P.RA)) / 2;
  $("knee").textContent = angle.toFixed(0);
  const cue = $("cue");
  if (!down && angle < 120) {
    down = true;
    deepest = angle;
    praised = false;
  }
  if (!down) return;
  deepest = Math.min(deepest, angle);
  // Video test: front-view clips read only 148-155° when standing, so 160° never counted a rep.
  if (angle > 140) {
    down = false;
    reps += 1;
    $("reps").textContent = reps;
    const deep = deepest <= DEEP_ENOUGH;
    cue.textContent = deep ? `第 ${reps} 下：夠深（${deepest.toFixed(0)}°）` : `第 ${reps} 下：再蹲低一點（${deepest.toFixed(0)}°）`;
    cue.className = deep ? "good" : "warn";
    say(deep ? (ZH_COUNT[reps] ?? String(reps)) : `${ZH_COUNT[reps] ?? reps}，下一下再低一點`);
  } else if (deepest > DEEP_ENOUGH) {
    cue.textContent = "再低一點…";
    cue.className = "warn";
  } else {
    cue.textContent = "很好，站起來";
    cue.className = "good";
    if (!praised) {
      praised = true;
      say("很好");
    }
  }
}

// --- webcam loop ------------------------------------------------------------

let landmarker = null;
let drawer = null;
let running = false;
let smoothed = null;

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
  const overlay = $("overlay");
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  drawer = new DrawingUtils(overlay.getContext("2d"));
  running = true;
  $("start").disabled = true;
  $("status").textContent = "追蹤中。站遠一點讓膝蓋和腳踝入鏡。";
  say("開始，跟著藍色教練蹲");
}

function track() {
  const video = $("cam");
  if (!running || video.readyState < 2) return;
  const res = landmarker.detectForVideo(video, performance.now());
  const ctx = $("overlay").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!res.worldLandmarks?.length) return;
  drawer.drawConnectors(res.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: "#7aa2ff", lineWidth: 3 });
  feed(res.worldLandmarks[0]);
}

function feed(lm) {
  if (!you.rest) return;
  smoothed = smoothed
    ? lm.map((p, i) => ({ ...p, x: smoothed[i].x + (p.x - smoothed[i].x) * 0.5, y: smoothed[i].y + (p.y - smoothed[i].y) * 0.5, z: smoothed[i].z + (p.z - smoothed[i].z) * 0.5 }))
    : lm.map((p) => ({ ...p }));
  applyPose(you, smoothed);
  updateSquat(smoothed);
}
// For testing without a camera.
window.__feed = (lm) => { smoothed = null; feed(lm); };
window.__poses = { STAND, SQUAT };
window.__avatars = { you, coach };
window.__animateCoach = animateCoach;

// --- camera views -----------------------------------------------------------

let currentView = "front";
function setView(view) {
  currentView = view;
  if (!you.rest) return;
  const target = new THREE.Vector3(0, 0.85, 0);
  // Two figures side by side need a wider shot, more so in a narrow window.
  const dist = (coachOn && coachLayout === "side" ? 5.2 : 3.4) * Math.max(1, 1.2 / camera.aspect);
  const offset = {
    front: you.rest.forward.clone().multiplyScalar(dist),
    side: you.rest.left.clone().multiplyScalar(dist),
    top: you.rest.forward.clone().multiplyScalar(1.4).setY(3.4),
  }[view];
  camera.position.copy(target).add(offset).setY(target.y + offset.y + (view === "top" ? 0 : 0.15));
  controls.target.copy(target);
  controls.update();
  document.querySelectorAll("#views button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
}
document.querySelectorAll("#views button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));

// --- controls ---------------------------------------------------------------

$("start").onclick = () => start().catch((err) => ($("status").textContent = "無法開始：" + err.message));
$("mirror").onchange = () => { you.mirror = $("mirror").checked; smoothed = null; };
$("voice").onchange = () => { voiceOn = $("voice").checked; };
$("coach").onchange = () => { coachOn = $("coach").checked; placeCoach(); setView(currentView); };
$("layout").onchange = () => { coachLayout = $("layout").value; placeCoach(); setView(currentView); };
$("reset").onclick = () => { reps = 0; $("reps").textContent = 0; $("cue").textContent = ""; };
$("vrmfile").onchange = (e) => {
  const file = e.target.files[0];
  if (file) loadBoth(URL.createObjectURL(file)).catch((err) => ($("status").textContent = "VRM 載入失敗：" + err.message));
};

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  track();
  animateCoach(clock.elapsedTime);
  you.vrm?.update(dt);
  coach.vrm?.update(dt);
  renderer.render(scene, camera);
});

loadBoth(vrmUrl).catch((err) => ($("status").textContent = "VRM 載入失敗：" + err.message));
