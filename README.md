# Body Buddy

A desk buddy: your webcam watches how you sit and where you look. A VRM avatar slumps and looks around along with you (and over-acts it), speaks up when you slouch, look away or doze off, and leads a neck and shoulder stretch. It can float on top of your other apps like a desktop pet.
Made for the NTU「虛擬人與遙現」2026 midterm.

Live: https://bodybuddy.cechung.com (use Chrome; allow the camera)

1. Press 開始, sit straight looking at the screen, press 校準.
2. The VRM mirrors your slouch, leaning in and shoulder tilt, exaggerated 1–3×, and reminds you by voice after a grace period, then sits up straight to show you.
3. Gaze: its head follows where you look (mirrored). If you look away, close your eyes or your face disappears while you are still in the chair for longer than 「分心多久才提醒」 (5/15/30 s), it turns to you and talks (「回來喔，你分心了」). The HUD pill next to the score shows 專注 / 看旁邊 / 想睡 / 看不到臉 / 不在位子.
4. 浮到桌面: moves the avatar and a one-line status into an always-on-top window ([Document Picture-in-Picture](https://developer.chrome.com/docs/web-platform/document-picture-in-picture), desktop Chrome 116+). The camera keeps running in the original tab, which can go to the background; close the small window or press 收回分頁 to bring it back.
5. 一起做肩頸伸展: five steps (tilt right, tilt left, shrug, turn right, turn left), demonstrated by the VRM and checked by the camera.

What it says depends on the problem (slouching, too close, looking away, no face, drowsy), how long it has lasted (it repeats with the duration every 20 s or so), and how many times it happened today (kept in this browser's localStorage); for example 「已經 25 秒沒看螢幕了，要不要先把手機放下？」 or 「今天第 3 次駝背了，要不要一起做個肩頸伸展？」. The lines come from a small local rule-based generator in `src/posture.js`, so no API key or network call is involved.

Load your own `.vrm` (for example one made from a photo with [Single-Photo-VRM](https://github.com/dAAAb/Single-Photo-VRM)) from the side panel. Everything runs in the browser; no video leaves your computer.

## How it works

- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (33 keypoints, in-browser WASM/GPU) → [three-vrm](https://github.com/pixiv/three-vrm).
- It uses five 2D points (nose, ears, shoulders) relative to your own calibration: ear height above the shoulder line, shoulder width (leaning in), shoulder line angle, and nose-vs-ears depth to refuse side or back views.
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) on the same frames gives the head direction (facial transformation matrix) and eye blendshapes (`eyeLookIn/Out/Up/Down`, `eyeBlink`). Relative to calibration, more than about 28° sideways (head plus eyes) or 25° down or 22° up counts as looking away; eyes mostly closed counts as drowsy.
- For testing without a camera, `window.__posture.update(landmarks, t)` and `window.__face.update(faceResult, t)` take simulated results.

## Run locally

```bash
npm install
npm run dev        # http://127.0.0.1:5191
npm run deploy     # build + wrangler deploy
npm run smoke      # checks the live site
```

## Evidence

[`tests/FINDINGS.md`](tests/FINDINGS.md) records what photos, the MultiPosture dataset and short videos showed, and which bugs they caught. The squat sections come from an earlier squat page that has since been dropped. Reproduce with Python 3.12:

```bash
cd tests && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
./fetch_data.sh                                   # model + Pexels media + MultiPosture (not redistributed here)
.venv/bin/python run.py photos                    # front-camera posture on photos
.venv/bin/python run2.py side side                # head-neck angle, side photos
.venv/bin/python run2.py squat squat              # knee angle, front vs side
.venv/bin/python run_video.py stretch videos/8343371*.mp4 videos/6967424*.mp4
.venv/bin/python run_video.py squat videos/6454275*.mp4 videos/4267356*.mp4
cd multiposture && ../.venv/bin/python eval.py
```

## Credits

- `public/buddy.vrm` (default): VRoid Project's sample avatar 桜田 史利矢 (Sakurada Fumiriya), released with every VRoid Hub permission set to allow (modification, redistribution, commercial use, no credit required; the companion samples are CC0), from the [VRoid Studio CC0 models](https://opengameart.org/content/vroid-studio-cc0-models) collection. Hair and eyebrows recoloured black by `avatar/recolor.py`.
- `public/template.vrm`: neutral body built by Single-Photo-VRM's `template/build.sh` from [NAVER Anny](https://github.com/naver/anny) (Apache-2.0 code; MPFB2 / MakeHuman CC0 assets).
- Test data, downloaded by `tests/fetch_data.sh`: photos and videos from [Pexels](https://www.pexels.com/license/); [MultiPosture](https://zenodo.org/records/14230872) (CC BY 4.0).
