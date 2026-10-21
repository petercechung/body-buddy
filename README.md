# Body Buddy

A desk buddy: your webcam watches how you sit, a VRM avatar slumps along with you (and over-acts it), reminds you to sit up, and leads a neck and shoulder stretch.
Made for the NTU「虛擬人與遙現」2026 midterm.

Live: https://bodybuddy.cechung.com (use Chrome; allow the camera)

1. Press 開始, sit straight, press 校準.
2. The VRM mirrors your slouch, leaning in and shoulder tilt, exaggerated 1–3×, and reminds you by voice after a grace period, then sits up straight to show you.
3. 一起做肩頸伸展: five steps (tilt right, tilt left, shrug, turn right, turn left), demonstrated by the VRM and checked by the camera.

Load your own `.vrm` (for example one made from a photo with [Single-Photo-VRM](https://github.com/dAAAb/Single-Photo-VRM)) from the side panel. Everything runs in the browser; no video leaves your computer.

## How it works

- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (33 keypoints, in-browser WASM/GPU) → [three-vrm](https://github.com/pixiv/three-vrm).
- It uses five 2D points (nose, ears, shoulders) relative to your own calibration: ear height above the shoulder line, shoulder width (leaning in), shoulder line angle, and nose-vs-ears depth to refuse side or back views.

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

- `public/fumiriya.vrm` (default, 桜田 史利矢) and `public/shino.vrm` (千駄ヶ谷 篠): VRoid Project sample avatars whose VRoid Hub license allows modification, redistribution and commercial use with no credit required, from the [VRoid Studio CC0 models](https://opengameart.org/content/vroid-studio-cc0-models) collection. Hair colour and glasses are applied in the page.
- `public/template.vrm`: neutral body built by Single-Photo-VRM's `template/build.sh` from [NAVER Anny](https://github.com/naver/anny) (Apache-2.0 code; MPFB2 / MakeHuman CC0 assets).
- Test data, downloaded by `tests/fetch_data.sh`: photos and videos from [Pexels](https://www.pexels.com/license/); [MultiPosture](https://zenodo.org/records/14230872) (CC BY 4.0).
