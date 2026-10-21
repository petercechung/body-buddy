"""Run the demos' detection logic frame by frame over video clips.

stretch: Desk Buddy's ear tilt / head turn / shrug measures against the first second as calibration,
         reporting which of the five stretch steps would have been recognised (held 3 s? or ever).
squat:   the squat demo's 3D knee angle and rep counter (down below 120°, up above 160°).
Each clip gets a timeline image with a filmstrip, so the numbers can be checked by eye.
"""

import math
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
MODEL = HERE / "models/pose_landmarker_heavy.task"  # fetched by fetch_data.sh
UP = 140  # a knee angle above this counts as standing up again; front views rarely read past 150
NOSE, LEAR, REAR, LS, RS, LH, RH, LK, RK, LA, RA = 0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28


def frames(path, fps=10):
    """Decode a clip to RGB frames at a fixed rate with ffmpeg."""
    import subprocess

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return arr, w, h


def angle3(a, b, c):
    u, v = np.subtract(a, b), np.subtract(c, b)
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def stretch_series(lm, aspect):
    p = lambda i: np.array([lm[i].x * aspect, lm[i].y])
    sw = np.linalg.norm(p(LS) - p(RS))
    ear_mid, sh_mid = (p(LEAR) + p(REAR)) / 2, (p(LS) + p(RS)) / 2
    return {
        "earTilt": math.degrees(math.atan2(p(REAR)[1] - p(LEAR)[1], abs(p(REAR)[0] - p(LEAR)[0]))),
        "yaw": (p(NOSE)[0] - ear_mid[0]) / sw,
        "neck": (sh_mid[1] - ear_mid[1]) / sw,
    }


def squat_series(world):
    w = lambda i: (world[i].x, world[i].y, world[i].z)
    return {"knee": (angle3(w(LH), w(LK), w(LA)) + angle3(w(RH), w(RK), w(RA))) / 2}


def analyse(path, mode, fps=10):
    arr, w, h = frames(path, fps)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    series = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for k, frame in enumerate(arr):
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)), int(k * 1000 / fps))
            if not res.pose_landmarks:
                series.append(None)
                continue
            series.append(stretch_series(res.pose_landmarks[0], w / h) if mode == "stretch" else squat_series(res.pose_world_landmarks[0]))
    return arr, series


def stretch_report(series, fps):
    valid = [s for s in series if s]
    base = {k: float(np.median([s[k] for s in valid[: fps]])) for k in valid[0]}
    checks = {
        "頭右倒": lambda s: s["earTilt"] - base["earTilt"] > 12,
        "頭左倒": lambda s: s["earTilt"] - base["earTilt"] < -12,
        # Tipping the head also lowers the ear midpoint, so a shrug only counts with the head level.
        "聳肩": lambda s: s["neck"] / base["neck"] < 0.88 and abs(s["earTilt"] - base["earTilt"]) < 8,
        "轉右": lambda s: s["yaw"] - base["yaw"] < -0.18,
        "轉左": lambda s: s["yaw"] - base["yaw"] > 0.18,
    }
    out = {}
    for name, ok in checks.items():
        run = best = 0
        for s in series:
            run = run + 1 if s and ok(s) else 0
            best = max(best, run)
        out[name] = f"最長連續 {best / fps:.1f}s" + ("（≥2s 會過關）" if best >= 2 * fps else "")
    return out, {k: [None if s is None else s[k] - base[k] for s in series] for k in base}


def squat_report(series):
    reps, down, deepest, log = 0, False, 180, []
    for s in series:
        if not s:
            continue
        a = s["knee"]
        if not down and a < 120:
            down, deepest = True, a
        if down:
            deepest = min(deepest, a)
            if a > UP:
                down, reps = False, reps + 1
                log.append(round(deepest))
    return {"reps": reps, "deepest_each": log}, {"knee": [None if s is None else s["knee"] for s in series]}


def draw(arr, lines, path, fps):
    W, H = 1200, 260
    img = Image.new("RGB", (W, H + 140), "white")
    d = ImageDraw.Draw(img)
    colors = [(220, 60, 40), (40, 120, 220), (30, 150, 90)]
    n = len(arr)
    for ci, (name, vals) in enumerate(lines.items()):
        pts = [v for v in vals if v is not None]
        lo, hi = min(pts), max(pts)
        if hi - lo < 1e-6:
            hi = lo + 1
        xy = [(k * W / max(1, n - 1), H - 10 - (v - lo) / (hi - lo) * (H - 20)) for k, v in enumerate(vals) if v is not None]
        d.line(xy, fill=colors[ci % 3], width=2)
        d.text((8, 8 + 14 * ci), f"{name}: {lo:.2f}..{hi:.2f}", fill=colors[ci % 3])
    for k in range(0, n, max(1, n // 10)):
        thumb = Image.fromarray(arr[k]).copy()
        thumb.thumbnail((W // 10 - 4, 130))
        img.paste(thumb, (int(k * W / n), H + 5))
        d.text((int(k * W / n) + 2, H + 5), f"{k / fps:.0f}s", fill=(255, 255, 0))
    img.save(path)


def main(mode, files):
    fps = 10
    out = HERE / f"out-video"
    out.mkdir(exist_ok=True)
    for f in files:
        arr, series = analyse(f, mode, fps)
        found = sum(s is not None for s in series)
        report, lines = stretch_report(series, fps) if mode == "stretch" else squat_report(series)
        print(Path(f).stem[:7], f"{len(series) / fps:.0f}s, 偵測到人 {found}/{len(series)} 幀", report)
        draw(arr, lines, out / f"{Path(f).stem[:7]}-{mode}.jpg", fps)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
