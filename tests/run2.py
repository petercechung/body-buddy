"""Photo checks for the two side-camera questions.

side:  head-neck angle (ear-shoulder line vs horizontal) on side-view desk photos.
squat: knee angle two ways -- 3D world landmarks (what the squat demo uses) and the
       2D image angle (close to the truth only when the camera sees the leg from the side).
Writes annotated thumbnails and a contact sheet per mode.
"""

import math
import sys
from pathlib import Path

import mediapipe as mp
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
MODEL = HERE / "models/pose_landmarker_heavy.task"  # fetched by fetch_data.sh
NOSE, LEAR, REAR, LS, RS, LH, RH, LK, RK, LA, RA = 0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28


def angle(a, b, c):
    """Angle at b, degrees, for points given as tuples of equal length."""
    u = [x - y for x, y in zip(a, b)]
    v = [x - y for x, y in zip(c, b)]
    dot = sum(x * y for x, y in zip(u, v))
    return math.degrees(math.acos(max(-1, min(1, dot / (math.hypot(*u) * math.hypot(*v))))))


def view_of(lm, aspect):
    """Shoulder width over torso length in the image: about 0.6+ facing the camera, near 0 side-on."""
    p = lambda i: (lm[i].x * aspect, lm[i].y)
    sw = math.dist(p(LS), p(RS))
    torso = math.dist(((p(LS)[0] + p(RS)[0]) / 2, (p(LS)[1] + p(RS)[1]) / 2), ((p(LH)[0] + p(RH)[0]) / 2, (p(LH)[1] + p(RH)[1]) / 2))
    return sw / torso if torso else 0


def side_metrics(lm, aspect):
    # Use the side facing the camera (higher visibility).
    ear, sh = (LEAR, LS) if lm[LEAR].visibility + lm[LS].visibility >= lm[REAR].visibility + lm[RS].visibility else (REAR, RS)
    ex, ey = lm[ear].x * aspect, lm[ear].y
    sx, sy = lm[sh].x * aspect, lm[sh].y
    cva = math.degrees(math.atan2(sy - ey, abs(ex - sx)))
    return {"head_neck_deg": cva, "view": view_of(lm, aspect)}, [(sh, ear)]


def squat_metrics(lm, world, aspect):
    out = {"view": view_of(lm, aspect)}
    for name, (h, k, a) in {"L": (LH, LK, LA), "R": (RH, RK, RA)}.items():
        out[f"knee3d_{name}"] = angle(*[(world[i].x, world[i].y, world[i].z) for i in (h, k, a)])
        out[f"knee2d_{name}"] = angle(*[(lm[i].x * aspect, lm[i].y) for i in (h, k, a)])
    return out, [(LH, LK), (LK, LA), (RH, RK), (RK, RA)]


def main(mode, folder):
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    out = HERE / f"out-{mode}"
    out.mkdir(exist_ok=True)
    thumbs = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for path in sorted(Path(folder).glob("*.jpg")):
            image = mp.Image.create_from_file(str(path))
            res = landmarker.detect(image)
            if not res.pose_landmarks:
                print(path.stem, "no person")
                continue
            lm, world = res.pose_landmarks[0], res.pose_world_landmarks[0]
            aspect = image.width / image.height
            m, lines = side_metrics(lm, aspect) if mode == "side" else squat_metrics(lm, world, aspect)
            print(path.stem, {k: round(v, 2) for k, v in m.items()})
            pic = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(pic)
            px = lambda i: (lm[i].x * pic.width, lm[i].y * pic.height)
            for a, b in lines:
                draw.line([px(a), px(b)], fill=(255, 60, 40), width=8)
            pic.thumbnail((320, 320))
            label = " ".join(f"{k}={v:.0f}" if abs(v) >= 2 else f"{k}={v:.2f}" for k, v in m.items())
            thumbs.append((pic, f"{path.stem} {label}"))
    W, cw = 4, 320
    sheet = Image.new("RGB", (W * cw, ((len(thumbs) + W - 1) // W) * (cw + 30)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (pic, label) in enumerate(thumbs):
        x, y = (k % W) * cw, (k // W) * (cw + 30)
        sheet.paste(pic, (x, y))
        d.text((x + 4, y + cw + 4), label[:60], fill="black")
        d.text((x + 4, y + cw + 16), label[60:120], fill="black")
    sheet.save(HERE / f"sheet-{mode}.jpg")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
