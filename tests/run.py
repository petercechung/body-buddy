"""Run the Desk Buddy posture metrics on still photos.

Same landmarks and formulas as fitness-demo/src/posture.js `measure()`:
nose, ears and shoulders from MediaPipe Pose; metrics in shoulder widths.
Writes annotated copies to out/ and a table to results.csv.
"""

import csv
import math
import sys
from pathlib import Path

import mediapipe as mp
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
MODEL = HERE / "models/pose_landmarker_heavy.task"  # fetched by fetch_data.sh
NOSE, LEAR, REAR, LS, RS = 0, 7, 8, 11, 12


def measure(lm, aspect):
    pt = lambda i: (lm[i].x * aspect, lm[i].y)
    ls, rs = pt(LS), pt(RS)
    smid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    emid = ((pt(LEAR)[0] + pt(REAR)[0]) / 2, (pt(LEAR)[1] + pt(REAR)[1]) / 2)
    sw = math.dist(ls, rs)
    return {
        "shoulderW": sw,
        "neck": (smid[1] - emid[1]) / sw,
        "tilt": math.degrees(math.atan2(rs[1] - ls[1], abs(rs[0] - ls[0]))),
        "yaw": (pt(NOSE)[0] - emid[0]) / sw,
        "vis": min(lm[i].visibility for i in (NOSE, LEAR, REAR, LS, RS)),
    }


def main(photos):
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    rows = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for path in photos:
            image = mp.Image.create_from_file(str(path))
            res = landmarker.detect(image)
            if not res.pose_landmarks:
                rows.append({"photo": path.stem, "found": False})
                continue
            lm = res.pose_landmarks[0]
            m = measure(lm, image.width / image.height)
            rows.append({"photo": path.stem, "found": True, **{k: round(v, 3) for k, v in m.items()}})
            pic = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(pic)
            px = lambda i: (lm[i].x * pic.width, lm[i].y * pic.height)
            for a, b in ((LS, RS), (LEAR, REAR), (LS, LEAR), (RS, REAR)):
                draw.line([px(a), px(b)], fill=(255, 80, 60), width=6)
            draw.text((20, 20), f"neck {m['neck']:.2f}  tilt {m['tilt']:.0f}  vis {m['vis']:.2f}", fill=(255, 255, 0))
            pic.thumbnail((600, 600))
            pic.save(out / f"{path.stem}.jpg")
    with open(HERE / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo", "found", "shoulderW", "neck", "tilt", "yaw", "vis"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sorted(Path(sys.argv[1]).glob("*.jpg")))
