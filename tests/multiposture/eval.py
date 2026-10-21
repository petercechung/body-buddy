"""Desk Buddy's front-camera posture rule on MultiPosture (Zenodo 14230872, CC BY 4.0).

Per subject, the median of their upright (TUP) frames stands in for the calibration
step; every frame is then judged with the same formulas and thresholds as
fitness-demo/src/posture.js.
"""

import csv
import math
import statistics as st
from collections import defaultdict

SLOUCH_FULL = 0.15  # neck ratio drop that counts as fully slouched
LEAN_FULL = 0.25    # shoulder width growth that counts as fully leaning in
TILT_FULL = 12      # degrees
FACING_MAX = 0.0  # this dataset uses hip-centred metric coordinates, where every subject sits near -0.12; the photo-tuned -0.12 only fits image coordinates


def metrics(r):
    g = lambda k: float(r[k])
    ls, rs = (g("left_shoulder_x"), g("left_shoulder_y")), (g("right_shoulder_x"), g("right_shoulder_y"))
    sw = math.dist(ls, rs)
    ear_y = (g("left_ear_y") + g("right_ear_y")) / 2
    return {
        "sw": sw,
        "neck": ((ls[1] + rs[1]) / 2 - ear_y) / sw,
        "tilt": math.degrees(math.atan2(rs[1] - ls[1], abs(rs[0] - ls[0]))),
        "facing": g("nose_z") - (g("left_ear_z") + g("right_ear_z")) / 2,
    }


def main():
    rows = list(csv.DictReader(open("data.csv")))
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append((r["upperbody_label"], metrics(r)))

    flagged = defaultdict(lambda: [0, 0])  # label -> [flagged as bad, judged]
    not_facing = defaultdict(int)
    per_subject = {}
    for subject, frames in by_subject.items():
        upright = [m for label, m in frames if label == "TUP"]
        base = {k: st.median(m[k] for m in upright) for k in ("sw", "neck", "tilt")}
        hits = [0, 0]
        for label, m in frames:
            if m["facing"] > FACING_MAX:
                not_facing[label] += 1
                continue
            slouch = min(1, max(0, (1 - m["neck"] / base["neck"]) / SLOUCH_FULL))
            lean = min(1, max(0, (m["sw"] / base["sw"] - 1) / LEAN_FULL))
            off = max(slouch, lean, abs(m["tilt"] - base["tilt"]) / TILT_FULL)
            bad = off > 0.5
            flagged[label][0] += bad
            flagged[label][1] += 1
            if label == "TLF":
                hits[0] += bad
                hits[1] += 1
        per_subject[subject] = hits

    names = {"TUP": "坐直", "TLF": "前傾", "TLB": "後仰", "TLL": "左傾", "TLR": "右傾"}
    print("label  判為不良 / 有判斷   沒正對鏡頭")
    for label in ("TUP", "TLF", "TLB", "TLL", "TLR"):
        b, n = flagged[label]
        print(f"{names[label]}  {b:4d}/{n:4d} = {b / n:5.1%}   {not_facing[label]}")
    print("前傾被抓到的比例，每位受試者：")
    print({s: f"{h}/{n}" for s, (h, n) in sorted(per_subject.items(), key=lambda kv: int(kv[0]))})


if __name__ == "__main__":
    main()
