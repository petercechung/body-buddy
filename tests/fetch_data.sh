#!/usr/bin/env bash
# Download the pose model and the test media used in FINDINGS.md.
# Media are not redistributed in this repo: Pexels photos/videos under the Pexels License,
# MultiPosture (Zenodo 10.5281/zenodo.14230872) under CC BY 4.0.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models photos side squat videos multiposture

[ -f models/pose_landmarker_heavy.task ] || curl -sSL -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

photo() { [ -f "$1/$2.jpg" ] || curl -sSL -o "$1/$2.jpg" "https://images.pexels.com/photos/$2/pexels-photo-$2.jpeg?w=1200"; }
for id in 12899164 12899195 12899199 12903024 12903153 12903161 3987030 3987031 3966781 8468122 21336099 6345310; do photo photos $id; done
for id in 3783744 12902945 7752896 27230118 7014340 5905879 7267214 6393019 3932570 546275 7256315 12899195 12903153 12903161 6578424 4238496; do photo side $id; done
for id in 7746282 7746286 30246175 30246176 8846119 8846586 8846530 4803865 4803862 8401808 8401827 4384679 8402239 6339648 4908608 4720566 4720555 8032893; do photo squat $id; done

for f in 8343371/8343371-sd_960_540_25fps 6967424/6967424-sd_960_540_25fps 6245972/6245972-sd_960_506_25fps \
         4053224/4053224-sd_960_540_25fps 6454275/6454275-sd_540_960_24fps 4267356/4267356-sd_960_540_30fps \
         4158861/4158861-sd_540_960_24fps 8836849/8836849-sd_506_960_25fps; do
  [ -f "videos/$(basename "$f").mp4" ] || curl -sSL -o "videos/$(basename "$f").mp4" "https://videos.pexels.com/video-files/$f.mp4"
done

[ -f multiposture/data.csv ] || curl -sSL -o multiposture/data.csv "https://zenodo.org/records/14230872/files/data.csv?download=1"
echo "done"
