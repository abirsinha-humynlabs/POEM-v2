#!/usr/bin/env bash
# Run the video-only path on one ZED stereo episode and publish the results.
#
#   scripts/run_zed_episode.sh <s3_input_prefix> <episode_tag> [max_frames]
#
# Both hands are processed (two passes): an egocentric capture normally shows
# either, and POEM-v2 handles one hand per run. Results go to
# s3://$BUCKET/labelling_results/head_pose_POEM-V2/<episode_tag>/ -- the only
# prefix this job is allowed to write to.
set -uo pipefail

SRC="${1:?usage: run_zed_episode.sh <s3_prefix> <episode_tag> [max_frames]}"
TAG="${2:?missing episode tag}"
MAXF="${3:-0}"

BUCKET="stage-humyn-egocentric-stereo-data"
DEST="s3://$BUCKET/labelling_results/head_pose_POEM-V2/$TAG"
PROFILE="${AWS_PROFILE_OVERRIDE:-stage}"
REPO="/home/ec2-user/projects/POEM-v2"
WORK="${WORK_ROOT:-/home/ec2-user/zed_work}/$TAG"
CKPT="$REPO/checkpoints/medium.pth.tar"

cd "$REPO" || exit 1
mkdir -p "$WORK"

echo "===== $TAG ====="
echo "source: $SRC"
echo "dest:   $DEST"

# ---- 1. fetch the stereo pair + calibration -------------------------------
for f in left_eye.mp4 right_eye.mp4 calibration.json; do
  if [ ! -s "$WORK/$f" ]; then
    echo "-- downloading $f"
    aws --profile "$PROFILE" s3 cp "${SRC}${f}" "$WORK/$f" --quiet || { echo "FAILED to fetch $f"; exit 1; }
  fi
done

# ---- 2. calibration -------------------------------------------------------
read -r W H FPS N < <(python - "$WORK/left_eye.mp4" <<'PY'
import sys, cv2
v = cv2.VideoCapture(sys.argv[1])
print(int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)),
      v.get(cv2.CAP_PROP_FPS), int(v.get(cv2.CAP_PROP_FRAME_COUNT)))
PY
)
echo "-- video: ${W}x${H} @ ${FPS} fps, ${N} frames"
python -m tool.zed_episode calib --zed "$WORK/calibration.json" --out "$WORK/calib.json" \
    --width "$W" --height "$H" || exit 1
python -m tool.make_calib check --calib "$WORK/calib.json" || exit 1

LIMIT=()
[ "$MAXF" != "0" ] && LIMIT=(--max-frames "$MAXF")

# ---- 3. dry run before spending GPU time ----------------------------------
echo "-- dry run"
python -m tool.infer_video --calib "$WORK/calib.json" --video-dir "$WORK" \
    --bbox-backend mediapipe --hand-side rh --bbox-strict-side --dry-run \
    --max-frames 60 --out "$WORK/dry" 2>&1 | tail -12

# ---- 4. inference, one pass per hand --------------------------------------
for SIDE in rh lh; do
  echo "-- inference: $SIDE"
  python -m tool.infer_video \
      --calib "$WORK/calib.json" --video-dir "$WORK" \
      --bbox-backend mediapipe --hand-side "$SIDE" --bbox-strict-side \
      --cfg config/release/eval_single.yaml --model medium --reload "$CKPT" \
      --device cuda:0 "${LIMIT[@]}" \
      --overlay --overlay-views left_eye --overlay-fps "$FPS" --overlay-scale 1.0 \
      --out "$WORK/$SIDE" 2>&1 | grep -vE "INFO\]|WARNING\]" | tail -8
done

# ---- 5. renderer-schema npz ----------------------------------------------
echo "-- convert"
python -m tool.zed_episode convert \
    --npz "$WORK/rh/keypoints.npz" "$WORK/lh/keypoints.npz" \
    --calib "$WORK/calib.json" --master left_eye \
    --out "$WORK/${TAG}_poem_enhanced_keypoints.npz" || exit 1

# ---- 5b. quality report ---------------------------------------------------
echo "-- quality"
python -m tool.zed_quality --work "$WORK" --tag "$TAG" \
    --out "$WORK/${TAG}_poem_QUALITY.json" || exit 1

# ---- 6. publish -----------------------------------------------------------
echo "-- upload"
up() { aws --profile "$PROFILE" s3 cp "$1" "$DEST/$2" --quiet && echo "   $DEST/$2"; }
up "$WORK/${TAG}_poem_enhanced_keypoints.npz" "${TAG}_poem_enhanced_keypoints.npz"
up "$WORK/calib.json"                          "${TAG}_calib.json"
up "$WORK/${TAG}_poem_QUALITY.json"            "${TAG}_poem_QUALITY.json"
for SIDE in rh lh; do
  [ -s "$WORK/$SIDE/keypoints.npz" ]  && up "$WORK/$SIDE/keypoints.npz"  "${TAG}_poem_${SIDE}_keypoints.npz"
  [ -s "$WORK/$SIDE/report.json" ]    && up "$WORK/$SIDE/report.json"    "${TAG}_poem_${SIDE}_report.json"
  [ -s "$WORK/$SIDE/overlay.mp4" ]    && up "$WORK/$SIDE/overlay.mp4"    "${TAG}_poem_${SIDE}_overlay_left_eye.mp4"
done

echo "===== $TAG done ====="
