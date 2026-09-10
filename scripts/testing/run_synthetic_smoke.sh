#!/usr/bin/env bash
# End-to-end smoke test of the video-only path on generated data.
#
# Stage A (no GPU, no torch): render synthetic captures and validate them with
#   `tool.infer_video --dry-run`. This is what to run on a laptop.
# Stage B (GPU box): if a checkpoint is given, run real inference on the same
#   captures. The imagery is synthetic, so judge the *plumbing* (it completes,
#   the npz has the right shapes, nothing is NaN), not the accuracy.
#
# Usage:
#   scripts/testing/run_synthetic_smoke.sh [OUT_DIR] [CHECKPOINT] [MODEL_SIZE]
#
# Examples:
#   scripts/testing/run_synthetic_smoke.sh /tmp/poem_smoke
#   scripts/testing/run_synthetic_smoke.sh /tmp/poem_smoke checkpoints/medium.pth.tar medium
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
PYTHON="${PYTHON:-python3}"
OUT="${1:-/tmp/poem_smoke}"
CKPT="${2:-}"
MODEL="${3:-medium}"

echo "== rendering synthetic captures under $OUT"
"$PYTHON" -m scripts.testing.make_synthetic_capture --out "$OUT/stereo" --rig stereo --frames 20 >/dev/null
"$PYTHON" -m scripts.testing.make_synthetic_capture --out "$OUT/ring" --rig ring --num-cams 3 --frames 20 \
    --drop-view cam1:5,6 >/dev/null
"$PYTHON" -m scripts.testing.make_synthetic_capture --out "$OUT/lefthand" --rig stereo --frames 20 \
    --hand-side lh >/dev/null
echo "   stereo / ring(3 views, dropouts) / lefthand"

echo ""
echo "== stage A: dry runs (no model)"
for case in stereo ring lefthand; do
  side="rh"; [ "$case" = "lefthand" ] && side="lh"
  echo "-- $case"
  "$PYTHON" -m tool.infer_video \
      --calib "$OUT/$case/calib.json" \
      --video-dir "$OUT/$case" \
      --hand-side "$side" \
      --bbox-backend npy --bbox-root "$OUT/$case/bbox" \
      --dry-run --out "$OUT/$case/dry" \
    | "$PYTHON" -c "import json,sys; r=json.load(sys.stdin); print('   frames', r['frames_predicted'], 'skipped', r['frames_skipped'], 'crop_err_px', r.get('crop_consistency_px_max'), 'warnings', r['warnings'])"
done

echo ""
echo "== environment"
"$PYTHON" -m tool.infer_video --env-report --out unused

if [ -z "$CKPT" ]; then
  echo ""
  echo "== stage B skipped: no checkpoint given."
  echo "   On the GPU box:  $0 $OUT checkpoints/${MODEL}.pth.tar $MODEL"
  exit 0
fi

echo ""
echo "== stage B: real inference with $CKPT (model=$MODEL)"
for case in stereo ring lefthand; do
  side="rh"; [ "$case" = "lefthand" ] && side="lh"
  echo "-- $case"
  "$PYTHON" -m tool.infer_video \
      --calib "$OUT/$case/calib.json" \
      --video-dir "$OUT/$case" \
      --hand-side "$side" \
      --bbox-backend npy --bbox-root "$OUT/$case/bbox" \
      --cfg config/release/eval_single.yaml --model "$MODEL" --reload "$CKPT" \
      --device auto --max-frames 10 --overlay \
      --out "$OUT/$case/run" \
    | "$PYTHON" -c "import json,sys; r=json.load(sys.stdin); print('   frames', r['frames_predicted'], 'reproj_px_mean', r.get('reproj_px_mean'), 'seconds', r['seconds'])"
  "$PYTHON" - "$OUT/$case/run/keypoints.npz" <<'PYEOF'
import sys
import numpy as np
payload = np.load(sys.argv[1])
joints = payload["joints_world"]
print("   joints_world", joints.shape, "finite" if np.all(np.isfinite(joints)) else "HAS NaN/Inf",
      "| z range %.3f..%.3f m" % (payload["joints_master"][..., 2].min(), payload["joints_master"][..., 2].max()))
PYEOF
done

echo ""
echo "== done. Overlays: $OUT/<case>/run/overlay.mp4"
