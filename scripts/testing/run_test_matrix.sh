#!/usr/bin/env bash
# Tiered test runner for the video-only inference path.
#
#   tier1  calibration / geometry / boxes / crops / video IO / dry-run pipeline
#          -> numpy + opencv + pytest. Runs on a laptop, no GPU, no torch.
#   tier2  batch tensors + DLT parity against the repo's torch implementation
#          -> CPU torch is enough.
#   tier3  model construction and forward passes
#          -> needs pytorch3d + manotorch + assets/mano_v1_2 (the GPU box).
#   tier4  accuracy checks on the released example data
#          -> needs POEM_CHECKPOINT and POEM_EXAMPLE_DATA.
#
# Usage:
#   scripts/testing/run_test_matrix.sh            # every tier the env supports
#   scripts/testing/run_test_matrix.sh tier1      # one tier
#   POEM_CHECKPOINT=... POEM_EXAMPLE_DATA=... scripts/testing/run_test_matrix.sh tier4
#
# Tests whose dependencies are missing are SKIPPED with a reason, never failed,
# so the same command is meaningful on a laptop and on a GPU host.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
PYTHON="${PYTHON:-python3}"
TIER="${1:-all}"
STATUS=0

run() {
  echo ""
  echo "=============================================================="
  echo "== $1"
  echo "=============================================================="
  shift
  "$PYTHON" -m pytest "$@"
  local rc=$?
  # 5 == "no tests collected", which is not a failure for a filtered tier
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then STATUS=$rc; fi
}

if [ "$TIER" = "all" ] || [ "$TIER" = "tier1" ]; then
  run "tier 1: geometry / calibration / boxes / video / pipeline (no GPU)" \
    tests/test_calib_io.py tests/test_geometry.py tests/test_bbox.py \
    tests/test_views.py tests/test_video_io.py tests/test_pipeline.py -q
fi

if [ "$TIER" = "all" ] || [ "$TIER" = "tier2" ]; then
  run "tier 2: torch batch contract (CPU torch)" tests/test_torch_batch.py -q
fi

if [ "$TIER" = "all" ] || [ "$TIER" = "tier3" ]; then
  run "tier 3: model build + forward (needs the full env)" tests/test_model_forward.py -v
fi

if [ "$TIER" = "all" ] || [ "$TIER" = "tier4" ]; then
  run "tier 4: released example data (needs checkpoint + data)" tests/test_example_data.py -v -s
fi

echo ""
if [ $STATUS -eq 0 ]; then
  echo "== test matrix: OK (skips above indicate tiers this machine cannot run)"
else
  echo "== test matrix: FAILURES (exit $STATUS)"
fi
exit $STATUS
