#!/usr/bin/env bash
# Post-process one episode's POEM output: temporal smoothing + rigid-bone fit,
# then re-score and publish.
#
#   scripts/stabilise_zed_episode.sh <episode_tag> <path/to/egocentric-hand-stabilisation>
#
# POEM-v2 on a ~12 cm egocentric baseline produces a hand that is roughly in the
# right place but neither rigid nor temporally smooth. The stabiliser fixes both
# (it cannot recover accuracy the baseline never captured -- see the QUALITY file).
set -uo pipefail

TAG="${1:?usage: stabilise_zed_episode.sh <episode_tag> <stab_repo>}"
STAB="${2:?missing path to egocentric-hand-stabilisation checkout}"

BUCKET="stage-humyn-egocentric-stereo-data"
DEST="s3://$BUCKET/labelling_results/head_pose_POEM-V2/$TAG"
PROFILE="${AWS_PROFILE_OVERRIDE:-stage}"
REPO="/home/ec2-user/projects/POEM-v2"
WORK="${WORK_ROOT:-/home/ec2-user/zed_work}/$TAG"

cd "$REPO" || exit 1
BASE="$WORK/${TAG}_poem_enhanced_keypoints.npz"
[ -s "$BASE" ] || { echo "$TAG: no converted npz at $BASE"; exit 1; }

echo "===== stabilise $TAG ====="

echo "-- temporal smoothing (zero-phase, gap aware)"
PYTHONPATH="$STAB" python "$STAB/temporal_smooth.py" \
    --npz "$BASE" --out "$WORK/${TAG}_smooth.npz" || exit 1

echo "-- rigid-bone fit"
PYTHONPATH="$STAB" python "$STAB/rigidify.py" \
    --npz "$WORK/${TAG}_smooth.npz" \
    --out "$WORK/${TAG}_poem_enhanced_keypoints_stabilised.npz" || exit 1

echo "-- re-score"
python -m tool.zed_quality --work "$WORK" --tag "$TAG" \
    --stabilised "$WORK/${TAG}_poem_enhanced_keypoints_stabilised.npz" \
    --baseline "$BASE" \
    --out "$WORK/${TAG}_poem_QUALITY.json" || exit 1

echo "-- upload"
up() { aws --profile "$PROFILE" s3 cp "$1" "$DEST/$2" --quiet && echo "   $DEST/$2"; }
up "$WORK/${TAG}_poem_enhanced_keypoints_stabilised.npz" "${TAG}_poem_enhanced_keypoints_stabilised.npz"
up "$WORK/${TAG}_poem_enhanced_keypoints.npz"            "${TAG}_poem_enhanced_keypoints.npz"
up "$WORK/${TAG}_poem_QUALITY.json"                      "${TAG}_poem_QUALITY.json"

echo "===== $TAG stabilised ====="
