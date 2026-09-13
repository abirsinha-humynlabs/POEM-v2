"""T10 - accuracy-level checks on the released example data (GPU box + checkpoint).

Synthetic captures prove the geometry; only real imagery proves the *model*.
This tier runs the published multi-view capture
(https://huggingface.co/kelvin34501/POEM-v2_example_data) through the
video-only path and asserts properties that must hold for a correct run and
would break under a mis-wired camera, a unit error or a bad crop:

* the network's 2D keypoints and its reprojected 3D agree to a few pixels;
* the hand is in front of the master camera and metric-sized;
* bone lengths are stable over time (a rigid hand cannot change size);
* the wrist does not teleport between consecutive frames.

Setup::

    export POEM_EXAMPLE_DATA=/path/to/extracted/example_data
    export POEM_CHECKPOINT=/path/to/checkpoints/medium.pth.tar
    pytest tests/test_example_data.py -v

Layout expected inside ``POEM_EXAMPLE_DATA`` (as shipped)::

    data/<sequence>/<camera>.mkv          (or data_v2/, as currently released)
    human_mask_hand/<sequence>/<camera>/bbox/%05d.npy
    calib/calib__*/cam_intr/<camera>.pkl,  .../cam_extr/<camera>.pkl
    hand_labels.json
"""

import glob
import json
import os

import numpy as np
import pytest

from conftest import CHECKPOINT, EXAMPLE_DATA, requires_checkpoint, requires_example_data
from tool.poemkit.bbox import NpyBBoxProvider
from tool.poemkit.calib import load_poem_pkl_calib
from tool.poemkit.render import HAND_BONES

CFG = "config/release/eval_single.yaml"
MAX_FRAMES = int(os.environ.get("POEM_TEST_FRAMES", "20"))

pytestmark = [requires_example_data, pytest.mark.realdata]


@pytest.fixture(scope="module")
def example_layout():
    root = EXAMPLE_DATA
    # the released tarball has shipped the sequences under both "data" and
    # (currently) "data_v2"; accept whichever this copy actually has.
    data_dir = next(
        (os.path.join(root, name) for name in ("data", "data_v2") if os.path.isdir(os.path.join(root, name))),
        os.path.join(root, "data"),
    )
    mask_dir = os.path.join(root, "human_mask_hand")
    calib_dirs = sorted(glob.glob(os.path.join(root, "calib", "calib__*")))
    labels_path = os.path.join(root, "hand_labels.json")

    for path in (data_dir, mask_dir, labels_path):
        if not os.path.exists(path):
            pytest.skip(f"example data incomplete: missing {path}")
    if not calib_dirs:
        pytest.skip(f"no calib__* directory under {os.path.join(root, 'calib')}")

    with open(labels_path) as ifs:
        labels = json.load(ifs)
    sequences = sorted(s for s in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, s)))
    if not sequences:
        pytest.skip(f"no sequences under {data_dir}")

    return {
        "root": root,
        "data_dir": data_dir,
        "mask_dir": mask_dir,
        "calib_dir": calib_dirs[0],
        "labels": labels,
        "sequences": sequences,
    }


def _sequence_inputs(layout, sequence):
    """(rig, {camera: video_path}, bbox_root, hand_side) for one sequence."""
    rig = load_poem_pkl_calib(layout["calib_dir"])
    seq_dir = os.path.join(layout["data_dir"], sequence)
    videos = {}
    for cam in rig.names:
        for ext in (".mkv", ".mp4"):
            candidate = os.path.join(seq_dir, cam + ext)
            if os.path.isfile(candidate):
                videos[cam] = candidate
                break
    if len(videos) < 2:
        pytest.skip(f"sequence {sequence} has {len(videos)} video(s) matching the calibration cameras")

    label = str(layout["labels"].get(sequence, "right")).lower()
    hand_side = "rh" if label.startswith("r") else "lh"
    return rig.subset(list(videos)), videos, os.path.join(layout["mask_dir"], sequence), hand_side


def _first_usable_frame(bbox_root, cameras, num_frames=600, min_views=2):
    """First frame whose *shipped* boxes cover ``min_views`` cameras.

    The released clips do not begin with the hand in shot. The mask pipeline
    still writes a file for those frames, but it holds an empty array, so the
    opening frames of a sequence can have fewer than two boxed views and are
    skipped by the runner. Starting a measurement at frame 0 would therefore
    grade the dead zone at the head of the clip rather than the model. Returns
    0 when nothing better is found, leaving the caller's own assertions to fail
    loudly rather than silently measuring nothing.
    """
    provider = NpyBBoxProvider(bbox_root)
    for frame_id in range(num_frames):
        boxed = sum(1 for cam in cameras if provider.get(cam, frame_id) is not None)
        if boxed >= min_views:
            return frame_id
    return 0


def test_layout_and_calibration(example_layout):
    rig = load_poem_pkl_calib(example_layout["calib_dir"])
    assert len(rig) >= 2, "the released rig should have several cameras"
    assert rig.sanity_check() == [], "the shipped calibration must pass the metric sanity checks"

    for (a, b), dist in rig.baselines().items():
        assert 0.05 < dist < 5.0, f"baseline {a}-{b} = {dist} m looks non-metric"


def test_dry_run_over_a_real_sequence(example_layout, tmp_path):
    """No torch needed: does the released capture survive the video-only path?"""
    from tool.infer_video import run_sequence

    sequence = example_layout["sequences"][0]
    rig, videos, bbox_root, hand_side = _sequence_inputs(example_layout, sequence)

    report = run_sequence(
        rig=rig,
        video_paths=videos,
        out_dir=str(tmp_path / "dry"),
        hand_side=hand_side,
        bbox_backend="npy",
        bbox_root=bbox_root,
        dry_run=True,
        frame_start=_first_usable_frame(bbox_root, rig.names),
        max_frames=MAX_FRAMES,
        progress=False,
    )

    assert report["frames_predicted"] > 0, f"no usable frames: {report['skipped_examples']}"
    assert report["crop_consistency_px_max"] < 1e-6


@requires_checkpoint
@pytest.mark.slow
def test_inference_is_metric_and_temporally_stable(example_layout, tmp_path):
    from tool.infer_video import run_sequence
    from tool.poemkit.runner import PoemRunner

    sequence = example_layout["sequences"][0]
    rig, videos, bbox_root, hand_side = _sequence_inputs(example_layout, sequence)
    size = next((s for s in ("small", "medium", "large", "huge") if s in CHECKPOINT), "medium")

    runner = PoemRunner(cfg_path=CFG, checkpoint=CHECKPOINT, model_size=size, device="auto", verbose=False)
    out_dir = str(tmp_path / "run")
    report = run_sequence(
        rig=rig,
        video_paths=videos,
        out_dir=out_dir,
        hand_side=hand_side,
        bbox_backend="npy",
        bbox_root=bbox_root,
        predictor=runner,
        frame_start=_first_usable_frame(bbox_root, rig.names),
        max_frames=MAX_FRAMES,
        progress=False,
    )

    assert report["frames_predicted"] >= max(2, MAX_FRAMES // 2)
    # 2D/3D agreement: the heatmap keypoints and the reprojected mesh joints
    # should sit within a few pixels of each other on a well-calibrated rig
    assert report["reproj_px_mean"] < 20.0, f"2D/3D disagreement {report['reproj_px_mean']:.1f} px"

    payload = np.load(os.path.join(out_dir, "keypoints.npz"))
    joints = payload["joints_master"]  # (F, 21, 3) meters, master camera frame
    assert np.all(np.isfinite(joints))

    # in front of the camera, inside the trained depth range
    assert np.all(joints[..., 2] > 0.05)
    assert np.percentile(joints[..., 2], 95) < 1.5

    # metric size: a human hand spans ~15-25 cm
    spans = np.linalg.norm(joints.max(axis=1) - joints.min(axis=1), axis=-1)
    assert 0.10 < float(np.median(spans)) < 0.35, f"median hand span {np.median(spans):.3f} m"

    # rigid bones: per-bone length must barely vary across frames
    bone_lengths = np.stack([np.linalg.norm(joints[:, a] - joints[:, b], axis=-1) for a, b in HAND_BONES], axis=1)
    assert np.all(bone_lengths > 0.005), "degenerate (zero-length) bones"
    assert float(np.median(bone_lengths.std(axis=0))) < 0.006, "bone lengths drift: the skeleton is not rigid"

    # temporal continuity: the wrist cannot jump between consecutive frames
    if joints.shape[0] >= 3:
        wrist_step = np.linalg.norm(np.diff(joints[:, 0], axis=0), axis=-1)
        assert float(np.percentile(wrist_step, 90)) < 0.05, f"wrist jitter {np.percentile(wrist_step, 90):.3f} m/frame"


@requires_checkpoint
@pytest.mark.slow
def test_left_hand_sequence_if_present(example_layout, tmp_path):
    """The mirroring path on real data, if the release contains a left-hand sequence."""
    from tool.infer_video import run_sequence
    from tool.poemkit.runner import PoemRunner

    left = [s for s in example_layout["sequences"]
            if str(example_layout["labels"].get(s, "")).lower().startswith("l")]
    if not left:
        pytest.skip("no left-hand sequence in this release")

    rig, videos, bbox_root, hand_side = _sequence_inputs(example_layout, left[0])
    assert hand_side == "lh"
    size = next((s for s in ("small", "medium", "large", "huge") if s in CHECKPOINT), "medium")
    runner = PoemRunner(cfg_path=CFG, checkpoint=CHECKPOINT, model_size=size, device="auto", verbose=False)

    report = run_sequence(rig=rig, video_paths=videos, out_dir=str(tmp_path / "lh"), hand_side=hand_side,
                          bbox_backend="npy", bbox_root=bbox_root, predictor=runner,
                          frame_start=_first_usable_frame(bbox_root, rig.names),
                          max_frames=min(MAX_FRAMES, 10), progress=False)
    assert report["frames_predicted"] > 0
    assert report["reproj_px_mean"] < 25.0


@requires_checkpoint
@pytest.mark.slow
def test_view_count_changes_the_answer_but_not_the_scale(example_layout, tmp_path):
    """Two views vs. all views: results should be close, and both metric.

    This is the practical question for a stereo rig -- how much is lost by
    dropping to the minimum two views.
    """
    from tool.infer_video import run_sequence
    from tool.poemkit.runner import PoemRunner

    sequence = example_layout["sequences"][0]
    rig, videos, bbox_root, hand_side = _sequence_inputs(example_layout, sequence)
    if len(rig) < 3:
        pytest.skip("needs >= 3 calibrated views to compare against a 2-view subset")

    size = next((s for s in ("small", "medium", "large", "huge") if s in CHECKPOINT), "medium")
    runner = PoemRunner(cfg_path=CFG, checkpoint=CHECKPOINT, model_size=size, device="auto", verbose=False)

    pair_names = rig.names[:2]
    # both runs must start where the *pair* is usable, so they cover the same
    # frames; the 2-view subset is the stricter of the two.
    start = _first_usable_frame(bbox_root, pair_names)

    full = run_sequence(rig=rig, video_paths=videos, out_dir=str(tmp_path / "all"), hand_side=hand_side,
                        bbox_backend="npy", bbox_root=bbox_root, predictor=runner,
                        frame_start=start, max_frames=min(MAX_FRAMES, 10), progress=False)

    pair = run_sequence(rig=rig.subset(pair_names), video_paths={n: videos[n] for n in pair_names},
                        out_dir=str(tmp_path / "pair"), hand_side=hand_side, bbox_backend="npy",
                        bbox_root=bbox_root, predictor=runner, frame_start=start,
                        max_frames=min(MAX_FRAMES, 10), progress=False)

    assert full["frames_predicted"] > 0 and pair["frames_predicted"] > 0
    a = np.load(os.path.join(str(tmp_path / "all"), "keypoints.npz"))
    b = np.load(os.path.join(str(tmp_path / "pair"), "keypoints.npz"))
    common = np.intersect1d(a["frame_ids"], b["frame_ids"])
    assert common.size > 0

    ja = a["joints_master"][np.isin(a["frame_ids"], common)]
    jb = b["joints_master"][np.isin(b["frame_ids"], common)]
    mpjpe = float(np.linalg.norm(ja - jb, axis=-1).mean())
    print(f"2-view vs {len(rig)}-view MPJPE: {mpjpe * 1000:.1f} mm")
    assert mpjpe < 0.05, f"2-view and {len(rig)}-view predictions differ by {mpjpe * 1000:.0f} mm"


def test_mediapipe_boxes_agree_with_the_shipped_boxes(example_layout):
    """If MediaPipe is installed, check it can replace the precomputed boxes --
    that is what makes a *video-only* capture (no mask pipeline) workable."""
    pytest.importorskip("mediapipe", reason="optional: pip install mediapipe")

    from tool.poemkit.bbox import MediaPipeBBoxProvider, NpyBBoxProvider
    from tool.poemkit.video import MultiViewReader

    sequence = example_layout["sequences"][0]
    rig, videos, bbox_root, hand_side = _sequence_inputs(example_layout, sequence)
    shipped = NpyBBoxProvider(bbox_root)
    detector = MediaPipeBBoxProvider(hand_side=hand_side)

    ious = []
    start = _first_usable_frame(bbox_root, rig.names)
    with MultiViewReader(videos, backend="auto") as reader:
        for frame_id in range(start, min(reader.num_frame, start + 10)):
            frames = reader.read(frame_id)
            for cam, frame in frames.items():
                ref = shipped.get(cam, frame_id)
                got = detector.get(cam, frame_id, frame)
                if ref is None or got is None:
                    continue
                x0 = max(ref[0], got[0])
                y0 = max(ref[1], got[1])
                x1 = min(ref[2], got[2])
                y1 = min(ref[3], got[3])
                inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                area_ref = (ref[2] - ref[0]) * (ref[3] - ref[1])
                area_got = (got[2] - got[0]) * (got[3] - got[1])
                ious.append(inter / max(1e-6, area_ref + area_got - inter))
    detector.close()

    if not ious:
        pytest.skip("MediaPipe found no hands in the sampled frames")
    median_iou = float(np.median(ious))
    print(f"MediaPipe vs shipped boxes: median IoU {median_iou:.2f} over {len(ious)} views")
    assert median_iou > 0.3, "MediaPipe boxes disagree badly; the crops would not match training"
