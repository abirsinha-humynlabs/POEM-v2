"""T6/T7 - the end-to-end sequence runner, with and without a network.

Two modes are covered:

* ``--dry-run``: no torch at all. Validates that a capture is *runnable*
  (calibration, synchronisation, boxes, crops) and is the check to run on a
  laptop before paying for a GPU.
* full run against the ``oracle_predictor`` fixture, which returns ground
  truth shaped exactly like the real model's output. That isolates every
  non-network stage -- master bookkeeping, un-mirroring, the 2D/3D agreement
  metric, npz/json export -- and asserts it is *correct*, not merely running.
"""

import json
import os

import numpy as np
import pytest

from tool.infer_video import build_parser, frame_indices, parse_video_args, resolve_hand_side, run_sequence
from tool.poemkit.calib import load_calib_json


def _oracle(oracle_predictor, manifest):
    gt = np.load(manifest["gt"])
    return oracle_predictor(gt["joints_world"]), gt


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
def test_dry_run_validates_a_stereo_capture(stereo_capture, tmp_path):
    report = run_sequence(
        rig=load_calib_json(stereo_capture["calib"]),
        video_paths=stereo_capture["videos"],
        out_dir=str(tmp_path / "dry"),
        bbox_backend="npy",
        bbox_root=stereo_capture["bbox_root"],
        dry_run=True,
        video_backend="cv2",
        progress=False,
    )

    assert report["dry_run"] is True
    assert report["frames_predicted"] == stereo_capture["num_frames"]
    assert report["frames_skipped"] == 0
    assert report["warnings"] == []
    # the crop-intrinsics bookkeeping must be exact
    assert report["crop_consistency_px_max"] < 1e-6
    assert os.path.isfile(tmp_path / "dry" / "report.json")
    assert not os.path.exists(tmp_path / "dry" / "keypoints.npz"), "a dry run must not fake results"


def test_dry_run_skips_frames_with_too_few_views(ring_capture, tmp_path):
    report = run_sequence(
        rig=load_calib_json(ring_capture["calib"]),
        video_paths=ring_capture["videos"],
        out_dir=str(tmp_path / "dry_ring"),
        bbox_backend="npy",
        bbox_root=ring_capture["bbox_root"],
        bbox_max_age=0,  # no reuse, so the simulated drop-outs really drop
        dry_run=True,
        video_backend="cv2",
        progress=False,
    )
    # frame 3 loses two of three views -> unusable; frame 2 keeps two -> usable
    assert report["frames_skipped"] == 1
    assert report["frames_predicted"] == ring_capture["num_frames"] - 1
    assert "only 1 usable view" in report["skipped_examples"][0]["reason"]


def test_bbox_reuse_rescues_dropped_frames(ring_capture, tmp_path):
    """With reuse enabled the stale box carries the frame that would be skipped."""
    report = run_sequence(
        rig=load_calib_json(ring_capture["calib"]),
        video_paths=ring_capture["videos"],
        out_dir=str(tmp_path / "dry_reuse"),
        bbox_backend="npy",
        bbox_root=ring_capture["bbox_root"],
        bbox_max_age=5,
        dry_run=True,
        video_backend="cv2",
        progress=False,
    )
    assert report["frames_skipped"] == 0
    assert report["bbox_stats"]["reused"] == 3  # cam1 frames 2,3 + cam2 frame 3


def test_dry_run_needs_no_predictor(stereo_capture, tmp_path):
    with pytest.raises(ValueError, match="predictor"):
        run_sequence(
            rig=load_calib_json(stereo_capture["calib"]),
            video_paths=stereo_capture["videos"],
            out_dir=str(tmp_path / "nope"),
            bbox_backend="npy",
            bbox_root=stereo_capture["bbox_root"],
            dry_run=False,
            predictor=None,
            video_backend="cv2",
            progress=False,
        )


# ---------------------------------------------------------------------------
# full run against the oracle
# ---------------------------------------------------------------------------
def test_full_run_recovers_ground_truth(stereo_capture, oracle_predictor, tmp_path):
    predictor, gt = _oracle(oracle_predictor, stereo_capture)
    out_dir = str(tmp_path / "run")

    report = run_sequence(
        rig=load_calib_json(stereo_capture["calib"]),
        video_paths=stereo_capture["videos"],
        out_dir=out_dir,
        bbox_backend="json",
        bbox_json=stereo_capture["bbox_json"],
        predictor=predictor,
        video_backend="cv2",
        save_verts=True,
        progress=False,
    )

    assert report["frames_predicted"] == stereo_capture["num_frames"]
    assert predictor.calls == stereo_capture["num_frames"]
    # the network's 2D and the reprojected 3D agree, as they must for consistent data
    assert report["reproj_px_mean"] < 1e-3

    payload = np.load(os.path.join(out_dir, "keypoints.npz"))
    assert payload["joints_world"].shape == (stereo_capture["num_frames"], 21, 3)
    assert payload["joints_master"].shape == (stereo_capture["num_frames"], 21, 3)
    assert payload["verts_world"].shape == (stereo_capture["num_frames"], 778, 3)
    np.testing.assert_allclose(payload["joints_world"], gt["joints_world"], atol=1e-4)
    np.testing.assert_array_equal(payload["frame_ids"], np.arange(stereo_capture["num_frames"]))


def test_master_frame_output_is_the_master_camera_frame(stereo_capture, oracle_predictor, tmp_path):
    """``joints_master`` must equal the world output pushed through the master extrinsic."""
    from tool.poemkit.geometry import transf_points

    predictor, gt = _oracle(oracle_predictor, stereo_capture)
    rig = load_calib_json(stereo_capture["calib"])
    out_dir = str(tmp_path / "master")
    run_sequence(rig=rig, video_paths=stereo_capture["videos"], out_dir=out_dir, bbox_backend="json",
                 bbox_json=stereo_capture["bbox_json"], predictor=predictor, video_backend="cv2", progress=False)

    payload = np.load(os.path.join(out_dir, "keypoints.npz"))
    for f in range(payload["joints_world"].shape[0]):
        expected = transf_points(rig[0].T_cw, payload["joints_world"][f])
        np.testing.assert_allclose(payload["joints_master"][f], expected, atol=1e-4)


def test_left_hand_run_returns_unmirrored_world_coords(left_hand_capture, oracle_predictor, tmp_path):
    predictor, gt = _oracle(oracle_predictor, left_hand_capture)
    out_dir = str(tmp_path / "lh")

    report = run_sequence(
        rig=load_calib_json(left_hand_capture["calib"]),
        video_paths=left_hand_capture["videos"],
        out_dir=out_dir,
        hand_side="lh",
        bbox_backend="json",
        bbox_json=left_hand_capture["bbox_json"],
        predictor=predictor,
        video_backend="cv2",
        progress=False,
    )

    assert report["hand_side"] == "lh"
    assert report["reproj_px_mean"] < 1e-2, "the un-mirroring must survive the 2D/3D comparison"
    payload = np.load(os.path.join(out_dir, "keypoints.npz"))
    np.testing.assert_allclose(payload["joints_world"], gt["joints_world"], atol=1e-4)


def test_three_view_run_with_dropouts(ring_capture, oracle_predictor, tmp_path):
    predictor, gt = _oracle(oracle_predictor, ring_capture)
    out_dir = str(tmp_path / "ring_run")

    report = run_sequence(
        rig=load_calib_json(ring_capture["calib"]),
        video_paths=ring_capture["videos"],
        out_dir=out_dir,
        bbox_backend="npy",
        bbox_root=ring_capture["bbox_root"],
        bbox_max_age=0,
        predictor=predictor,
        video_backend="cv2",
        progress=False,
    )

    assert report["frames_predicted"] == ring_capture["num_frames"] - 1
    payload = np.load(os.path.join(out_dir, "keypoints.npz"))
    views = [v.split(",") for v in payload["views"].tolist()]
    assert any(len(v) == 2 for v in views), "the 2-of-3-view frames must be kept"
    assert any(len(v) == 3 for v in views)
    kept = payload["frame_ids"]
    np.testing.assert_allclose(payload["joints_world"], gt["joints_world"][kept], atol=1e-4)


def test_json_export_mirrors_the_npz(stereo_capture, oracle_predictor, tmp_path):
    predictor, _ = _oracle(oracle_predictor, stereo_capture)
    out_dir = str(tmp_path / "export")
    run_sequence(rig=load_calib_json(stereo_capture["calib"]), video_paths=stereo_capture["videos"],
                 out_dir=out_dir, bbox_backend="json", bbox_json=stereo_capture["bbox_json"],
                 predictor=predictor, video_backend="cv2", progress=False)

    with open(os.path.join(out_dir, "keypoints.json")) as ifs:
        payload = json.load(ifs)
    assert payload["units"] == "meters"
    assert len(payload["frames"]) == stereo_capture["num_frames"]
    assert np.array(payload["frames"][0]["joints_world"]).shape == (21, 3)

    npz = np.load(os.path.join(out_dir, "keypoints.npz"))
    np.testing.assert_allclose(np.array(payload["frames"][0]["joints_world"]), npz["joints_world"][0], atol=1e-5)


def test_frame_range_selection(stereo_capture, oracle_predictor, tmp_path):
    predictor, _ = _oracle(oracle_predictor, stereo_capture)
    report = run_sequence(
        rig=load_calib_json(stereo_capture["calib"]),
        video_paths=stereo_capture["videos"],
        out_dir=str(tmp_path / "range"),
        bbox_backend="json",
        bbox_json=stereo_capture["bbox_json"],
        predictor=predictor,
        video_backend="cv2",
        frame_start=2,
        frame_end=8,
        frame_step=2,
        progress=False,
    )
    payload = np.load(os.path.join(tmp_path / "range", "keypoints.npz"))
    np.testing.assert_array_equal(payload["frame_ids"], [2, 4, 6])
    assert report["frames_selected"] == 3


def test_overlay_video_is_written(stereo_capture, oracle_predictor, tmp_path):
    predictor, _ = _oracle(oracle_predictor, stereo_capture)
    out_dir = str(tmp_path / "overlay")
    run_sequence(rig=load_calib_json(stereo_capture["calib"]), video_paths=stereo_capture["videos"],
                 out_dir=out_dir, bbox_backend="json", bbox_json=stereo_capture["bbox_json"],
                 predictor=predictor, video_backend="cv2", overlay=True, progress=False)

    overlay_path = os.path.join(out_dir, "overlay.mp4")
    assert os.path.isfile(overlay_path) and os.path.getsize(overlay_path) > 0
    from tool.poemkit.video import CV2FrameReader

    reader = CV2FrameReader(overlay_path)
    try:
        assert len(reader) >= 1
        # two views tiled side by side at half scale
        assert reader.frame_size[0] == stereo_capture["image_size"][0]
    finally:
        reader.close()


def test_calibration_warnings_reach_the_report(stereo_capture, oracle_predictor, tmp_path):
    """A rig whose image_size disagrees with the video must be flagged, not silently used."""
    predictor, _ = _oracle(oracle_predictor, stereo_capture)
    rig = load_calib_json(stereo_capture["calib"])
    rig[1].image_size = (99, 99)

    report = run_sequence(rig=rig, video_paths=stereo_capture["videos"], out_dir=str(tmp_path / "warn"),
                          bbox_backend="json", bbox_json=stereo_capture["bbox_json"], predictor=predictor,
                          video_backend="cv2", progress=False)
    assert any("image_size" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------
def test_parse_video_args_forms(stereo_capture, tmp_path):
    rig_names = stereo_capture["cameras"]
    explicit = parse_video_args([f"left={stereo_capture['videos']['left']}",
                                 f"right={stereo_capture['videos']['right']}"], None, rig_names)
    assert list(explicit) == ["left", "right"]

    from_dir = parse_video_args([], stereo_capture["out_dir"], rig_names)
    assert list(from_dir) == ["left", "right"]

    bare = parse_video_args([stereo_capture["videos"]["left"]], None, rig_names)
    assert list(bare) == ["left"], "a bare path takes its camera name from the filename"


def test_parse_video_args_rejects_unknown_and_missing(stereo_capture):
    with pytest.raises(ValueError, match="not in the calibration"):
        parse_video_args([f"middle={stereo_capture['videos']['left']}"], None, stereo_capture["cameras"])
    with pytest.raises(FileNotFoundError):
        parse_video_args(["left=/nope/none.mp4"], None, stereo_capture["cameras"])
    with pytest.raises(ValueError, match="no videos resolved"):
        parse_video_args([], None, stereo_capture["cameras"])


def test_video_name_order_follows_the_calibration(stereo_capture):
    """Whatever order the flags come in, the master stays camera 0 of the rig."""
    reversed_args = [f"right={stereo_capture['videos']['right']}", f"left={stereo_capture['videos']['left']}"]
    assert list(parse_video_args(reversed_args, None, stereo_capture["cameras"])) == ["left", "right"]


def test_frame_indices_math():
    assert frame_indices(10, 0, None, 1, None) == list(range(10))
    assert frame_indices(10, 2, 6, 2, None) == [2, 4]
    assert frame_indices(10, 0, 100, 1, None) == list(range(10)), "end must clamp to the video length"
    assert frame_indices(10, 0, None, 1, 3) == [0, 1, 2]
    assert frame_indices(10, 0, None, 0, None) == list(range(10)), "step 0 must not hang"


def test_resolve_hand_side_from_json(stereo_capture, left_hand_capture):
    parser = build_parser()
    args = parser.parse_args(["--out", os.path.join("x", stereo_capture["sequence"]),
                              "--hand-side-json", os.path.join(stereo_capture["out_dir"], "hand_labels.json")])
    assert resolve_hand_side(args, stereo_capture["sequence"]) == "rh"

    args_lh = parser.parse_args(["--out", "y",
                                 "--hand-side-json", os.path.join(left_hand_capture["out_dir"], "hand_labels.json")])
    assert resolve_hand_side(args_lh, left_hand_capture["sequence"]) == "lh"

    with pytest.raises(KeyError):
        resolve_hand_side(args_lh, "no_such_sequence")


def test_cli_refuses_a_single_video(stereo_capture, capsys):
    """The mono guard at the CLI level: one video in, exit code 2, clear message."""
    from tool.infer_video import main

    code = main([
        "--calib", stereo_capture["calib"],
        "--video", f"left={stereo_capture['videos']['left']}",
        "--out", "/tmp/should_not_matter",
        "--dry-run",
    ])
    assert code == 2
    assert "mono video is not supported" in capsys.readouterr().err


def test_cli_dry_run_end_to_end(stereo_capture, tmp_path, capsys):
    from tool.infer_video import main

    code = main([
        "--calib", stereo_capture["calib"],
        "--video-dir", stereo_capture["out_dir"],
        "--bbox-backend", "npy",
        "--bbox-root", stereo_capture["bbox_root"],
        "--dry-run",
        "--out", str(tmp_path / "cli"),
    ])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frames_predicted"] == stereo_capture["num_frames"]


def test_cli_requires_a_checkpoint_for_a_real_run(stereo_capture, tmp_path, capsys):
    from tool.infer_video import main

    code = main([
        "--calib", stereo_capture["calib"],
        "--video-dir", stereo_capture["out_dir"],
        "--bbox-backend", "npy", "--bbox-root", stereo_capture["bbox_root"],
        "--out", str(tmp_path / "nockpt"),
    ])
    assert code == 2
    assert "--reload is required" in capsys.readouterr().err


def test_cli_env_report(capsys):
    from tool.infer_video import main

    assert main(["--env-report", "--out", "unused"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "modules" in report and "devices" in report
