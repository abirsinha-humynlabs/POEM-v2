"""T4 - view packing: crops, master frame, the >=2-view contract, left-hand mirroring.

This is the layer that decides what the network actually sees. The properties
under test are the ones that, if broken, still produce numbers:

* the master view's camera->master transform must be exactly identity, since
  predictions are returned in that frame;
* the crop intrinsics must agree with the crop actually rendered;
* fewer than two views must be refused (single-view forward reads ground
  truth in ``lib/models/POEM.py``);
* the left-hand path must round-trip: mirror in, un-mirror out.
"""

import numpy as np
import pytest

from tool.poemkit.bbox import NpyBBoxProvider
from tool.poemkit.calib import load_calib_json
from tool.poemkit.geometry import apply_affine_2d, affine_crop_transform, project_points, transf_points
from tool.poemkit.views import PackError, joints_uv_to_original, prepare_views, project_to_views


def _load(manifest, frame_id=0, hand_side="rh", **kwargs):
    from tool.poemkit.video import MultiViewReader

    rig = load_calib_json(manifest["calib"])
    provider = NpyBBoxProvider(manifest["bbox_root"])
    with MultiViewReader({n: manifest["videos"][n] for n in rig.names}, backend="cv2") as reader:
        frames = reader.read(frame_id)
    boxes = {name: provider.get(name, frame_id) for name in rig.names}
    packet = prepare_views(frames=frames, bboxes=boxes, rig=rig, hand_side=hand_side, frame_id=frame_id, **kwargs)
    return rig, frames, boxes, packet


def test_packet_shapes_and_master_identity(stereo_capture):
    rig, frames, _, packet = _load(stereo_capture)

    assert packet.num_views == 2
    assert packet.names[0] == rig.names[0], "view 0 must be the master"
    assert packet.crops.shape == (2, 256, 256, 3)
    assert packet.crops.dtype == np.uint8
    assert packet.K_crop.shape == (2, 3, 3)
    assert packet.T_mc.shape == (2, 4, 4)
    np.testing.assert_allclose(packet.T_mc[0], np.eye(4), atol=1e-9)
    assert not packet.flipped


def test_camera_to_master_transforms_are_consistent(stereo_capture):
    """``T_mc[i]`` must carry a point from camera i's frame into the master's."""
    rig, _, _, packet = _load(stereo_capture)
    point_world = np.array([[0.02, -0.01, 0.55]])

    master_expected = transf_points(rig[packet.names[0]].T_cw, point_world)
    for i, name in enumerate(packet.names):
        in_cam_i = transf_points(rig[name].T_cw, point_world)
        np.testing.assert_allclose(transf_points(packet.T_mc[i], in_cam_i), master_expected, atol=1e-9)


def test_crop_intrinsics_match_the_rendered_crop(stereo_capture, stereo_gt):
    """Project GT joints with ``K_crop`` and with "affine of the full projection"; they must agree
    and land inside the crop."""
    rig, _, _, packet = _load(stereo_capture, frame_id=1)
    gt_world = stereo_gt["joints_world"][1]

    for i, name in enumerate(packet.names):
        cam = rig[name]
        pts_cam = transf_points(cam.T_cw, gt_world)
        uv_crop = project_points(packet.K_crop[i], pts_cam)
        affine = affine_crop_transform(packet.centers[i], packet.scales[i], (256, 256))
        uv_expected = apply_affine_2d(affine, project_points(cam.K, pts_cam))
        np.testing.assert_allclose(uv_crop, uv_expected, atol=1e-8)
        # the crop is a 2x expansion of the hand box, so the hand sits well inside
        assert uv_crop.min() > 0 and uv_crop.max() < 256


def test_crop_pixels_are_not_blank(stereo_capture):
    """A silently mis-scaled affine yields an all-zero crop; check there is signal."""
    _, _, _, packet = _load(stereo_capture)
    for crop in packet.crops:
        assert crop.std() > 5.0, "crop looks blank; the affine or the source frame is wrong"


def test_single_view_is_refused(stereo_capture):
    """The core mono restriction, enforced at the packing layer."""
    rig = load_calib_json(stereo_capture["calib"])
    provider = NpyBBoxProvider(stereo_capture["bbox_root"])
    from tool.poemkit.video import MultiViewReader

    with MultiViewReader({n: stereo_capture["videos"][n] for n in rig.names}, backend="cv2") as reader:
        frames = reader.read(0)
    boxes = {rig.names[0]: provider.get(rig.names[0], 0), rig.names[1]: None}

    with pytest.raises(PackError, match="only 1 usable view"):
        prepare_views(frames=frames, bboxes=boxes, rig=rig, frame_id=0)


def test_frames_with_two_of_three_views_still_pack(ring_capture):
    """Occlusion robustness: cam1 has no box on frame 2, the other two carry it."""
    _, _, _, packet = _load(ring_capture, frame_id=2)
    assert packet.num_views == 2
    assert "cam1" not in packet.names
    assert packet.dropped.get("cam1") == "no bbox"


def test_frame_with_one_view_left_is_refused(ring_capture):
    """Frame 3 of the ring fixture loses two of three views."""
    with pytest.raises(PackError):
        _load(ring_capture, frame_id=3)


def test_malformed_boxes_are_dropped_with_a_reason(stereo_capture):
    rig = load_calib_json(stereo_capture["calib"])
    from tool.poemkit.video import MultiViewReader

    with MultiViewReader({n: stereo_capture["videos"][n] for n in rig.names}, backend="cv2") as reader:
        frames = reader.read(0)

    cases = {
        "degenerate bbox": np.array([100.0, 100.0, 100.0, 100.0]),
        "malformed bbox": np.array([np.nan, 0.0, 10.0, 10.0]),
    }
    for expected_reason, bad_box in cases.items():
        boxes = {rig.names[0]: bad_box, rig.names[1]: np.array([50.0, 50.0, 150.0, 150.0])}
        with pytest.raises(PackError) as excinfo:
            prepare_views(frames=frames, bboxes=boxes, rig=rig, frame_id=0)
        assert expected_reason in str(excinfo.value)


def test_master_override_reorders_the_packet(ring_capture):
    _, _, _, packet = _load(ring_capture, frame_id=0, master="cam2")
    assert packet.names[0] == "cam2"
    np.testing.assert_allclose(packet.T_mc[0], np.eye(4), atol=1e-9)

    rig = load_calib_json(ring_capture["calib"])
    with pytest.raises(ValueError, match="not in rig"):
        prepare_views(frames={n: np.zeros((10, 10, 3), np.uint8) for n in rig.names},
                      bboxes={n: np.array([0.0, 0, 5, 5]) for n in rig.names},
                      rig=rig, master="nope")


def test_master_to_world_roundtrip(stereo_capture, stereo_gt):
    """World -> master -> world must be the identity for the right-hand path."""
    rig, _, _, packet = _load(stereo_capture, frame_id=1)
    gt_world = stereo_gt["joints_world"][1]
    in_master = transf_points(rig[packet.names[0]].T_cw, gt_world)
    np.testing.assert_allclose(packet.master_to_world(in_master), gt_world, atol=1e-9)


def test_left_hand_path_mirrors_and_unmirrors(left_hand_capture):
    """For a left hand the frames and extrinsics are mirrored; the output must come
    back in the original (un-mirrored) world frame."""
    from tool.poemkit.geometry import mirror_points_x

    rig, frames, _, packet = _load(left_hand_capture, frame_id=1, hand_side="lh")
    gt = np.load(left_hand_capture["gt"])
    gt_world = gt["joints_world"][1]

    assert packet.flipped
    # the model sees the mirrored world; feed it that and expect the real world back
    world_for_model = mirror_points_x(gt_world)
    in_master = transf_points(packet.T_cw_used[0], world_for_model)
    np.testing.assert_allclose(packet.master_to_world(in_master), gt_world, atol=1e-9)


def test_left_hand_mirrored_projection_lands_on_the_flipped_pixels(left_hand_capture):
    """The mirrored extrinsics + original K must project the mirrored hand onto the
    horizontally flipped image coordinates."""
    from tool.poemkit.geometry import mirror_points_x

    rig, _, _, packet = _load(left_hand_capture, frame_id=1, hand_side="lh")
    gt = np.load(left_hand_capture["gt"])
    world_for_model = mirror_points_x(gt["joints_world"][1])

    for i, name in enumerate(packet.names):
        cam = rig[name]
        cam_idx = rig.names.index(name)
        uv_original = gt["joints_2d"][cam_idx, 1]
        uv_mirrored = project_points(cam.K, transf_points(packet.T_cw_used[i], world_for_model))
        expected_u = 2.0 * cam.K[0, 2] - uv_original[:, 0]
        np.testing.assert_allclose(uv_mirrored[:, 0], expected_u, atol=1e-6)
        np.testing.assert_allclose(uv_mirrored[:, 1], uv_original[:, 1], atol=1e-6)


def test_joints_uv_to_original_inverts_the_crop(stereo_capture, stereo_gt):
    rig, _, _, packet = _load(stereo_capture, frame_id=2)
    gt_world = stereo_gt["joints_world"][2]
    uv_crop = np.stack([
        project_points(packet.K_crop[i], transf_points(rig[name].T_cw, gt_world))
        for i, name in enumerate(packet.names)
    ], axis=0)

    back = joints_uv_to_original(packet, uv_crop, rig)
    for name in packet.names:
        expected = project_points(rig[name].K, transf_points(rig[name].T_cw, gt_world))
        np.testing.assert_allclose(back[name], expected, atol=1e-6)


def test_joints_uv_to_original_handles_the_left_hand(left_hand_capture):
    """After un-cropping, a left-hand run's 2D keypoints must be back in the
    *unflipped* frame, i.e. on top of the real pixels."""
    from tool.poemkit.geometry import mirror_points_x

    rig, _, _, packet = _load(left_hand_capture, frame_id=1, hand_side="lh")
    gt = np.load(left_hand_capture["gt"])
    world_for_model = mirror_points_x(gt["joints_world"][1])
    uv_crop = np.stack([
        project_points(packet.K_crop[i], transf_points(packet.T_cw_used[i], world_for_model))
        for i in range(packet.num_views)
    ], axis=0)

    back = joints_uv_to_original(packet, uv_crop, rig)
    for name in packet.names:
        np.testing.assert_allclose(back[name], gt["joints_2d"][rig.names.index(name), 1], atol=1e-5)


def test_joints_uv_view_count_is_validated(stereo_capture):
    rig, _, _, packet = _load(stereo_capture)
    with pytest.raises(ValueError, match="views"):
        joints_uv_to_original(packet, np.zeros((5, 21, 2)), rig)


def test_project_to_views_covers_every_camera(stereo_capture, stereo_gt):
    rig = load_calib_json(stereo_capture["calib"])
    proj = project_to_views(stereo_gt["joints_world"][0], rig)
    assert set(proj) == set(rig.names)
    for ci, name in enumerate(rig.names):
        np.testing.assert_allclose(proj[name], stereo_gt["joints_2d"][ci, 0], atol=1e-3)


def test_invalid_hand_side_is_rejected(stereo_capture):
    rig = load_calib_json(stereo_capture["calib"])
    with pytest.raises(ValueError, match="hand_side"):
        prepare_views(frames={n: np.zeros((10, 10, 3), np.uint8) for n in rig.names},
                      bboxes={n: np.array([0.0, 0, 5, 5]) for n in rig.names},
                      rig=rig, hand_side="both")
