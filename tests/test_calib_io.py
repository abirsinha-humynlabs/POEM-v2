"""T1 - calibration IO: the numbers POEM-v2 cannot work without.

Wrong extrinsics are the single most likely reason a video-only run produces
plausible-looking but wrong metric 3D, and nothing downstream complains. These
tests pin the formats, the inversion conventions and the sanity checks.
"""

import json
import os

import numpy as np
import pytest

from tool.make_calib import look_at_extrinsic, make_ring_rig
from tool.poemkit.calib import (
    Camera,
    CameraRig,
    inv_se3,
    load_calib_json,
    load_poem_pkl_calib,
    rig_from_rectified_stereo,
    save_calib_json,
)


def test_inv_se3_matches_numpy_inverse():
    T = look_at_extrinsic([0.3, -0.2, -0.5], [0.0, 0.0, 0.4])
    np.testing.assert_allclose(inv_se3(T), np.linalg.inv(T), atol=1e-12)
    np.testing.assert_allclose(inv_se3(inv_se3(T)), T, atol=1e-12)


def test_rectified_stereo_baseline_and_master_frame():
    rig = rig_from_rectified_stereo(fx=700, fy=700, cx=640, cy=360, baseline_m=0.12, image_size=(1280, 720))
    assert rig.names == ["left", "right"]
    # the master camera defines the world frame
    np.testing.assert_allclose(rig[0].T_cw, np.eye(4), atol=1e-12)
    # the right camera sits +baseline along world x
    np.testing.assert_allclose(rig[1].center, [0.12, 0.0, 0.0], atol=1e-12)
    assert rig.baselines()[("left", "right")] == pytest.approx(0.12)
    assert rig.sanity_check() == []


def test_rectified_stereo_projection_geometry():
    """A point on the world z axis lands right of center in the left eye and
    left of center in the right eye -- i.e. positive disparity."""
    from tool.poemkit.geometry import project_points, transf_points

    rig = rig_from_rectified_stereo(fx=600, fy=600, cx=320, cy=240, baseline_m=0.1, image_size=(640, 480))
    point_world = np.array([[0.0, 0.0, 0.6]])
    uv_left = project_points(rig[0].K, transf_points(rig[0].T_cw, point_world))[0]
    uv_right = project_points(rig[1].K, transf_points(rig[1].T_cw, point_world))[0]

    np.testing.assert_allclose(uv_left, [320.0, 240.0], atol=1e-9)
    assert uv_right[0] < uv_left[0], "right eye must see the point further left (positive disparity)"
    assert uv_right[0] == pytest.approx(320.0 - 600 * 0.1 / 0.6)
    assert uv_right[1] == pytest.approx(240.0), "a rectified pair has no vertical disparity"


def test_json_roundtrip(tmp_path):
    rig = make_ring_rig(num_cams=3, fx=650.0, image_size=(1024, 768))
    path = save_calib_json(rig, str(tmp_path / "calib.json"))
    loaded = load_calib_json(path)

    assert loaded.names == rig.names
    for a, b in zip(rig, loaded):
        np.testing.assert_allclose(a.K, b.K, atol=1e-12)
        np.testing.assert_allclose(a.T_cw, b.T_cw, atol=1e-12)
        assert a.image_size == b.image_size


def test_poem_pkl_layout(tmp_path):
    """The released example data's pickle layout still loads (back-compat)."""
    import pickle

    rig = make_ring_rig(num_cams=2, names=["camera_1", "camera_2"])
    for sub in ("cam_intr", "cam_extr"):
        os.makedirs(tmp_path / sub)
    for cam in rig:
        with open(tmp_path / "cam_intr" / f"{cam.name}.pkl", "wb") as ofs:
            pickle.dump(cam.K, ofs)
        with open(tmp_path / "cam_extr" / f"{cam.name}.pkl", "wb") as ofs:
            pickle.dump(cam.T_cw, ofs)

    loaded = load_poem_pkl_calib(str(tmp_path), camera_names=["camera_1", "camera_2"])
    assert loaded.names == ["camera_1", "camera_2"]
    np.testing.assert_allclose(loaded["camera_2"].T_cw, rig["camera_2"].T_cw, atol=1e-12)


def test_ring_rig_is_master_referenced():
    rig = make_ring_rig(num_cams=4, arc_deg=100.0)
    np.testing.assert_allclose(rig[0].T_cw, np.eye(4), atol=1e-9)
    assert len(rig.baselines()) == 6
    assert rig.sanity_check() == []


def test_look_at_puts_target_on_the_optical_axis():
    from tool.poemkit.geometry import project_points, transf_points

    target = np.array([0.1, 0.0, 0.5])
    T_cw = look_at_extrinsic([0.5, -0.1, 0.0], target)
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    uv = project_points(K, transf_points(T_cw, target[None]))[0]
    np.testing.assert_allclose(uv, [320.0, 240.0], atol=1e-6)


def test_single_view_rig_is_flagged():
    rig = CameraRig([Camera("only", np.eye(3), np.eye(4))])
    problems = rig.sanity_check()
    assert any("needs >= 2 calibrated views" in p for p in problems)
    with pytest.raises(ValueError):
        rig.sanity_check(strict=True)


def test_millimeter_extrinsics_are_flagged():
    """A rig calibrated in millimeters is the classic silent-failure case."""
    rig = rig_from_rectified_stereo(fx=700, fy=700, cx=640, cy=360, baseline_m=120.0)  # mm masquerading as m
    assert any("non-metric" in p for p in rig.sanity_check())


def test_colocated_views_are_flagged():
    rig = rig_from_rectified_stereo(fx=700, fy=700, cx=640, cy=360, baseline_m=1e-6)
    assert any("co-located" in p for p in rig.sanity_check())


def test_non_orthonormal_rotation_is_flagged():
    bad = np.eye(4)
    bad[:3, :3] = np.diag([1.0, 1.0, 2.0])
    rig = CameraRig([
        Camera("a", np.eye(3), np.eye(4)),
        Camera("b", np.eye(3), bad),
    ])
    problems = rig.sanity_check()
    assert any("orthonormal" in p for p in problems)


def test_principal_point_outside_image_is_flagged():
    rig = CameraRig([
        Camera("a", np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1.0]]), np.eye(4), image_size=(1280, 720)),
        Camera("b", np.array([[700.0, 0, 5000], [0, 700.0, 360], [0, 0, 1.0]]), np.eye(4), image_size=(1280, 720)),
    ])
    assert any("outside image" in p for p in rig.sanity_check())


def test_duplicate_camera_names_rejected():
    cam = Camera("dup", np.eye(3), np.eye(4))
    with pytest.raises(ValueError, match="duplicate"):
        CameraRig([cam, Camera("dup", np.eye(3), np.eye(4))])


def test_malformed_matrices_rejected():
    with pytest.raises(ValueError):
        Camera("bad_K", np.eye(4), np.eye(4))
    with pytest.raises(ValueError):
        Camera("bad_T", np.eye(3), np.eye(3))


def test_generated_capture_calibration_is_clean(stereo_capture, ring_capture):
    for manifest in (stereo_capture, ring_capture):
        rig = load_calib_json(manifest["calib"])
        assert len(rig) >= 2
        assert rig.sanity_check() == []
        with open(manifest["calib"]) as ifs:
            assert "cameras" in json.load(ifs)
