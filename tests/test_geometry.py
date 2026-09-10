"""T2 - crop math, projection, triangulation, mirroring.

The crop bookkeeping is where a video-only pipeline quietly goes wrong: the
network sees a 256x256 crop, so its cameras must be the *crop's* cameras. Two
independent paths must agree:

    project(K_crop, X_cam)  ==  crop_affine( project(K_full, X_cam) )

These tests also assert parity with the repo's own implementations
(``lib.utils.transform``, ``lib.utils.triangulation``) whenever the full
environment is importable, so the laptop-testable copies cannot drift.
"""

import numpy as np
import pytest

from conftest import requires_model_env, requires_torch
from tool.poemkit.geometry import (
    affine_crop_transform,
    apply_affine_2d,
    bbox_get_center_scale,
    intrinsics_after_crop,
    mirror_points_x,
    project_points,
    reprojection_error,
    transf_points,
    triangulate_dlt_np,
)

K_FULL = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
OUT_RES = (256, 256)


def test_bbox_center_scale_expands_and_floors():
    center, scale = bbox_get_center_scale([100, 100, 140, 180], expand=2.0, mindim=0.0)
    np.testing.assert_allclose(center, [120.0, 140.0])
    assert scale == pytest.approx(160.0)  # max(40, 80) * 2

    _, small = bbox_get_center_scale([100, 100, 110, 110], expand=2.0, mindim=200.0)
    assert small == pytest.approx(200.0), "mindim must floor tiny detections"


def test_bbox_center_scale_rejects_short_input():
    with pytest.raises(ValueError):
        bbox_get_center_scale([1, 2, 3])


def test_crop_intrinsics_match_cropping_the_projection():
    """The core invariant: crop-K projection == affine of full-frame projection."""
    center, scale = bbox_get_center_scale([280, 200, 360, 300])
    affine = affine_crop_transform(center, scale, OUT_RES)
    K_crop = intrinsics_after_crop(K_FULL, center, scale, OUT_RES)

    rng = np.random.default_rng(0)
    pts_cam = np.stack([
        rng.uniform(-0.15, 0.15, 32),
        rng.uniform(-0.15, 0.15, 32),
        rng.uniform(0.35, 0.95, 32),
    ], axis=-1)

    uv_direct = project_points(K_crop, pts_cam)
    uv_via_affine = apply_affine_2d(affine, project_points(K_FULL, pts_cam))
    np.testing.assert_allclose(uv_direct, uv_via_affine, atol=1e-9)


def test_crop_maps_window_onto_the_full_output():
    """The crop window's corners land on the crop's corners, its center in the middle."""
    center, scale = np.array([320.0, 240.0]), 200.0
    affine = affine_crop_transform(center, scale, OUT_RES)
    corners = np.array([
        [center[0] - scale / 2, center[1] - scale / 2],
        [center[0] + scale / 2, center[1] + scale / 2],
        center,
    ])
    mapped = apply_affine_2d(affine, corners)
    np.testing.assert_allclose(mapped[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(mapped[1], [256.0, 256.0], atol=1e-9)
    np.testing.assert_allclose(mapped[2], [128.0, 128.0], atol=1e-9)


def test_non_square_output_keeps_aspect_convention():
    """Reproduces the repo's ``scale_ratio`` behaviour for a non-square output."""
    affine = affine_crop_transform([100.0, 100.0], 200.0, (128, 64))
    assert affine[0, 0] == pytest.approx(128 / 200)
    assert affine[1, 1] == pytest.approx(64 / 200 * (128 / 64))


def test_projection_and_transform_roundtrip():
    T_cw = np.array([
        [0.0, 0.0, 1.0, -0.3],
        [0.0, 1.0, 0.0, 0.05],
        [-1.0, 0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ])
    pts_world = np.array([[0.1, 0.0, 0.5], [-0.05, 0.02, 0.62]])
    pts_cam = transf_points(T_cw, pts_world)
    back = transf_points(np.linalg.inv(T_cw), pts_cam)
    np.testing.assert_allclose(back, pts_world, atol=1e-12)

    uv = project_points(K_FULL, pts_cam)
    # a projected point re-lifted along its ray at the known depth returns itself
    rays = np.stack([(uv[:, 0] - 320.0) / 600.0, (uv[:, 1] - 240.0) / 600.0, np.ones(len(uv))], axis=-1)
    np.testing.assert_allclose(rays * pts_cam[:, 2:], pts_cam, atol=1e-9)


def test_projection_survives_zero_depth():
    uv = project_points(K_FULL, np.array([[0.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(uv)), "z==0 must be clamped, not produce inf/nan"


def test_triangulation_recovers_points_from_two_views():
    from tool.poemkit.calib import rig_from_rectified_stereo

    rig = rig_from_rectified_stereo(fx=600, fy=600, cx=320, cy=240, baseline_m=0.15)
    rng = np.random.default_rng(3)
    pts = np.stack([rng.uniform(-0.1, 0.1, 21), rng.uniform(-0.1, 0.1, 21), rng.uniform(0.4, 0.8, 21)], axis=-1)

    Ks = np.stack([c.K for c in rig])
    Ts = np.stack([c.T_cw for c in rig])
    uv = np.stack([project_points(Ks[i], transf_points(Ts[i], pts)) for i in range(2)], axis=0)

    recovered = triangulate_dlt_np(uv, Ks, Ts)
    np.testing.assert_allclose(recovered, pts, atol=1e-9)


def test_triangulation_needs_two_views():
    with pytest.raises(ValueError, match=">= 2 views"):
        triangulate_dlt_np(np.zeros((1, 21, 2)), np.eye(3)[None], np.eye(4)[None])


def test_triangulation_degrades_gracefully_with_noise():
    """A 12 cm stereo baseline at ~0.6 m: 1 px of 2D noise stays sub-centimeter."""
    from tool.poemkit.calib import rig_from_rectified_stereo

    rig = rig_from_rectified_stereo(fx=600, fy=600, cx=320, cy=240, baseline_m=0.12)
    Ks = np.stack([c.K for c in rig])
    Ts = np.stack([c.T_cw for c in rig])
    rng = np.random.default_rng(7)
    pts = np.stack([rng.uniform(-0.08, 0.08, 21), rng.uniform(-0.08, 0.08, 21), rng.uniform(0.5, 0.7, 21)], axis=-1)
    uv = np.stack([project_points(Ks[i], transf_points(Ts[i], pts)) for i in range(2)], axis=0)
    uv_noisy = uv + rng.normal(0.0, 1.0, uv.shape)

    err = np.linalg.norm(triangulate_dlt_np(uv_noisy, Ks, Ts) - pts, axis=-1)
    assert err.mean() < 0.01, f"mean error {err.mean():.4f} m is worse than the expected ~mm-cm regime"


def test_reprojection_error_on_synthetic_capture(stereo_capture, stereo_gt):
    """GT 3D reprojects onto GT 2D to within float noise, in the capture's own calibration."""
    from tool.poemkit.calib import load_calib_json

    rig = load_calib_json(stereo_capture["calib"])
    Ks = np.stack([c.K for c in rig])
    Ts = np.stack([c.T_cw for c in rig])
    frame = 2
    errs = reprojection_error(stereo_gt["joints_world"][frame], stereo_gt["joints_2d"][:, frame], Ks, Ts)
    assert errs.max() < 1e-3


def test_mirror_points_x_is_an_involution():
    pts = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.5]])
    np.testing.assert_allclose(mirror_points_x(mirror_points_x(pts)), pts)
    np.testing.assert_allclose(mirror_points_x(pts)[:, 0], -pts[:, 0])
    assert mirror_points_x(pts) is not pts, "must not modify the caller's array"


@requires_model_env
def test_parity_with_repo_affine_helpers():
    """Our numpy crop math must equal ``lib.utils.transform``'s (rot=0)."""
    from lib.utils.transform import _affine_transform, _affine_transform_post_rot

    center, scale = np.array([311.0, 207.0]), 173.0
    optical_center = np.array([320.0, 240.0])

    np.testing.assert_allclose(affine_crop_transform(center, scale, OUT_RES),
                               _affine_transform(center=center, scale=scale, out_res=OUT_RES, rot=0),
                               atol=1e-5)
    expected_K = _affine_transform_post_rot(center=center, scale=scale, optical_center=optical_center,
                                            out_res=OUT_RES, rot=0).dot(K_FULL)
    np.testing.assert_allclose(intrinsics_after_crop(K_FULL, center, scale, OUT_RES, optical_center),
                               expected_K, atol=1e-4)


@requires_torch
def test_parity_with_repo_triangulation():
    """Our numpy DLT must match ``batch_triangulate_dlt_torch`` on the same input."""
    import torch

    from lib.utils.triangulation import batch_triangulate_dlt_torch
    from tool.poemkit.calib import rig_from_rectified_stereo

    rig = rig_from_rectified_stereo(fx=620, fy=620, cx=320, cy=240, baseline_m=0.2)
    Ks = np.stack([c.K for c in rig])
    Ts = np.stack([c.T_cw for c in rig])
    rng = np.random.default_rng(11)
    pts = np.stack([rng.uniform(-0.1, 0.1, 21), rng.uniform(-0.1, 0.1, 21), rng.uniform(0.4, 0.9, 21)], axis=-1)
    uv = np.stack([project_points(Ks[i], transf_points(Ts[i], pts)) for i in range(2)], axis=0)

    ours = triangulate_dlt_np(uv, Ks, Ts)
    theirs = batch_triangulate_dlt_torch(
        torch.as_tensor(uv, dtype=torch.float64)[None],
        torch.as_tensor(Ks, dtype=torch.float64)[None],
        torch.as_tensor(Ts, dtype=torch.float64)[None],
    )[0].numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-6)
    np.testing.assert_allclose(ours, pts, atol=1e-6)
