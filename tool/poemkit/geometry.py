"""Numpy geometry helpers: crop math, projection, mirroring, triangulation.

These are deliberately duplicated from ``lib/utils/transform.py`` /
``lib/utils/triangulation.py`` in *numpy only* form, because those modules
import ``pytorch3d`` at module scope and therefore cannot be imported without
a CUDA-flavoured environment. ``tests/test_geometry.py`` asserts numerical
parity with the repo originals whenever the full environment *is* available,
so the two can never silently drift.

Only ``rot=0`` is implemented: inference does no rotation augmentation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "bbox_get_center_scale",
    "affine_trans_no_rot",
    "affine_crop_transform",
    "intrinsics_after_crop",
    "apply_affine_2d",
    "transf_points",
    "project_points",
    "mirror_points_x",
    "triangulate_dlt_np",
    "reprojection_error",
]


def bbox_get_center_scale(bbox: Sequence[float], expand: float = 2.0, mindim: float = 200.0):
    """Square crop window around a bbox. Mirrors ``tool/infer_hand.py``.

    Args:
        bbox: ``(x_min, y_min, x_max, y_max)`` in pixels.
        expand: side-length multiplier (2.0 in the released demo).
        mindim: floor on the crop side length, in pixels.

    Returns:
        ``(center_xy, scale)`` where ``scale`` is the square side length.
    """
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if bbox.shape[0] < 4:
        raise ValueError(f"bbox must have 4 numbers, got {bbox.shape[0]}")
    w, h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    scale = max(w, h) * float(expand)
    scale = max(scale, float(mindim))
    center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float64)
    return center, scale


def affine_trans_no_rot(center: Sequence[float], scale: float, out_res: Sequence[int]) -> np.ndarray:
    """3x3 image->crop affine, identical to ``lib.utils.transform._get_affine_trans_no_rot``."""
    res_w, res_h = float(out_res[0]), float(out_res[1])
    scale_ratio = res_w / res_h
    affine = np.zeros((3, 3), dtype=np.float64)
    affine[0, 0] = res_w / scale
    affine[1, 1] = res_h / scale * scale_ratio
    affine[0, 2] = res_w * (-float(center[0]) / scale + 0.5)
    affine[1, 2] = res_h * (-float(center[1]) / scale * scale_ratio + 0.5)
    affine[2, 2] = 1.0
    return affine


def affine_crop_transform(center: Sequence[float], scale: float, out_res: Sequence[int]) -> np.ndarray:
    """Image->crop transform used to warp the frame (``_affine_transform`` with rot=0)."""
    return affine_trans_no_rot(center, scale, out_res)


def intrinsics_after_crop(
    K: np.ndarray,
    center: Sequence[float],
    scale: float,
    out_res: Sequence[int],
    optical_center: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Intrinsics of the cropped image.

    Equivalent to ``_affine_transform_post_rot(...) @ K`` in
    ``tool/infer_hand.py``. With rot=0 the optical center plays no role, but it
    is accepted (and ignored) to keep the call sites symmetric with the repo.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    del optical_center  # only matters when rot != 0
    return affine_trans_no_rot(center, scale, out_res) @ K


def apply_affine_2d(affine: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine to ``(..., N, 2)`` pixel coordinates."""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    hom = np.concatenate([points_2d, np.ones_like(points_2d[..., :1])], axis=-1)  # (..., N, 3)
    out = hom @ np.asarray(affine, dtype=np.float64).T
    return out[..., :2] / np.clip(out[..., 2:], 1e-9, None)


def transf_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) to ``(N, 3)`` points."""
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def project_points(K: np.ndarray, points_cam: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Pinhole-project camera-frame ``(N, 3)`` points to ``(N, 2)`` pixels."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    hom = points_cam @ np.asarray(K, dtype=np.float64).T
    z = hom[..., 2:]
    z = np.where(np.abs(z) < eps, eps, z)
    return hom[..., :2] / z


def mirror_points_x(points: np.ndarray) -> np.ndarray:
    """Negate x. Used by the left-hand (world-mirroring) path."""
    out = np.asarray(points, dtype=np.float64).copy()
    out[..., 0] *= -1.0
    return out


def triangulate_dlt_np(points_2d: np.ndarray, Ks: np.ndarray, T_cws: np.ndarray) -> np.ndarray:
    """Linear DLT triangulation; numpy twin of ``batch_triangulate_dlt_torch``.

    Args:
        points_2d: ``(N_views, J, 2)`` pixel observations.
        Ks: ``(N_views, 3, 3)`` intrinsics.
        T_cws: ``(N_views, 4, 4)`` world->camera transforms.

    Returns:
        ``(J, 3)`` points in the world frame of ``T_cws``.
    """
    points_2d = np.asarray(points_2d, dtype=np.float64)
    Ks = np.asarray(Ks, dtype=np.float64)
    T_cws = np.asarray(T_cws, dtype=np.float64)
    if points_2d.ndim != 3 or points_2d.shape[-1] != 2:
        raise ValueError(f"points_2d must be (N, J, 2), got {points_2d.shape}")
    n_views, n_joints = points_2d.shape[0], points_2d.shape[1]
    if n_views < 2:
        raise ValueError(f"DLT triangulation needs >= 2 views, got {n_views}")

    M = Ks @ T_cws[:, :3, :]  # (N, 3, 4)
    out = np.zeros((n_joints, 3), dtype=np.float64)
    for j in range(n_joints):
        rows = []
        for n in range(n_views):
            u, v = points_2d[n, j]
            rows.append(u * M[n, 2, :] - M[n, 0, :])
            rows.append(v * M[n, 2, :] - M[n, 1, :])
        _, _, vt = np.linalg.svd(np.stack(rows, axis=0))
        w = vt[-1]
        out[j] = w[:3] / (w[3] if abs(w[3]) > 1e-12 else 1e-12)
    return out


def reprojection_error(
    points_3d_world: np.ndarray,
    points_2d: np.ndarray,
    Ks: np.ndarray,
    T_cws: np.ndarray,
) -> np.ndarray:
    """Per-view, per-point pixel reprojection error ``(N_views, J)``.

    This is the cheapest end-to-end health check for a video-only run: if the
    calibration or the crop bookkeeping is wrong, this blows up even though
    the network still returns plausible-looking joints.
    """
    points_3d_world = np.asarray(points_3d_world, dtype=np.float64)
    points_2d = np.asarray(points_2d, dtype=np.float64)
    errs = []
    for n in range(points_2d.shape[0]):
        cam_pts = transf_points(T_cws[n], points_3d_world)
        proj = project_points(Ks[n], cam_pts)
        errs.append(np.linalg.norm(proj - points_2d[n], axis=-1))
    return np.stack(errs, axis=0)
