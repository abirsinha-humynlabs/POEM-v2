"""Calibration IO for video-only POEM-v2 inference.

POEM-v2 needs, per camera:
  * ``K``     3x3 pinhole intrinsics of the *video frames* being fed in
  * ``T_cw``  4x4 SE(3) world->camera transform, so that ``P_c = T_cw @ P_w``

Everything else in this repo is derived from those two. This module reads
them from a few common sources and always hands back the same in-memory
representation (:class:`CameraRig`), so the inference script does not care
where the numbers came from.

Supported sources
-----------------
1. ``calib.json``  -- the native format of this toolkit (see SCHEMA below).
2. The POEM-v2 example-data layout: a directory with ``cam_intr/<name>.pkl``
   and ``cam_extr/<name>.pkl`` (as used by ``tool/infer_hand.py``).
3. An OpenCV ``stereoCalibrate`` result stored as a FileStorage ``.yml``/``.xml``
   (keys ``K1 D1 K2 D2 R T``, or the rectified ``P1 P2`` variant).
4. Plain stereo parameters (fx, fy, cx, cy, baseline) for an already
   *rectified* stereo pair -- the common case for a stereo camera whose SDK
   only reports a baseline.

SCHEMA (calib.json)
-------------------
{
  "cameras": [
    {"name": "cam0",
     "image_size": [1280, 720],          # [w, h] of the video frames
     "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
     "T_cw": [[...4x4...]],              # world->camera; cam0 is usually identity
     "dist": [k1,k2,p1,p2,k3]            # optional; only used by undistort tooling
    }, ...
  ]
}

Conventions / gotchas
---------------------
* The **first** camera in the list is the *master*: POEM-v2 returns 3D in the
  master camera frame, and the toolkit re-expresses it in the world frame of
  whatever extrinsics you supplied.
* Translation units are **meters**. The model's ``POSITION_RANGE`` covers
  x,y in [-0.6, 0.6] and z in [0, 1.2] around the master camera, so a rig
  calibrated in millimeters will silently produce garbage. :meth:`sanity_check`
  flags the obvious cases.
* If your frames are rectified/undistorted, ``dist`` must be zeros and ``K``
  must be the *rectified* intrinsics (i.e. from ``P1``/``P2``), not the raw ones.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Camera",
    "CameraRig",
    "load_calib_json",
    "save_calib_json",
    "load_poem_pkl_calib",
    "load_opencv_stereo_yml",
    "rig_from_rectified_stereo",
]


def _as_mat(value, shape, name) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {mat.shape}")
    return mat


def inv_se3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) matrix without a general matrix inverse."""
    T = _as_mat(T, (4, 4), "T")
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


class Camera:
    """One physical view: intrinsics + world->camera extrinsics."""

    def __init__(
        self,
        name: str,
        K: np.ndarray,
        T_cw: np.ndarray,
        image_size: Optional[Sequence[int]] = None,
        dist: Optional[Sequence[float]] = None,
    ):
        self.name = str(name)
        self.K = _as_mat(K, (3, 3), f"{name}.K")
        self.T_cw = _as_mat(T_cw, (4, 4), f"{name}.T_cw")
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self.dist = None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)

    # -- derived quantities -------------------------------------------------
    @property
    def T_wc(self) -> np.ndarray:
        """camera->world transform (the inverse of :attr:`T_cw`)."""
        return inv_se3(self.T_cw)

    @property
    def center(self) -> np.ndarray:
        """Camera optical center in world coordinates."""
        return self.T_wc[:3, 3]

    @property
    def optical_center(self) -> np.ndarray:
        """Principal point (cx, cy) in pixels."""
        return np.array([self.K[0, 2], self.K[1, 2]], dtype=np.float64)

    def to_dict(self) -> Dict:
        out = {
            "name": self.name,
            "K": self.K.tolist(),
            "T_cw": self.T_cw.tolist(),
        }
        if self.image_size is not None:
            out["image_size"] = list(self.image_size)
        if self.dist is not None:
            out["dist"] = self.dist.tolist()
        return out

    @classmethod
    def from_dict(cls, payload: Dict) -> "Camera":
        return cls(
            name=payload["name"],
            K=payload["K"],
            T_cw=payload["T_cw"],
            image_size=payload.get("image_size"),
            dist=payload.get("dist"),
        )

    def __repr__(self) -> str:
        return f"Camera(name={self.name!r}, fx={self.K[0, 0]:.1f}, center={np.round(self.center, 3).tolist()})"


class CameraRig:
    """An ordered set of cameras. ``rig[0]`` is the master view."""

    def __init__(self, cameras: Sequence[Camera]):
        self.cameras: List[Camera] = list(cameras)
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate camera names in rig: {names}")

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self):
        return iter(self.cameras)

    def __getitem__(self, key):
        if isinstance(key, str):
            for cam in self.cameras:
                if cam.name == key:
                    return cam
            raise KeyError(f"no camera named {key!r}; have {self.names}")
        return self.cameras[key]

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.cameras]

    @property
    def intr_map(self) -> Dict[str, np.ndarray]:
        return {c.name: c.K.astype(np.float32) for c in self.cameras}

    @property
    def extr_map(self) -> Dict[str, np.ndarray]:
        return {c.name: c.T_cw.astype(np.float32) for c in self.cameras}

    def subset(self, names: Sequence[str]) -> "CameraRig":
        return CameraRig([self[n] for n in names])

    def baselines(self) -> Dict[Tuple[str, str], float]:
        """Pairwise distances (meters) between camera centers."""
        out = {}
        for i, a in enumerate(self.cameras):
            for b in self.cameras[i + 1:]:
                out[(a.name, b.name)] = float(np.linalg.norm(a.center - b.center))
        return out

    def sanity_check(self, strict: bool = False) -> List[str]:
        """Return a list of human-readable warnings about this rig.

        Catches the failure modes that otherwise show up as silently wrong 3D:
        non-metric translations, a non-SE(3) extrinsic, degenerate baselines,
        a single view, and principal points outside the declared image size.
        """
        problems: List[str] = []
        if len(self) < 2:
            problems.append(
                f"rig has {len(self)} camera(s); POEM-v2 inference needs >= 2 calibrated views "
                "(a single view falls back to GT-seeded reference joints, see lib/models/POEM.py)")
        for cam in self.cameras:
            R = cam.T_cw[:3, :3]
            orth = np.abs(R @ R.T - np.eye(3)).max()
            if orth > 1e-4:
                problems.append(f"{cam.name}: rotation block is not orthonormal (max |RR^T - I| = {orth:.2e})")
            det = float(np.linalg.det(R))
            if abs(det - 1.0) > 1e-3:
                problems.append(f"{cam.name}: det(R) = {det:.4f}, expected +1 (left-handed / mirrored extrinsic?)")
            if cam.image_size is not None:
                w, h = cam.image_size
                cx, cy = cam.optical_center
                if not (0 < cx < w and 0 < cy < h):
                    problems.append(f"{cam.name}: principal point ({cx:.1f}, {cy:.1f}) outside image {w}x{h}")
            if cam.K[0, 0] <= 0 or cam.K[1, 1] <= 0:
                problems.append(f"{cam.name}: non-positive focal length")
        for (a, b), dist in self.baselines().items():
            if dist < 1e-4:
                problems.append(f"baseline {a}-{b} is {dist:.2e} m: views are co-located, triangulation will fail")
            elif dist > 10.0:
                problems.append(f"baseline {a}-{b} is {dist:.1f} m: extrinsics look non-metric (mm instead of m?)")
        if strict and problems:
            raise ValueError("calibration sanity check failed:\n  - " + "\n  - ".join(problems))
        return problems

    def to_dict(self) -> Dict:
        return {"cameras": [c.to_dict() for c in self.cameras]}

    @classmethod
    def from_dict(cls, payload: Dict) -> "CameraRig":
        if "cameras" not in payload:
            raise ValueError("calib payload has no 'cameras' key")
        return cls([Camera.from_dict(c) for c in payload["cameras"]])

    def __repr__(self) -> str:
        return f"CameraRig({self.names})"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_calib_json(path: str) -> CameraRig:
    with open(path, "r") as ifs:
        payload = json.load(ifs)
    return CameraRig.from_dict(payload)


def save_calib_json(rig: CameraRig, path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as ofs:
        json.dump(rig.to_dict(), ofs, indent=2)
    return path


def load_poem_pkl_calib(calib_dir: str, camera_names: Optional[Sequence[str]] = None) -> CameraRig:
    """Read the ``cam_intr/*.pkl`` + ``cam_extr/*.pkl`` layout of the released example data.

    ``tool/infer_hand.py`` reads exactly these files; the pickles hold a 3x3
    K and a 4x4 world->camera matrix respectively.
    """
    intr_dir = os.path.join(calib_dir, "cam_intr")
    extr_dir = os.path.join(calib_dir, "cam_extr")
    for d in (intr_dir, extr_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"expected {d} in the calibration directory")

    if camera_names is None:
        camera_names = sorted(os.path.splitext(f)[0] for f in os.listdir(intr_dir) if f.endswith(".pkl"))

    cams = []
    for name in camera_names:
        with open(os.path.join(intr_dir, f"{name}.pkl"), "rb") as ifs:
            K = np.array(pickle.load(ifs), dtype=np.float64)
        with open(os.path.join(extr_dir, f"{name}.pkl"), "rb") as ifs:
            T_cw = np.array(pickle.load(ifs), dtype=np.float64)
        cams.append(Camera(name=name, K=K, T_cw=T_cw))
    return CameraRig(cams)


def load_opencv_stereo_yml(
    path: str,
    names: Tuple[str, str] = ("cam0", "cam1"),
    rectified: bool = False,
    image_size: Optional[Sequence[int]] = None,
) -> CameraRig:
    """Read an OpenCV FileStorage stereo calibration into a rig.

    Args:
        rectified: if True, read ``P1``/``P2`` (the rectified projection
            matrices from ``cv2.stereoRectify``) and derive the baseline from
            ``P2[0, 3] = -fx * baseline``; the rig is then expressed in the
            *rectified* left-camera frame with zero distortion. If False,
            read ``K1 D1 K2 D2 R T`` from ``cv2.stereoCalibrate``.
    """
    import cv2  # local import: keeps numpy-only consumers cv2-free

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"cannot open OpenCV FileStorage at {path}")

    def _read(key):
        node = fs.getNode(key)
        return None if node.empty() else node.mat()

    try:
        if rectified:
            P1, P2 = _read("P1"), _read("P2")
            if P1 is None or P2 is None:
                raise ValueError(f"{path}: rectified=True needs P1 and P2 nodes")
            K1, K2 = P1[:3, :3], P2[:3, :3]
            fx = float(K2[0, 0])
            if abs(fx) < 1e-9:
                raise ValueError(f"{path}: P2 has zero focal length")
            baseline = -float(P2[0, 3]) / fx
            T_cw0 = np.eye(4)
            T_cw1 = np.eye(4)
            T_cw1[0, 3] = -baseline  # right camera sits at +baseline along world x
            dist = np.zeros(5)
            return CameraRig([
                Camera(names[0], K1, T_cw0, image_size=image_size, dist=dist),
                Camera(names[1], K2, T_cw1, image_size=image_size, dist=dist),
            ])

        K1, K2 = _read("K1"), _read("K2")
        if K1 is None or K2 is None:
            K1, K2 = _read("cameraMatrix1"), _read("cameraMatrix2")
        R, T = _read("R"), _read("T")
        if any(x is None for x in (K1, K2, R, T)):
            raise ValueError(f"{path}: expected K1,K2,R,T (or cameraMatrix1/2) nodes")
        D1, D2 = _read("D1"), _read("D2")
        if D1 is None:
            D1, D2 = _read("distCoeffs1"), _read("distCoeffs2")
        T_cw0 = np.eye(4)
        T_cw1 = np.eye(4)
        T_cw1[:3, :3] = np.asarray(R, dtype=np.float64)
        T_cw1[:3, 3] = np.asarray(T, dtype=np.float64).reshape(3)
        return CameraRig([
            Camera(names[0], K1, T_cw0, image_size=image_size,
                   dist=None if D1 is None else np.asarray(D1).reshape(-1)),
            Camera(names[1], K2, T_cw1, image_size=image_size,
                   dist=None if D2 is None else np.asarray(D2).reshape(-1)),
        ])
    finally:
        fs.release()


def rig_from_rectified_stereo(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline_m: float,
    image_size: Optional[Sequence[int]] = None,
    names: Tuple[str, str] = ("left", "right"),
    fx_right: Optional[float] = None,
    fy_right: Optional[float] = None,
    cx_right: Optional[float] = None,
    cy_right: Optional[float] = None,
) -> CameraRig:
    """Build a two-view rig for an already-rectified stereo pair.

    World frame == rectified left camera frame. The right camera is displaced
    by ``+baseline_m`` along the world x axis, so its world->camera transform
    has translation ``-baseline_m`` in x. Rectified pairs share the rotation
    and (usually) the intrinsics; per-eye intrinsics can still be overridden.
    """
    if baseline_m <= 0:
        raise ValueError(f"baseline must be positive, got {baseline_m}")
    K_l = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_r = np.array([
        [fx if fx_right is None else fx_right, 0.0, cx if cx_right is None else cx_right],
        [0.0, fy if fy_right is None else fy_right, cy if cy_right is None else cy_right],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    T_cw_l = np.eye(4)
    T_cw_r = np.eye(4)
    T_cw_r[0, 3] = -float(baseline_m)
    dist = np.zeros(5)
    return CameraRig([
        Camera(names[0], K_l, T_cw_l, image_size=image_size, dist=dist),
        Camera(names[1], K_r, T_cw_r, image_size=image_size, dist=dist),
    ])
