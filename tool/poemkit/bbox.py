"""Per-view hand bounding boxes.

POEM-v2 is *not* a detector: it consumes a square crop around the hand in
every view. ``tool/infer_hand.py`` reads pre-computed boxes from
``<mask_dir>/<cam>/bbox/%05d.npy``. For video-only capture you need something
to produce those boxes, so this module offers interchangeable providers with
one interface::

    provider = build_bbox_provider(...)
    bbox = provider.get(cam_name, frame_id, frame)   # -> (4,) xyxy or None

Providers
---------
``npy``        the released layout: ``<root>/<cam>/bbox/%05d.npy``
``json``       ``{"<cam>": {"<frame_id>": [x0, y0, x1, y1]}}``
``mediapipe``  MediaPipe Hands 2D landmarks -> tight box (CPU, optional dep)
``full``       whole frame; a sanity/debug fallback, expect poor accuracy

Every provider is wrapped in :class:`BBoxTracker`, which holds the last good
box per camera so a single dropped detection does not kill the frame. That
matters because a frame is usable only when >= 2 views have a box.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "bbox_from_points",
    "clip_bbox",
    "BBoxProvider",
    "NpyBBoxProvider",
    "JsonBBoxProvider",
    "MediaPipeBBoxProvider",
    "FullFrameBBoxProvider",
    "BBoxTracker",
    "build_bbox_provider",
]


def bbox_from_points(points_2d: np.ndarray, pad_ratio: float = 0.15) -> np.ndarray:
    """Tight xyxy box around 2D points, padded by a fraction of its larger side."""
    pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        raise ValueError("no points to build a bbox from")
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    pad = max(x1 - x0, y1 - y0) * float(pad_ratio)
    return np.array([x0 - pad, y0 - pad, x1 + pad, y1 + pad], dtype=np.float32)


def clip_bbox(bbox: Sequence[float], image_size: Sequence[int]) -> np.ndarray:
    """Clamp a box to the frame. Note the crop itself may still run off-frame:
    ``bbox_get_center_scale`` expands by 2x and warpAffine zero-pads, which is
    what the model was trained on."""
    w, h = float(image_size[0]), float(image_size[1])
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4].copy()
    b[0] = np.clip(b[0], 0, w - 1)
    b[1] = np.clip(b[1], 0, h - 1)
    b[2] = np.clip(b[2], 1, w)
    b[3] = np.clip(b[3], 1, h)
    return b


class BBoxProvider:
    """Interface: return an xyxy box for ``(cam_name, frame_id)`` or None."""

    name = "base"

    def get(self, cam_name: str, frame_id: int, frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NpyBBoxProvider(BBoxProvider):
    """``<root>/<cam>/bbox/%05d.npy`` -- the layout of the released example data.

    The stored array is either ``(4,)`` or ``(K, 4)``; as in the reference
    demo the first row wins. An empty array means "no hand here".
    """

    name = "npy"

    def __init__(self, root: str, subdir: str = "bbox", pattern: str = "{frame_id:05d}.npy"):
        self.root = root
        self.subdir = subdir
        self.pattern = pattern

    def _path(self, cam_name: str, frame_id: int) -> str:
        return os.path.join(self.root, cam_name, self.subdir, self.pattern.format(frame_id=frame_id))

    def get(self, cam_name, frame_id, frame=None):
        path = self._path(cam_name, frame_id)
        if not os.path.isfile(path):
            return None
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim > 1:
            arr = arr[0]
        if arr.shape[0] < 4:
            return None
        return arr[:4]

    def available_frames(self, cam_name: str) -> int:
        return len(glob.glob(os.path.join(self.root, cam_name, self.subdir, "*.npy")))


class JsonBBoxProvider(BBoxProvider):
    """``{"<cam>": {"<frame_id>": [x0,y0,x1,y1] | null}}`` -- easiest hand-off
    format from an external detector."""

    name = "json"

    def __init__(self, path: str):
        with open(path, "r") as ifs:
            payload = json.load(ifs)
        self.table: Dict[str, Dict[str, Optional[list]]] = payload.get("boxes", payload)

    def get(self, cam_name, frame_id, frame=None):
        per_cam = self.table.get(cam_name)
        if not per_cam:
            return None
        value = per_cam.get(str(frame_id), per_cam.get(frame_id))
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr[:4] if arr.shape[0] >= 4 else None


class MediaPipeBBoxProvider(BBoxProvider):
    """Boxes from MediaPipe Hands 2D landmarks (runs fine on CPU/macOS).

    MediaPipe supplies the *2D* landmarks; POEM-v2 turns the multi-view crops
    into metric 3D. Optional dependency -- ``pip install mediapipe``. The
    landmarks are also returned via :attr:`last_landmarks` so they can be
    stored next to the POEM output for comparison.

    ``strict_side`` decides what happens when the requested hand is not found
    in a view. The default (False) falls back to whichever hand was detected,
    which is right for a single-hand capture. Set it for footage where both
    hands are visible -- egocentric video especially -- so a view is dropped
    rather than filled with the other hand.
    """

    name = "mediapipe"

    def __init__(
        self,
        hand_side: str = "rh",
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        pad_ratio: float = 0.25,
        prefer_side: bool = True,
        strict_side: bool = False,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("MediaPipe backend requested but mediapipe is not installed "
                              "(`pip install mediapipe`)") from exc
        self._mp = mp
        self.hand_side = hand_side
        self.prefer_side = prefer_side
        self.strict_side = strict_side
        self.pad_ratio = pad_ratio
        # one graph per camera: MediaPipe Hands is stateful across frames
        self._solvers: Dict[str, object] = {}
        self._solver_kwargs = dict(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.last_landmarks: Dict[str, Optional[np.ndarray]] = {}

    def _solver(self, cam_name: str):
        if cam_name not in self._solvers:
            self._solvers[cam_name] = self._mp.solutions.hands.Hands(**self._solver_kwargs)
        return self._solvers[cam_name]

    def get(self, cam_name, frame_id, frame=None):
        if frame is None:
            raise ValueError("MediaPipe backend needs the RGB frame")
        h, w = frame.shape[:2]
        result = self._solver(cam_name).process(np.ascontiguousarray(frame))
        self.last_landmarks[cam_name] = None
        if not result.multi_hand_landmarks:
            return None

        wanted = "Right" if self.hand_side == "rh" else "Left"
        chosen = None
        if self.prefer_side and result.multi_handedness:
            for lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                # MediaPipe labels handedness assuming a mirrored (selfie) image;
                # for a normal forward-facing camera its "Right" is the real right hand.
                if handed.classification[0].label == wanted:
                    chosen = lm
                    break
        if chosen is None:
            if self.strict_side:
                # Two-hand footage: substituting the other hand here would box a
                # *different* hand in each view, and the model would triangulate
                # across them -- geometrically consistent nonsense that shows up
                # only as a large reprojection error. Refuse the view instead;
                # the frame survives if another view still has the wanted hand,
                # and is skipped otherwise.
                return None
            chosen = result.multi_hand_landmarks[0]

        pts = np.array([[p.x * w, p.y * h] for p in chosen.landmark], dtype=np.float32)
        self.last_landmarks[cam_name] = pts
        return bbox_from_points(pts, pad_ratio=self.pad_ratio)

    def close(self):
        for solver in self._solvers.values():
            solver.close()
        self._solvers.clear()


class FullFrameBBoxProvider(BBoxProvider):
    """Whole frame as the box. Useful to prove the plumbing end to end; the
    resulting crop is far coarser than training, so expect degraded 3D."""

    name = "full"

    def __init__(self, image_size: Optional[Sequence[int]] = None):
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))

    def get(self, cam_name, frame_id, frame=None):
        if frame is not None:
            h, w = frame.shape[:2]
        elif self.image_size is not None:
            w, h = self.image_size
        else:
            return None
        return np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)


class BBoxTracker:
    """Wraps a provider and remembers the last good box per camera.

    ``max_age`` bounds how many consecutive misses may reuse the stale box
    before the view is dropped; ``0`` disables reuse entirely.
    """

    def __init__(self, provider: BBoxProvider, max_age: int = 5):
        self.provider = provider
        self.max_age = int(max_age)
        self._last: Dict[str, np.ndarray] = {}
        self._age: Dict[str, int] = {}
        self.stats = {"hit": 0, "reused": 0, "miss": 0}

    def get(self, cam_name: str, frame_id: int, frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        bbox = self.provider.get(cam_name, frame_id, frame)
        if bbox is not None:
            self._last[cam_name] = np.asarray(bbox, dtype=np.float32)
            self._age[cam_name] = 0
            self.stats["hit"] += 1
            return self._last[cam_name]

        age = self._age.get(cam_name, self.max_age + 1) + 1
        self._age[cam_name] = age
        if cam_name in self._last and age <= self.max_age:
            self.stats["reused"] += 1
            return self._last[cam_name]
        self.stats["miss"] += 1
        return None

    def close(self):
        self.provider.close()


def build_bbox_provider(
    backend: str,
    bbox_root: Optional[str] = None,
    bbox_json: Optional[str] = None,
    hand_side: str = "rh",
    image_size: Optional[Sequence[int]] = None,
    max_age: int = 5,
    strict_side: bool = False,
) -> BBoxTracker:
    """Factory used by the CLI; always returns a tracker-wrapped provider."""
    backend = backend.lower()
    if backend == "npy":
        if not bbox_root:
            raise ValueError("backend 'npy' requires --bbox-root")
        provider: BBoxProvider = NpyBBoxProvider(bbox_root)
    elif backend == "json":
        if not bbox_json:
            raise ValueError("backend 'json' requires --bbox-json")
        provider = JsonBBoxProvider(bbox_json)
    elif backend == "mediapipe":
        provider = MediaPipeBBoxProvider(hand_side=hand_side, strict_side=strict_side)
    elif backend == "full":
        provider = FullFrameBBoxProvider(image_size=image_size)
    else:
        raise ValueError(f"unknown bbox backend {backend!r}; expected npy|json|mediapipe|full")
    return BBoxTracker(provider, max_age=max_age)
