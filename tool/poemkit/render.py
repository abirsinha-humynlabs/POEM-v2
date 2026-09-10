"""cv2-only overlay drawing (no matplotlib, no open3d, no GUI).

``lib/viztools/draw.py`` cannot be imported without pytorch3d/matplotlib, so
the small amount of drawing needed to eyeball a video-only run lives here.
Joint connectivity matches ``lib.viztools.draw.plot_hand``: joint 0 is the
wrist and joints 1..20 are five 4-long finger chains.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["HAND_BONES", "FINGER_COLORS", "draw_hand_2d", "draw_points_2d", "tile_views"]

HAND_BONES = tuple((i, i + 1) if (i % 4) else (0, i + 1) for i in range(20))
# -> ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),... ) five chains off the wrist

FINGER_COLORS = (
    (0, 0, 255),      # thumb   (BGR)
    (0, 165, 255),    # index
    (0, 255, 255),    # middle
    (0, 255, 0),      # ring
    (255, 0, 0),      # pinky
)


def _finger_color(joint_idx: int):
    if joint_idx == 0:
        return (255, 255, 255)
    return FINGER_COLORS[(joint_idx - 1) // 4]


def draw_hand_2d(
    image_bgr: np.ndarray,
    joints_2d: np.ndarray,
    radius: int = 3,
    thickness: int = 2,
    inplace: bool = False,
) -> np.ndarray:
    """Draw a 21-joint hand skeleton on a BGR image."""
    import cv2

    joints_2d = np.asarray(joints_2d, dtype=np.float64).reshape(-1, 2)
    if joints_2d.shape[0] < 21:
        raise ValueError(f"expected 21 joints, got {joints_2d.shape[0]}")
    canvas = image_bgr if inplace else image_bgr.copy()
    for a, b in HAND_BONES:
        pa, pb = joints_2d[a], joints_2d[b]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), _finger_color(b), thickness, cv2.LINE_AA)
    for idx in range(21):
        p = joints_2d[idx]
        if not np.all(np.isfinite(p)):
            continue
        cv2.circle(canvas, (int(p[0]), int(p[1])), radius, _finger_color(idx), cv2.FILLED, cv2.LINE_AA)
    return canvas


def draw_points_2d(
    image_bgr: np.ndarray,
    points_2d: np.ndarray,
    color: Sequence[int] = (0, 255, 0),
    radius: int = 1,
    inplace: bool = False,
) -> np.ndarray:
    """Scatter arbitrary 2D points (e.g. the 778 reprojected mesh vertices)."""
    import cv2

    canvas = image_bgr if inplace else image_bgr.copy()
    pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    h, w = canvas.shape[:2]
    for p in pts:
        if not np.all(np.isfinite(p)):
            continue
        x, y = int(p[0]), int(p[1])
        if -radius <= x < w + radius and -radius <= y < h + radius:
            cv2.circle(canvas, (x, y), radius, tuple(int(c) for c in color), cv2.FILLED)
    return canvas


def tile_views(images_bgr: Sequence[np.ndarray], scale: float = 0.5,
               labels: Optional[Sequence[str]] = None) -> np.ndarray:
    """Concatenate per-view images horizontally, optionally captioned."""
    import cv2

    tiles = []
    for i, img in enumerate(images_bgr):
        tile = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img.copy()
        if labels is not None and i < len(labels):
            cv2.putText(tile, str(labels[i]), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    height = min(t.shape[0] for t in tiles)
    tiles = [t[:height] for t in tiles]
    return np.concatenate(tiles, axis=1)
