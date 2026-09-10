"""Generate a synthetic multi-view hand capture (videos + calibration + boxes + GT).

Why this exists: the real check of a video-only pipeline is whether the
geometry survives the trip from pixels to metric 3D, and that can be tested
without a GPU or a checkpoint -- as long as there is a capture whose answer is
known. This renders one: an articulated 21-joint hand moving in front of a
2..N camera rig, with

  <out>/<cam>.mp4                  synchronised videos, one per view
  <out>/calib.json                 intrinsics + extrinsics (cam0 == world)
  <out>/bbox.json                  per-view boxes for --bbox-backend json
  <out>/bbox/<cam>/bbox/%05d.npy   the same boxes in the released layout
  <out>/hand_labels.json           {sequence: left|right}
  <out>/gt.npz                     GT joints (world + per-view 2D + per-cam 3D)

The images are cartoon hands, so they say nothing about model *accuracy* --
they exercise calibration, synchronisation, cropping, mirroring, packing,
export and (on a GPU box) the forward pass. Accuracy needs the released
example data or a real capture; see plan_of_action.md.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from tool.make_calib import make_ring_rig
from tool.poemkit.calib import CameraRig, rig_from_rectified_stereo, save_calib_json
from tool.poemkit.geometry import project_points, transf_points

# 21 joints: 0 = wrist, then 5 chains of 4 (thumb, index, middle, ring, pinky),
# matching tool/poemkit/render.HAND_BONES and lib.viztools.draw.plot_hand.
FINGER_ROOT_X = (-0.035, -0.018, 0.0, 0.017, 0.032)
FINGER_ROOT_Y = (0.020, 0.075, 0.080, 0.075, 0.068)
SEGMENT_LEN = ((0.032, 0.030, 0.024), (0.040, 0.026, 0.020), (0.044, 0.028, 0.021), (0.040, 0.026, 0.020),
               (0.032, 0.022, 0.018))
SPREAD_DEG = (-45.0, -8.0, 0.0, 8.0, 16.0)


def canonical_hand(curl: float = 0.0, hand_side: str = "rh") -> np.ndarray:
    """A plausible 21-joint hand in its own frame, meters. ``curl`` in [0, 1] flexes the fingers."""
    joints = np.zeros((21, 3), dtype=np.float64)
    for finger in range(5):
        base = 1 + finger * 4  # [root/MCP, PIP, DIP, TIP]
        root = np.array([FINGER_ROOT_X[finger], FINGER_ROOT_Y[finger], 0.0])
        spread = np.deg2rad(SPREAD_DEG[finger])
        direction = np.array([np.sin(spread), np.cos(spread), 0.0])
        joints[base] = root

        point = root
        angle = 0.0
        for seg in range(3):
            # each successive segment bends further out of the palm plane
            angle += curl * np.deg2rad(35.0 + 10.0 * seg)
            step = SEGMENT_LEN[finger][seg] * (direction * np.cos(angle) + np.array([0.0, 0.0, -np.sin(angle)]))
            point = point + step
            joints[base + 1 + seg] = point
    if hand_side == "lh":
        joints[:, 0] *= -1.0
    return joints


def _rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def hand_trajectory(num_frames: int, hand_side: str = "rh", depth: float = 0.55,
                    radius: float = 0.06) -> np.ndarray:
    """``(F, 21, 3)`` world-frame joints: the hand orbits, rotates and flexes."""
    out = np.zeros((num_frames, 21, 3), dtype=np.float64)
    for f in range(num_frames):
        t = f / max(1, num_frames - 1)
        curl = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)  # open -> fist -> open
        local = canonical_hand(curl=curl, hand_side=hand_side)
        R = _rot_xyz(np.deg2rad(-20.0 + 25.0 * np.sin(2 * np.pi * t)),
                     np.deg2rad(30.0 * np.sin(2 * np.pi * t + 1.0)),
                     np.deg2rad(15.0 * np.cos(2 * np.pi * t)))
        center = np.array([radius * np.cos(2 * np.pi * t), radius * np.sin(2 * np.pi * t) * 0.5, depth])
        out[f] = local @ R.T + center
    return out


def _draw_hand(canvas: np.ndarray, uv: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a filled cartoon hand so the frames are not pure line art."""
    import cv2

    from tool.poemkit.render import HAND_BONES

    palm_idx = [0, 1, 5, 9, 13, 17]
    palm = uv[palm_idx].astype(np.int32)
    if np.all(np.isfinite(palm)):
        cv2.fillConvexPoly(canvas, cv2.convexHull(palm), (172, 190, 214))
    for a, b in HAND_BONES:
        pa, pb = uv[a], uv[b]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (160, 178, 205), 12, cv2.LINE_AA)
    for idx in range(21):
        p = uv[idx]
        if not np.all(np.isfinite(p)):
            continue
        cv2.circle(canvas, (int(p[0]), int(p[1])), 7, (140, 158, 190), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(canvas, (int(p[0]), int(p[1])), 3, (90, 105, 135), cv2.FILLED, cv2.LINE_AA)
    noise = rng.normal(0.0, 3.0, canvas.shape)
    return np.clip(canvas.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _background(w: int, h: int, seed: int) -> np.ndarray:
    """A static textured background: gives the backbone something to lock onto."""
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = (70, 70, 75)
    for _ in range(120):
        x, y = rng.integers(0, w), rng.integers(0, h)
        rr = int(rng.integers(6, 40))
        color = tuple(int(c) for c in rng.integers(40, 130, size=3))
        y0, y1 = max(0, y - rr), min(h, y + rr)
        x0, x1 = max(0, x - rr), min(w, x + rr)
        base[y0:y1, x0:x1] = color
    return base


def build_rig(rig_kind: str, num_cams: int, image_size: Sequence[int], fx: float, baseline: float) -> CameraRig:
    if rig_kind == "stereo":
        w, h = image_size
        return rig_from_rectified_stereo(fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0, baseline_m=baseline,
                                         image_size=image_size, names=("left", "right"))
    if rig_kind == "ring":
        return make_ring_rig(num_cams=num_cams, radius=0.5, height=0.1, fx=fx, image_size=image_size,
                             arc_deg=70.0, target=(0.0, 0.0, 0.5))
    raise ValueError(f"unknown rig kind {rig_kind!r}; expected stereo|ring")


def generate(
    out_dir: str,
    rig_kind: str = "stereo",
    num_cams: int = 2,
    num_frames: int = 30,
    image_size: Sequence[int] = (640, 480),
    fx: float = 500.0,
    baseline: float = 0.12,
    hand_side: str = "rh",
    fps: float = 30.0,
    seed: int = 0,
    drop_view_frames: Optional[Dict[str, Sequence[int]]] = None,
    bbox_jitter_px: float = 3.0,
    sequence_name: Optional[str] = None,
) -> Dict:
    """Render the capture and write every artifact. Returns a manifest dict."""
    from tool.poemkit.bbox import bbox_from_points
    from tool.poemkit.video import VideoWriter

    os.makedirs(out_dir, exist_ok=True)
    sequence_name = sequence_name or os.path.basename(os.path.normpath(out_dir))
    rng = np.random.default_rng(seed)
    w, h = int(image_size[0]), int(image_size[1])

    rig = build_rig(rig_kind, num_cams, (w, h), fx, baseline)
    save_calib_json(rig, os.path.join(out_dir, "calib.json"))

    joints_world = hand_trajectory(num_frames, hand_side=hand_side)
    drop_view_frames = {k: set(v) for k, v in (drop_view_frames or {}).items()}

    backgrounds = {cam.name: _background(w, h, seed + i) for i, cam in enumerate(rig)}
    writers = {cam.name: VideoWriter(os.path.join(out_dir, f"{cam.name}.mp4"), fps=fps) for cam in rig}

    joints_2d = np.full((len(rig), num_frames, 21, 2), np.nan)
    joints_cam = np.zeros((len(rig), num_frames, 21, 3))
    boxes: Dict[str, Dict[str, Optional[List[float]]]] = {cam.name: {} for cam in rig}

    try:
        for f in range(num_frames):
            for ci, cam in enumerate(rig):
                pts_cam = transf_points(cam.T_cw, joints_world[f])
                uv = project_points(cam.K, pts_cam)
                joints_cam[ci, f] = pts_cam
                joints_2d[ci, f] = uv

                canvas = backgrounds[cam.name].copy()
                canvas = _draw_hand(canvas, uv, rng)
                writers[cam.name].write_bgr(canvas)

                if f in drop_view_frames.get(cam.name, ()):  # simulated detector failure
                    boxes[cam.name][str(f)] = None
                    continue
                box = bbox_from_points(uv, pad_ratio=0.2).astype(np.float64)
                box += rng.normal(0.0, bbox_jitter_px, size=4)
                boxes[cam.name][str(f)] = box.tolist()

                bbox_dir = os.path.join(out_dir, "bbox", cam.name, "bbox")
                os.makedirs(bbox_dir, exist_ok=True)
                np.save(os.path.join(bbox_dir, f"{f:05d}.npy"), box.astype(np.float32))
    finally:
        for writer in writers.values():
            writer.close()

    with open(os.path.join(out_dir, "bbox.json"), "w") as ofs:
        json.dump({"boxes": boxes}, ofs)
    with open(os.path.join(out_dir, "hand_labels.json"), "w") as ofs:
        json.dump({sequence_name: "right" if hand_side == "rh" else "left"}, ofs)

    np.savez_compressed(
        os.path.join(out_dir, "gt.npz"),
        joints_world=joints_world.astype(np.float32),
        joints_2d=joints_2d.astype(np.float32),
        joints_cam=joints_cam.astype(np.float32),
        cameras=np.array(rig.names),
        hand_side=np.array(hand_side),
        num_frames=np.array(num_frames),
    )

    manifest = {
        "out_dir": os.path.abspath(out_dir),
        "sequence": sequence_name,
        "rig": rig_kind,
        "cameras": rig.names,
        "videos": {cam.name: os.path.join(out_dir, f"{cam.name}.mp4") for cam in rig},
        "calib": os.path.join(out_dir, "calib.json"),
        "bbox_json": os.path.join(out_dir, "bbox.json"),
        "bbox_root": os.path.join(out_dir, "bbox"),
        "gt": os.path.join(out_dir, "gt.npz"),
        "num_frames": num_frames,
        "image_size": [w, h],
        "hand_side": hand_side,
        "dropped_views": {k: sorted(v) for k, v in drop_view_frames.items()},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as ofs:
        json.dump(manifest, ofs, indent=2)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory for the capture")
    p.add_argument("--rig", choices=["stereo", "ring"], default="stereo")
    p.add_argument("--num-cams", type=int, default=3, help="only used by --rig ring")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--image-size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--fx", type=float, default=500.0)
    p.add_argument("--baseline", type=float, default=0.12, help="only used by --rig stereo")
    p.add_argument("--hand-side", choices=["rh", "lh"], default="rh")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-view", action="append", default=[], metavar="CAM:F1,F2",
                   help="simulate a detector failure: no bbox for CAM on those frames")
    args = p.parse_args(argv)

    drops: Dict[str, Sequence[int]] = {}
    for entry in args.drop_view:
        cam, _, frames = entry.partition(":")
        drops[cam] = [int(v) for v in frames.split(",") if v.strip()]

    manifest = generate(
        out_dir=args.out,
        rig_kind=args.rig,
        num_cams=args.num_cams,
        num_frames=args.frames,
        image_size=args.image_size,
        fx=args.fx,
        baseline=args.baseline,
        hand_side=args.hand_side,
        fps=args.fps,
        seed=args.seed,
        drop_view_frames=drops,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
