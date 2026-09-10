"""Produce (or inspect) the ``calib.json`` that ``tool/infer_video.py`` consumes.

POEM-v2 lives or dies by its extrinsics, so getting them into the right form
is its own step. Sub-commands:

``stereo``  rectified stereo pair from fx, fy, cx, cy and a baseline in meters
``opencv``  an OpenCV ``stereoCalibrate`` / ``stereoRectify`` FileStorage yml
``poem``    the released example data's ``cam_intr/*.pkl`` + ``cam_extr/*.pkl``
``ring``    N synthetic cameras on a ring aimed at a point (tests, mock rigs)
``check``   validate an existing calib.json and print the warnings

Examples::

    python -m tool.make_calib stereo --fx 700 --fy 700 --cx 640 --cy 360 \
        --baseline 0.12 --image-size 1280 720 --out data/cap/calib.json

    python -m tool.make_calib opencv --yml stereo.yml --rectified --out calib.json
    python -m tool.make_calib check --calib data/cap/calib.json
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import numpy as np

from tool.poemkit.calib import (
    Camera,
    CameraRig,
    load_calib_json,
    load_opencv_stereo_yml,
    load_poem_pkl_calib,
    rig_from_rectified_stereo,
    save_calib_json,
)


def look_at_extrinsic(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, -1.0, 0.0)):
    """World->camera transform for a camera at ``eye`` looking at ``target``.

    Uses the OpenCV convention: +x right, +y down, +z into the scene, which is
    what the pinhole ``K`` in this repo assumes.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("eye and target coincide")
    forward /= norm
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("up vector is parallel to the viewing direction")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    R_wc = np.stack([right, down, forward], axis=0)  # rows: camera axes in world
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_wc
    T_cw[:3, 3] = -R_wc @ eye
    return T_cw


def make_ring_rig(
    num_cams: int = 3,
    radius: float = 0.6,
    height: float = 0.15,
    target: Sequence[float] = (0.0, 0.0, 0.0),
    fx: float = 700.0,
    image_size: Sequence[int] = (1280, 720),
    arc_deg: float = 90.0,
    names: Optional[Sequence[str]] = None,
) -> CameraRig:
    """``num_cams`` cameras spread over ``arc_deg`` of a ring, all aimed at ``target``.

    The rig is expressed so that the first camera's frame is the world frame,
    matching how a real capture is usually calibrated (master = camera 0).
    """
    if num_cams < 2:
        raise ValueError("a POEM-v2 rig needs at least 2 cameras")
    w, h = int(image_size[0]), int(image_size[1])
    K = np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    names = list(names) if names is not None else [f"cam{i}" for i in range(num_cams)]
    if len(names) != num_cams:
        raise ValueError(f"got {len(names)} names for {num_cams} cameras")

    angles = np.linspace(-np.deg2rad(arc_deg) / 2.0, np.deg2rad(arc_deg) / 2.0, num_cams)
    target = np.asarray(target, dtype=np.float64)
    extrs = []
    for angle in angles:
        eye = target + np.array([radius * np.sin(angle), -height, -radius * np.cos(angle)])
        extrs.append(look_at_extrinsic(eye, target))

    # re-express everything in camera 0's frame so cam0 == world == master
    T_cw0 = extrs[0]
    cams = [Camera(names[i], K, extrs[i] @ np.linalg.inv(T_cw0), image_size=(w, h), dist=np.zeros(5))
            for i in range(num_cams)]
    return CameraRig(cams)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stereo", help="rectified stereo pair from intrinsics + baseline")
    s.add_argument("--fx", type=float, required=True)
    s.add_argument("--fy", type=float, default=None, help="defaults to fx")
    s.add_argument("--cx", type=float, required=True)
    s.add_argument("--cy", type=float, required=True)
    s.add_argument("--baseline", type=float, required=True, help="meters between the two optical centers")
    s.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("W", "H"))
    s.add_argument("--names", nargs=2, default=["left", "right"])
    s.add_argument("--out", required=True)

    o = sub.add_parser("opencv", help="from an OpenCV FileStorage stereo calibration")
    o.add_argument("--yml", required=True)
    o.add_argument("--rectified", action="store_true", help="read P1/P2 instead of K1,K2,R,T")
    o.add_argument("--names", nargs=2, default=["left", "right"])
    o.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("W", "H"))
    o.add_argument("--out", required=True)

    q = sub.add_parser("poem", help="from the released example data's pickles")
    q.add_argument("--calib-dir", required=True, help="the dir containing cam_intr/ and cam_extr/")
    q.add_argument("--names", nargs="*", default=None, help="camera order; first is the master")
    q.add_argument("--out", required=True)

    r = sub.add_parser("ring", help="synthetic ring rig (tests / mock captures)")
    r.add_argument("--num-cams", type=int, default=3)
    r.add_argument("--radius", type=float, default=0.6)
    r.add_argument("--height", type=float, default=0.15)
    r.add_argument("--fx", type=float, default=700.0)
    r.add_argument("--arc-deg", type=float, default=90.0)
    r.add_argument("--image-size", nargs=2, type=int, default=[1280, 720], metavar=("W", "H"))
    r.add_argument("--names", nargs="*", default=None)
    r.add_argument("--out", required=True)

    c = sub.add_parser("check", help="validate a calib.json")
    c.add_argument("--calib", required=True)
    c.add_argument("--strict", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "check":
        rig = load_calib_json(args.calib)
        problems = rig.sanity_check(strict=args.strict)
        print(json.dumps({
            "cameras": rig.names,
            "baselines_m": {f"{a}-{b}": round(d, 4) for (a, b), d in rig.baselines().items()},
            "warnings": problems,
        }, indent=2))
        return 1 if problems else 0

    if args.cmd == "stereo":
        rig = rig_from_rectified_stereo(
            fx=args.fx, fy=args.fy if args.fy is not None else args.fx,
            cx=args.cx, cy=args.cy, baseline_m=args.baseline,
            image_size=args.image_size, names=tuple(args.names),
        )
    elif args.cmd == "opencv":
        rig = load_opencv_stereo_yml(args.yml, names=tuple(args.names), rectified=args.rectified,
                                     image_size=args.image_size)
    elif args.cmd == "poem":
        rig = load_poem_pkl_calib(args.calib_dir, camera_names=args.names or None)
    elif args.cmd == "ring":
        rig = make_ring_rig(num_cams=args.num_cams, radius=args.radius, height=args.height, fx=args.fx,
                            image_size=args.image_size, arc_deg=args.arc_deg, names=args.names or None)
    else:  # pragma: no cover - argparse enforces the choices
        raise ValueError(args.cmd)

    save_calib_json(rig, args.out)
    problems = rig.sanity_check()
    print(f"wrote {args.out}: {rig.names}")
    for problem in problems:
        print(f"[calib warning] {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
