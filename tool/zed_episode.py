#!/usr/bin/env python
"""Helpers for running the video-only path on a ZED stereo episode.

Two jobs the CLI does not do on its own:

``calib``   turn a ZED ``calibration.json`` into the ``calib.json`` the runner
            reads. The ZED file stores its *rectified* block with identical
            left/right intrinsics, an identity rotation and a field named
            ``baseline_meters`` that actually holds **millimetres** -- feeding
            that straight through would put the second camera 120 m away and
            produce plausible-looking but meaningless metric output, which is
            exactly the failure ``CameraRig.sanity_check()`` screens for.

``convert`` turn one or more ``keypoints.npz`` files from ``tool.infer_video``
            into the flat per-detection layout the wrist-trajectory renderers
            expect: ``frame_idx``/``kp2d``/``kp3d_cam``/``K``. Several inputs
            (e.g. a right-hand and a left-hand pass) are concatenated, since
            that layout carries one row per detection, not per frame.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _K(block):
    return np.array([[block["fx"], 0.0, block["cx"]],
                     [0.0, block["fy"], block["cy"]],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def baseline_meters(calib: dict) -> float:
    """Baseline in METERS from a ZED calibration block.

    ``baseline_meters`` is misnamed in these files: the ZED SDK reports the
    baseline in millimetres and the exporter copied the number verbatim. A
    stereo rig is centimetres apart, so a value above 10 means millimetres.
    """
    raw = float(calib["baseline_meters"])
    return raw / 1000.0 if raw > 10.0 else raw


def build_calib(zed_calib_path: str, out_path: str, width: int, height: int,
                left_name: str = "left_eye", right_name: str = "right_eye") -> dict:
    with open(zed_calib_path) as ifs:
        zed = json.load(ifs)
    block = zed["rectified"] if "rectified" in zed else zed["raw"]
    base = baseline_meters(block)

    # Rectified pair: both cameras share intrinsics and orientation; the right
    # camera sits +baseline along the master's x axis, so T_cw maps a master
    # point to the right camera by subtracting it.
    T_left = np.eye(4)
    T_right = np.eye(4)
    T_right[0, 3] = -base

    rig = {"cameras": [
        {"name": left_name, "image_size": [width, height],
         "K": _K(block["left"]).tolist(), "T_cw": T_left.tolist()},
        {"name": right_name, "image_size": [width, height],
         "K": _K(block["right"]).tolist(), "T_cw": T_right.tolist()},
    ]}
    with open(out_path, "w") as ofs:
        json.dump(rig, ofs, indent=2)
    print("wrote %s: baseline %.5f m, fx %.2f, image %dx%d"
          % (out_path, base, block["left"]["fx"], width, height))
    return rig


def convert(npz_paths, calib_path: str, out_path: str, master: str = "left_eye"):
    """POEM output -> the renderer's flat per-detection layout."""
    with open(calib_path) as ifs:
        rig = json.load(ifs)
    cam = next(c for c in rig["cameras"] if c["name"] == master)
    K = np.asarray(cam["K"], dtype=np.float64)

    frame_idx, kp2d, kp3d = [], [], []
    for path in npz_paths:
        if not os.path.isfile(path):
            print("  skip (absent): %s" % path)
            continue
        d = np.load(path)
        joints = d["joints_master"]            # (F, 21, 3) meters, master frame
        if joints.size == 0:
            print("  skip (empty): %s" % path)
            continue
        # project into the FULL master frame -- the renderer draws on left_eye.mp4
        uvw = joints @ K.T
        uv = uvw[..., :2] / np.clip(uvw[..., 2:3], 1e-6, None)
        frame_idx.append(np.asarray(d["frame_ids"], dtype=np.int64))
        kp2d.append(uv.astype(np.float32))
        kp3d.append(joints.astype(np.float32))
        print("  + %-48s %d detections" % (os.path.basename(os.path.dirname(path)), len(d["frame_ids"])))

    if not frame_idx:
        raise SystemExit("no usable keypoints in: %s" % ", ".join(npz_paths))

    frame_idx = np.concatenate(frame_idx)
    kp2d = np.concatenate(kp2d)
    kp3d = np.concatenate(kp3d)
    order = np.argsort(frame_idx, kind="stable")   # renderer walks frames forward

    np.savez_compressed(out_path, frame_idx=frame_idx[order], kp2d=kp2d[order],
                        kp3d_cam=kp3d[order], K=K.astype(np.float32))
    print("wrote %s: %d detections over frames %d..%d"
          % (out_path, len(frame_idx), frame_idx.min(), frame_idx.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calib", help="ZED calibration.json -> calib.json")
    c.add_argument("--zed", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--width", type=int, required=True)
    c.add_argument("--height", type=int, required=True)

    v = sub.add_parser("convert", help="keypoints.npz -> renderer npz")
    v.add_argument("--npz", nargs="+", required=True)
    v.add_argument("--calib", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--master", default="left_eye")

    a = ap.parse_args()
    if a.cmd == "calib":
        build_calib(a.zed, a.out, a.width, a.height)
    else:
        convert(a.npz, a.calib, a.out, a.master)


if __name__ == "__main__":
    main()
