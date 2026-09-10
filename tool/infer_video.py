"""Video-only POEM-v2 inference: N synchronised videos + calibration -> 3D hand keypoints.

This is a headless, dataset-free generalisation of ``tool/infer_hand.py``:

* any number of views >= 2 (stereo included), named however you like;
* calibration from ``calib.json``, an OpenCV stereo yml, or the released
  ``cam_intr/cam_extr`` pickles;
* hand boxes from a detector (MediaPipe), from precomputed files, or full-frame;
* no ``cv2.imshow`` / open3d, so it runs over SSH;
* results written as ``.npz`` + ``.json``, with an optional overlay video;
* ``--dry-run`` validates the whole pipeline *without* torch, which is how you
  check a capture on a laptop before renting a GPU.

Only RGB video is consumed: no depth, no IMU, no per-frame ground truth.
Two or more calibrated views are mandatory -- see ``plan_of_action.md``.

Examples
--------
Rectified stereo pair, MediaPipe boxes, on a GPU box::

    python -m tool.infer_video \
        --calib data/mycapture/calib.json \
        --video left=data/mycapture/left.mp4 --video right=data/mycapture/right.mp4 \
        --hand-side rh --bbox-backend mediapipe \
        --cfg config/release/eval_single.yaml --model medium \
        --reload checkpoints/medium.pth.tar --device cuda:0 \
        --out exp/mycapture --overlay

Validate a capture with no GPU and no torch installed::

    python -m tool.infer_video --calib ... --video ... --dry-run --out /tmp/check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from tool.poemkit import bbox as bbox_mod
from tool.poemkit import calib as calib_mod
from tool.poemkit import geometry as geo
from tool.poemkit import video as video_mod
from tool.poemkit.views import PackError, joints_uv_to_original, prepare_views, project_to_views

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".MP4", ".MKV")


# ---------------------------------------------------------------------------
# input resolution
# ---------------------------------------------------------------------------
def parse_video_args(video_args: Sequence[str], video_dir: Optional[str], rig_names: Sequence[str]) -> Dict[str, str]:
    """Resolve ``--video name=path`` entries and/or ``--video-dir`` into {camera: path}."""
    paths: Dict[str, str] = {}
    for entry in video_args or []:
        if "=" in entry:
            name, path = entry.split("=", 1)
        else:  # bare path: infer the camera name from the file stem
            path = entry
            name = os.path.splitext(os.path.basename(path))[0]
        paths[name.strip()] = os.path.expanduser(path.strip())

    if video_dir:
        for name in rig_names:
            if name in paths:
                continue
            for ext in VIDEO_EXTS:
                candidate = os.path.join(video_dir, f"{name}{ext}")
                if os.path.isfile(candidate):
                    paths[name] = candidate
                    break

    missing_files = {n: p for n, p in paths.items() if not os.path.isfile(p)}
    if missing_files:
        raise FileNotFoundError(f"video file(s) not found: {missing_files}")
    if not paths:
        raise ValueError("no videos resolved; pass --video name=path (repeatable) or --video-dir")

    unknown = [n for n in paths if n not in rig_names]
    if unknown:
        raise ValueError(f"video name(s) {unknown} are not in the calibration ({list(rig_names)}); "
                         "names must match so each view gets its own K and T_cw")
    return {name: paths[name] for name in rig_names if name in paths}


def load_rig(args) -> calib_mod.CameraRig:
    """Load calibration from whichever of the supported sources was given."""
    if args.calib:
        path = os.path.expanduser(args.calib)
        if os.path.isdir(path):
            return calib_mod.load_poem_pkl_calib(path)
        if path.endswith((".yml", ".yaml", ".xml")):
            return calib_mod.load_opencv_stereo_yml(path, rectified=args.calib_rectified)
        return calib_mod.load_calib_json(path)
    if args.stereo_params:
        fx, fy, cx, cy, baseline = (float(v) for v in args.stereo_params)
        return calib_mod.rig_from_rectified_stereo(fx, fy, cx, cy, baseline, names=tuple(args.stereo_names))
    raise ValueError("no calibration given: pass --calib (json / OpenCV yml / example-data dir) "
                     "or --stereo-params fx fy cx cy baseline")


def resolve_hand_side(args, sequence_name: str) -> str:
    """'rh' / 'lh' from --hand-side, or from a ``{sequence: 'left'|'right'}`` json."""
    if args.hand_side_json:
        with open(os.path.expanduser(args.hand_side_json), "r") as ifs:
            table = json.load(ifs)
        value = table.get(sequence_name)
        if value is None:
            raise KeyError(f"{args.hand_side_json} has no entry for sequence {sequence_name!r} "
                           f"(keys: {sorted(table)[:8]}...)")
        return "rh" if str(value).lower().startswith("r") else "lh"
    return args.hand_side


def frame_indices(num_frame: int, start: int, end: Optional[int], step: int, limit: Optional[int]) -> List[int]:
    stop = num_frame if end is None or end < 0 else min(end, num_frame)
    idxs = list(range(max(0, start), stop, max(1, step)))
    return idxs[:limit] if limit else idxs


# ---------------------------------------------------------------------------
# the run itself
# ---------------------------------------------------------------------------
def run_sequence(
    rig: calib_mod.CameraRig,
    video_paths: Dict[str, str],
    out_dir: str,
    hand_side: str = "rh",
    bbox_backend: str = "mediapipe",
    bbox_root: Optional[str] = None,
    bbox_json: Optional[str] = None,
    bbox_max_age: int = 5,
    predictor=None,
    dry_run: bool = False,
    video_backend: str = "auto",
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
    min_views: int = 2,
    master: Optional[str] = None,
    out_res: Sequence[int] = (256, 256),
    bbox_expand: float = 2.0,
    bbox_mindim: float = 200.0,
    save_verts: bool = False,
    overlay: bool = False,
    overlay_scale: float = 0.5,
    overlay_fps: float = 30.0,
    progress: bool = True,
) -> Dict:
    """Process one synchronised multi-view sequence.

    ``predictor`` is any object with ``predict_packet(packet) -> dict`` (i.e.
    :class:`tool.poemkit.runner.PoemRunner`, or a stub in the tests). With
    ``dry_run=True`` no predictor is needed and only the data pipeline runs.

    Returns a report dict; also writes ``keypoints.npz`` and ``report.json``
    into ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    if predictor is None and not dry_run:
        raise ValueError("run_sequence needs a predictor unless dry_run=True")

    warnings: List[str] = list(rig.sanity_check())
    reader = video_mod.MultiViewReader(video_paths, backend=video_backend)
    warnings += reader.check_alignment()

    first_size = reader.frame_size(list(video_paths.keys())[0])
    for name in video_paths:
        cam = rig[name]
        size = reader.frame_size(name)
        if cam.image_size is not None and tuple(cam.image_size) != tuple(size):
            warnings.append(f"{name}: calibration image_size {cam.image_size} != video {tuple(size)}; "
                            "intrinsics must describe the frames you feed in (rescale K if you resized)")

    tracker = bbox_mod.build_bbox_provider(
        bbox_backend,
        bbox_root=bbox_root,
        bbox_json=bbox_json,
        hand_side=hand_side,
        image_size=first_size,
        max_age=bbox_max_age,
    )

    idxs = frame_indices(reader.num_frame, frame_start, frame_end, frame_step, max_frames)
    if not idxs:
        raise ValueError(f"no frames selected (video has {reader.num_frame} frames)")

    writer = None
    if overlay:
        writer = video_mod.VideoWriter(os.path.join(out_dir, "overlay.mp4"), fps=overlay_fps)

    records: List[Dict] = []
    skipped: List[Dict] = []
    t_start = time.time()

    try:
        for n_done, frame_id in enumerate(idxs):
            frames = reader.read(frame_id)
            boxes = {name: tracker.get(name, frame_id, frames[name]) for name in frames}

            try:
                packet = prepare_views(
                    frames=frames,
                    bboxes=boxes,
                    rig=rig,
                    hand_side=hand_side,
                    out_res=out_res,
                    bbox_expand=bbox_expand,
                    bbox_mindim=bbox_mindim,
                    frame_id=frame_id,
                    min_views=min_views,
                    master=master,
                )
            except PackError as exc:
                skipped.append({"frame_id": frame_id, "reason": str(exc)})
                continue

            record: Dict = {
                "frame_id": frame_id,
                "views": list(packet.names),
                "bbox": {name: np.asarray(boxes[name], dtype=float).tolist() for name in packet.names},
            }

            if dry_run:
                # No network: still exercise the crop/intrinsic bookkeeping by
                # checking that projecting a probe point with the cropped K
                # equals cropping the full-frame projection of the same point.
                record["crop_consistency_px"] = _crop_consistency(packet, rig)
                records.append(record)
            else:
                out = predictor.predict_packet(packet)
                record["joints_world"] = out["joints_world"]
                record["joints_master"] = out["joints_master"]
                if save_verts:
                    record["verts_world"] = out["verts_world"]
                # 2D/3D agreement: the heatmap keypoints (mapped back to the
                # original frames) vs. the reprojected 3D output. High values
                # mean bad calibration, unsynchronised views, or a hand the
                # model could not resolve -- the only no-ground-truth signal
                # available, so it is stored per frame.
                record["reproj_px"] = _uv_vs_reproj_px(out, packet, rig)
                records.append(record)

                if writer is not None:
                    writer.write_bgr(_overlay_frame(frames, packet, rig, out, overlay_scale))

            if progress and (n_done % 50 == 0 or n_done == len(idxs) - 1):
                elapsed = time.time() - t_start
                # progress goes to stderr so stdout stays a clean JSON report
                print(f"[poemkit] {n_done + 1}/{len(idxs)} frames  kept={len(records)}  "
                      f"skipped={len(skipped)}  {elapsed:.1f}s", file=sys.stderr, flush=True)
    finally:
        tracker.close()
        reader.close()
        if writer is not None:
            writer.close()

    report = {
        "sequence": os.path.basename(os.path.normpath(out_dir)),
        "videos": video_paths,
        "cameras": rig.names,
        "master": master or rig.names[0],
        "hand_side": hand_side,
        "bbox_backend": bbox_backend,
        "frames_selected": len(idxs),
        "frames_predicted": len(records),
        "frames_skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "bbox_stats": tracker.stats,
        "warnings": warnings,
        "dry_run": bool(dry_run),
        "seconds": round(time.time() - t_start, 2),
    }
    if records and not dry_run:
        reproj = np.array([r["reproj_px"] for r in records], dtype=float)
        report["reproj_px_mean"] = float(reproj.mean())
        report["reproj_px_p95"] = float(np.percentile(reproj, 95))
    if records and dry_run:
        cons = np.array([r["crop_consistency_px"] for r in records], dtype=float)
        report["crop_consistency_px_max"] = float(cons.max())

    _write_outputs(out_dir, records, report, save_verts=save_verts, dry_run=dry_run)
    return report


def _uv_vs_reproj_px(out: Dict, packet, rig) -> float:
    """Mean pixel distance between predicted 2D keypoints and reprojected 3D joints."""
    if "joints_uv" not in out:
        return float("nan")
    uv_orig = joints_uv_to_original(packet, out["joints_uv"], rig)
    proj = project_to_views(out["joints_world"], rig, packet.names)
    dists = [np.linalg.norm(uv_orig[name] - proj[name], axis=-1) for name in packet.names]
    return float(np.mean(np.stack(dists, axis=0)))


def _crop_consistency(packet, rig) -> float:
    """Max pixel disagreement between "project with cropped K" and "crop the projection".

    A probe point is placed 0.5 m in front of each camera; if the crop
    intrinsics were built wrongly the two paths diverge and the network would
    be fed coordinates inconsistent with its cameras.
    """
    worst = 0.0
    for i, name in enumerate(packet.names):
        # mirroring the frame about the principal point leaves K unchanged, so
        # this check is valid for the left-hand path too
        K_full = rig[name].K
        probe_cam = np.array([[0.02, -0.03, 0.5], [-0.05, 0.04, 0.7]], dtype=np.float64)
        uv_full = geo.project_points(K_full, probe_cam)
        affine = geo.affine_crop_transform(packet.centers[i], packet.scales[i], packet.crops.shape[1:3][::-1])
        uv_crop_expected = geo.apply_affine_2d(affine, uv_full)
        uv_crop_actual = geo.project_points(packet.K_crop[i], probe_cam)
        worst = max(worst, float(np.abs(uv_crop_expected - uv_crop_actual).max()))
    return worst


def _overlay_frame(frames, packet, rig, out, scale):
    import cv2

    from tool.poemkit.render import draw_hand_2d, tile_views

    tiles = []
    proj = project_to_views(out["joints_world"], rig, packet.names)
    for name in packet.names:
        img = cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR)
        tiles.append(draw_hand_2d(img, proj[name]))
    return tile_views(tiles, scale=scale, labels=packet.names)


def _write_outputs(out_dir: str, records: List[Dict], report: Dict, save_verts: bool, dry_run: bool):
    with open(os.path.join(out_dir, "report.json"), "w") as ofs:
        json.dump(report, ofs, indent=2, default=str)

    if dry_run or not records:
        return

    frame_ids = np.array([r["frame_id"] for r in records], dtype=np.int64)
    joints_world = np.stack([r["joints_world"] for r in records], axis=0).astype(np.float32)
    joints_master = np.stack([r["joints_master"] for r in records], axis=0).astype(np.float32)
    payload = {
        "frame_ids": frame_ids,
        "joints_world": joints_world,     # (F, 21, 3) meters, rig world frame
        "joints_master": joints_master,   # (F, 21, 3) meters, master camera frame
        "reproj_px": np.array([r["reproj_px"] for r in records], dtype=np.float32),
        "views": np.array([",".join(r["views"]) for r in records]),
        "cameras": np.array(report["cameras"]),
        "hand_side": np.array(report["hand_side"]),
    }
    if save_verts:
        payload["verts_world"] = np.stack([r["verts_world"] for r in records], axis=0).astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, "keypoints.npz"), **payload)

    # A plain-json copy of the joints, for tooling that would rather not read npz.
    with open(os.path.join(out_dir, "keypoints.json"), "w") as ofs:
        json.dump(
            {
                "hand_side": report["hand_side"],
                "cameras": report["cameras"],
                "units": "meters",
                "frame": "world (as defined by the supplied extrinsics)",
                "frames": [{
                    "frame_id": int(r["frame_id"]),
                    "views": r["views"],
                    "joints_world": np.asarray(r["joints_world"]).round(6).tolist(),
                } for r in records],
            },
            ofs,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="POEM-v2 multi-view hand keypoints from video only",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--calib", type=str, default=None,
                     help="calib.json | OpenCV stereo .yml | example-data dir with cam_intr/ cam_extr/")
    src.add_argument("--calib-rectified", action="store_true",
                     help="for an OpenCV yml: read the rectified P1/P2 instead of K1,K2,R,T")
    src.add_argument("--stereo-params", nargs=5, metavar=("FX", "FY", "CX", "CY", "BASELINE_M"), default=None,
                     help="build a rectified stereo rig inline instead of passing --calib")
    src.add_argument("--stereo-names", nargs=2, default=["left", "right"], help="camera names for --stereo-params")
    src.add_argument("--video", action="append", default=[], metavar="NAME=PATH",
                     help="one video per view; repeat the flag. NAME must match the calibration")
    src.add_argument("--video-dir", type=str, default=None,
                     help="directory holding <camera_name>.<ext> for each camera in the calibration")
    src.add_argument("--video-backend", choices=["auto", "cv2", "ffmpeg"], default="auto")

    hand = p.add_argument_group("hand / boxes")
    hand.add_argument("--hand-side", choices=["rh", "lh"], default="rh",
                      help="left hands are handled by mirroring images and extrinsics")
    hand.add_argument("--hand-side-json", type=str, default=None,
                      help="json {sequence_name: 'left'|'right'}, as in the released example data")
    hand.add_argument("--bbox-backend", choices=["mediapipe", "npy", "json", "full"], default="mediapipe")
    hand.add_argument("--bbox-root", type=str, default=None, help="for --bbox-backend npy: <root>/<cam>/bbox/%%05d.npy")
    hand.add_argument("--bbox-json", type=str, default=None, help="for --bbox-backend json")
    hand.add_argument("--bbox-max-age", type=int, default=5, help="frames a stale box may be reused (0 disables)")
    hand.add_argument("--bbox-expand", type=float, default=2.0)
    hand.add_argument("--bbox-mindim", type=float, default=200.0)

    mdl = p.add_argument_group("model")
    mdl.add_argument("--cfg", "-c", type=str, default=os.path.join("config", "release", "eval_single.yaml"))
    mdl.add_argument("--model", type=str, default="medium", help="small|medium|large|huge|medium_MANO")
    mdl.add_argument("--reload", type=str, default=None, help="path to checkpoints/<model>.pth.tar")
    mdl.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:N|mps")
    mdl.add_argument("--backbone-pretrained", type=str, default=None,
                     help="HRNet ImageNet weights; ignored (blanked) when --reload is given")
    mdl.add_argument("--allow-random-weights", action="store_true",
                     help="run without --reload (plumbing checks only; output is meaningless)")

    rng = p.add_argument_group("range / output")
    rng.add_argument("--frame-start", type=int, default=0)
    rng.add_argument("--frame-end", type=int, default=-1)
    rng.add_argument("--frame-step", type=int, default=1)
    rng.add_argument("--max-frames", type=int, default=None)
    rng.add_argument("--min-views", type=int, default=2, help="frames with fewer boxed views are skipped")
    rng.add_argument("--master", type=str, default=None, help="camera to use as the master (default: first in calib)")
    rng.add_argument("--out", "-o", type=str, required=True, help="output directory")
    rng.add_argument("--save-verts", action="store_true", help="also store the 778 mesh vertices per frame")
    rng.add_argument("--overlay", action="store_true", help="write overlay.mp4 with reprojected joints")
    rng.add_argument("--overlay-scale", type=float, default=0.5)
    rng.add_argument("--overlay-fps", type=float, default=30.0)

    p.add_argument("--dry-run", action="store_true",
                   help="validate calibration/videos/boxes/crops without building the model")
    p.add_argument("--env-report", action="store_true", help="print what is installed and exit")
    p.add_argument("--strict-calib", action="store_true", help="abort instead of warning on calibration problems")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.env_report:
        from tool.poemkit.runner import environment_report

        print(json.dumps(environment_report(), indent=2))
        return 0

    rig = load_rig(args)
    problems = rig.sanity_check(strict=args.strict_calib)
    for problem in problems:
        print(f"[calib warning] {problem}", file=sys.stderr)

    video_paths = parse_video_args(args.video, args.video_dir, rig.names)
    if len(video_paths) < 2:
        print(f"[error] resolved {len(video_paths)} video(s): {list(video_paths)}. POEM-v2 needs >= 2 "
              "synchronised, calibrated views; mono video is not supported (see plan_of_action.md).",
              file=sys.stderr)
        return 2

    sequence_name = os.path.basename(os.path.normpath(args.out))
    hand_side = resolve_hand_side(args, sequence_name)

    predictor = None
    if not args.dry_run:
        if args.reload is None and not args.allow_random_weights:
            print("[error] --reload is required (or pass --allow-random-weights for a plumbing-only run)",
                  file=sys.stderr)
            return 2
        from tool.poemkit.runner import PoemRunner

        predictor = PoemRunner(
            cfg_path=args.cfg,
            checkpoint=args.reload,
            model_size=args.model,
            device=args.device,
            backbone_pretrained=args.backbone_pretrained,
        )

    report = run_sequence(
        rig=rig,
        video_paths=video_paths,
        out_dir=args.out,
        hand_side=hand_side,
        bbox_backend=args.bbox_backend,
        bbox_root=args.bbox_root,
        bbox_json=args.bbox_json,
        bbox_max_age=args.bbox_max_age,
        predictor=predictor,
        dry_run=args.dry_run,
        video_backend=args.video_backend,
        frame_start=args.frame_start,
        frame_end=None if args.frame_end < 0 else args.frame_end,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        min_views=args.min_views,
        master=args.master,
        bbox_expand=args.bbox_expand,
        bbox_mindim=args.bbox_mindim,
        save_verts=args.save_verts,
        overlay=args.overlay,
        overlay_scale=args.overlay_scale,
        overlay_fps=args.overlay_fps,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["frames_predicted"] or report["dry_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
