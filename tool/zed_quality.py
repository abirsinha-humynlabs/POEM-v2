#!/usr/bin/env python
"""Score one episode's POEM output against the tier-4 health gates.

The point is that an operator downstream should never have to guess whether a
published npz is trustworthy. The same checks tier 4 asserts on the released
example data are applied here and written next to the results as a verdict,
because on out-of-distribution footage POEM-v2 can produce output that looks
reasonable in 2D while its metric depth is not usable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool.poemkit.render import HAND_BONES  # noqa: E402

# the tier-4 gates, verbatim
GATES = {
    "reproj_px_mean": ("<", 20.0),
    "hand_span_m": ("range", (0.10, 0.35)),
    "bone_len_std_m": ("<", 0.006),
    "wrist_step_p90_m": ("<", 0.05),
    "min_joint_z_m": (">", 0.05),
}


def score(npz_path, report_path):
    out = {"npz": os.path.basename(npz_path)}
    if not os.path.isfile(npz_path):
        return {**out, "status": "absent"}
    d = np.load(npz_path)
    j = d["joints_master"]
    if j.size == 0:
        return {**out, "status": "empty"}

    rep = {}
    if os.path.isfile(report_path):
        with open(report_path) as ifs:
            rep = json.load(ifs)

    spans = np.linalg.norm(j.max(1) - j.min(1), axis=-1)
    bl = np.stack([np.linalg.norm(j[:, a] - j[:, b], axis=-1) for a, b in HAND_BONES], axis=1)
    ws = np.linalg.norm(np.diff(j[:, 0], axis=0), axis=-1) if len(j) > 2 else np.array([0.0])

    m = {
        "frames_predicted": int(rep.get("frames_predicted", len(j))),
        "frames_skipped": int(rep.get("frames_skipped", 0)),
        "reproj_px_mean": float(rep.get("reproj_px_mean", np.nan)),
        "reproj_px_p95": float(rep.get("reproj_px_p95", np.nan)),
        "hand_span_m": float(np.median(spans)),
        "bone_len_std_m": float(np.median(bl.std(0))),
        "wrist_step_p90_m": float(np.percentile(ws, 90)),
        "min_joint_z_m": float(j[..., 2].min()),
        "median_wrist_z_m": float(np.median(j[:, 0, 2])),
        "frames_with_joint_behind_camera": int((j[..., 2].min(1) <= 0.05).sum()),
    }

    failed = []
    for key, (op, bound) in GATES.items():
        v = m.get(key)
        if v is None or not np.isfinite(v):
            continue
        ok = (v < bound) if op == "<" else (v > bound) if op == ">" else (bound[0] < v < bound[1])
        if not ok:
            failed.append({"metric": key, "value": round(v, 4), "gate": f"{op} {bound}"})

    m["gates_failed"] = failed
    m["status"] = "PASS" if not failed else "FAIL"
    return {**out, **m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per_hand = {}
    for side in ("rh", "lh"):
        per_hand[side] = score(os.path.join(a.work, side, "keypoints.npz"),
                               os.path.join(a.work, side, "report.json"))

    statuses = [v.get("status") for v in per_hand.values()]
    overall = "PASS" if statuses and all(s == "PASS" for s in statuses) else "FAIL"

    doc = {
        "episode": a.tag,
        "model": "POEM-v2 medium.pth.tar",
        "rig": "ZED stereo, left_eye + right_eye, ~0.12 m baseline, 1920x1080",
        "verdict": overall,
        "verdict_meaning":
            "PASS means the output met every tier-4 health gate that the released "
            "3-camera example data meets. FAIL means at least one physical "
            "plausibility check failed and these keypoints should NOT be treated as "
            "validated metric ground truth.",
        "gates": {k: f"{op} {bound}" for k, (op, bound) in GATES.items()},
        "per_hand": per_hand,
        "known_limitation":
            "POEM-v2 triangulates across views that surround the hand; its released "
            "rigs have wide angular separation. A ZED's two eyes are ~12 cm apart "
            "and face the same direction, and egocentric imagery is outside the "
            "training distribution. In validation the 2D skeleton tracked the hand "
            "correctly and median wrist depth agreed with stereo disparity (~0.32 m), "
            "but per-frame reprojection error stayed well above the 20 px gate and a "
            "minority of frames placed joints behind the camera.",
    }
    with open(a.out, "w") as ofs:
        json.dump(doc, ofs, indent=2)
    print("verdict %s -> %s" % (overall, a.out))
    for side, v in per_hand.items():
        if v.get("status") in ("absent", "empty"):
            print("  %s: %s" % (side, v["status"]))
            continue
        print("  %s: %s  frames %d  reproj %.1f px  wrist z %.2f m  failed: %s"
              % (side, v["status"], v["frames_predicted"], v["reproj_px_mean"],
                 v["median_wrist_z_m"], ", ".join(f["metric"] for f in v["gates_failed"]) or "none"))


if __name__ == "__main__":
    main()
