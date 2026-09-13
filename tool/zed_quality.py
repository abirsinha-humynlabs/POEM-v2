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


def score_flat(path, label):
    """Score a renderer-layout npz (frame_idx/kp2d/kp3d_cam/hand/kept) per hand.

    Used for the converted and stabilised files, where rows are per detection
    rather than per frame, so the wrist-step check must only compare rows that
    are one video frame apart.
    """
    if not path or not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    fi = d["frame_idx"].astype(int)
    k3 = d["kp3d_cam"].astype(float)
    hnd = d["hand"].astype(int) if "hand" in d.files else np.zeros(len(fi), int)
    keep = d["kept"].astype(bool) if "kept" in d.files else np.ones(len(fi), bool)
    fi, k3, hnd = fi[keep], k3[keep], hnd[keep]

    out = {"label": label, "rows": int(len(fi))}
    for side, name in ((1, "rh"), (0, "lh")):
        m = hnd == side
        if m.sum() < 3:
            continue
        j, f = k3[m], fi[m]
        o = np.argsort(f, kind="stable")
        j, f = j[o], f[o]
        bl = np.stack([np.linalg.norm(j[:, a] - j[:, b], axis=-1) for a, b in HAND_BONES], axis=1)
        step = np.linalg.norm(np.diff(j[:, 0], axis=0), axis=-1)[np.diff(f) == 1]
        m_ = {
            "rows": int(len(j)),
            "hand_span_m": float(np.median(np.linalg.norm(j.max(1) - j.min(1), axis=-1))),
            "bone_len_std_m": float(np.median(bl.std(0))),
            "wrist_step_p90_m": float(np.percentile(step, 90)) if len(step) else float("nan"),
            "min_joint_z_m": float(j[..., 2].min()),
        }
        failed = []
        for key, (op, bound) in GATES.items():
            v = m_.get(key)
            if v is None or not np.isfinite(v):
                continue
            ok = (v < bound) if op == "<" else (v > bound) if op == ">" else (bound[0] < v < bound[1])
            if not ok:
                failed.append({"metric": key, "value": round(v, 4), "gate": f"{op} {bound}"})
        m_["gates_failed"] = failed
        m_["status"] = "PASS" if not failed else "FAIL"
        out[name] = m_
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None, help="converted (unstabilised) renderer-layout npz")
    ap.add_argument("--stabilised", default=None, help="smoothed + rigidified npz")
    a = ap.parse_args()

    per_hand = {}
    for side in ("rh", "lh"):
        per_hand[side] = score(os.path.join(a.work, side, "keypoints.npz"),
                               os.path.join(a.work, side, "report.json"))

    statuses = [v.get("status") for v in per_hand.values()]
    overall = "PASS" if statuses and all(s == "PASS" for s in statuses) else "FAIL"

    base = score_flat(a.baseline, "converted (POEM raw)")
    stab = score_flat(a.stabilised, "temporal_smooth + rigidify")
    stab_status = None
    if stab:
        sides = [stab[k]["status"] for k in ("rh", "lh") if k in stab]
        stab_status = "PASS" if sides and all(x == "PASS" for x in sides) else "FAIL"

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
        "stabilised": {
            "verdict": stab_status,
            "pipeline": "egocentric-hand-stabilisation: temporal_smooth.py (zero-phase, "
                        "gap-aware, Tukey-robust) then rigidify.py (one canonical bone-length "
                        "template per side, pose-smoothed)",
            "before": base,
            "after": stab,
            "caveat": "Stabilisation makes the hand rigid, temporally smooth and "
                      "self-consistent, which is what the physical gates measure. It cannot "
                      "recover pose accuracy the 12 cm baseline never captured, so the raw "
                      "model's 2D-vs-3D reprojection disagreement above is still the honest "
                      "read on absolute accuracy.",
        } if stab else None,
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
    print("verdict raw=%s stabilised=%s -> %s" % (overall, stab_status, a.out))
    for tag, doc_ in (("before", base), ("after", stab)):
        if not doc_:
            continue
        for side in ("rh", "lh"):
            if side in doc_:
                v = doc_[side]
                print("  %-6s %s  %s  n=%d  span %.3f m  bone-std %.2f mm  wrist-p90 %.1f mm"
                      % (tag, side, v["status"], v["rows"], v["hand_span_m"],
                         v["bone_len_std_m"] * 1000, v["wrist_step_p90_m"] * 1000))
    for side, v in per_hand.items():
        if v.get("status") in ("absent", "empty"):
            print("  %s: %s" % (side, v["status"]))
            continue
        print("  %s: %s  frames %d  reproj %.1f px  wrist z %.2f m  failed: %s"
              % (side, v["status"], v["frames_predicted"], v["reproj_px_mean"],
                 v["median_wrist_z_m"], ", ".join(f["metric"] for f in v["gates_failed"]) or "none"))


if __name__ == "__main__":
    main()
