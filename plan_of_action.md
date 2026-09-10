# Plan of action — POEM-v2 hand keypoints from video only

Working notes for picking this up later (or on another machine). Status as of
**2026-09-10**, branch `feature/video-only-inference`.

**Goal:** 3D hand keypoints from RGB video alone — no depth, no IMU, no ground
truth — with the code ready to run the moment a GPU box is available.

---

## 1. Answers to the two questions that started this

### Can hand keypoints come from video only?

**Yes, from ≥ 2 calibrated synchronised views. No, from mono.**

* Output is 21 joints + 778 MANO vertices per frame, in **metric meters**,
  referenced to the first ("master") camera — `pred_joints_3d` /
  `pred_verts_3d` in `lib/models/POEM.py`.
* Nothing in the pipeline reads depth, IMU or point clouds. Inputs are RGB
  crops, `K` and `T_cw` per view.
* **Stereo (2 views) is the shipped minimum** — the release config's own
  `VIEW_RANGE` is `[2, 2]`. Rectified pairs are fine; rectification itself is
  not used, only calibration.
* **Mono is a dead end.** `lib/models/POEM.py:276-305`: with 2+ views the model
  triangulates its own predicted 2D keypoints (DLT) for the transformer's
  reference joints; with one view per sample it reads
  `batch["master_joints_3d"]`, i.e. ground-truth 3D. The paper's single-view
  numbers are GT-seeded and Procrustes-aligned. Making mono work would mean
  replacing that seed with a monocular 3D estimator — a change to the model, not
  a config flag.
* Two things video does not provide and you must supply: a **per-view hand
  bbox** each frame (POEM-v2 is not a detector) and the **hand side**.

### Linux / GPU needed?

**Practically yes: Linux + NVIDIA GPU.**

* The released scripts hardcode `cuda:0` / `torch.cuda.set_device`; install is
  pinned to Python 3.8 + torch 1.11.0+cu113 + the **cu113 pytorch3d wheel**
  (`docs/installation.md`), which is Linux-only in practice.
* The model itself is device-agnostic (`x.device` throughout), and the CUDA-only
  `neural_renderer` is *not* on the inference path — so `--device cpu` runs, at
  seconds per frame. Good for plumbing checks, useless for video.
* macOS/MPS is not a path: pytorch3d's `knn_points`/`ball_query` have no Metal
  kernels. The current laptop has neither CUDA nor pytorch3d — hence the tiered
  test design below.

## 2. What is now in the repo

New, all additive — no existing file was modified:

| path | role |
| --- | --- |
| `tool/poemkit/calib.py` | `Camera`/`CameraRig`; calib.json, OpenCV yml, released pickles, rectified-stereo builder; metric sanity checks |
| `tool/poemkit/geometry.py` | crop affines, crop intrinsics, projection, mirroring, numpy DLT (numpy twins of the pytorch3d-importing repo utils) |
| `tool/poemkit/bbox.py` | bbox providers (mediapipe / npy / json / full) + last-good-box tracker |
| `tool/poemkit/video.py` | cv2 and ffmpeg frame readers, multi-view synchronised reader, mp4 writer |
| `tool/poemkit/views.py` | `prepare_views` → `ViewPacket` → `packet_to_batch`; master bookkeeping, left-hand mirroring, ≥2-view contract |
| `tool/poemkit/runner.py` | cfg overrides (model size, checkpoint), device resolution, `PoemRunner`, `environment_report` |
| `tool/poemkit/render.py` | cv2-only skeleton overlay (no matplotlib/open3d/GUI) |
| `tool/infer_video.py` | the CLI: N views, headless, npz/json export, `--dry-run`, `--env-report` |
| `tool/make_calib.py` | build/inspect `calib.json` (stereo, OpenCV, released pickles, synthetic ring) |
| `scripts/testing/make_synthetic_capture.py` | renders a multi-view capture with exact ground truth |
| `scripts/testing/run_test_matrix.sh` | tiered test runner |
| `scripts/testing/run_synthetic_smoke.sh` | end-to-end smoke: dry runs, then real inference if a checkpoint is given |
| `tests/` | 121 test cases, tiered by what the machine can support |
| `docs/video_inference.md` | usage, calibration formats, metrics, hardware |

Differences from `tool/infer_hand.py`: any number/naming of views, calibration
from several sources, pluggable detectors, no GUI, device flag, results written
to disk, and a torch-free `--dry-run` validation mode.

## 3. Test tiers and current results

`scripts/testing/run_test_matrix.sh` (skips are never failures):

| tier | scope | needs | status |
| --- | --- | --- | --- |
| 1 | calibration, geometry, boxes, crops, video IO, dry-run pipeline, CLI | numpy + opencv + pytest | **90 passed, 1 skipped** on macOS, no GPU |
| 2 | torch batch contract, DLT parity with the repo's implementation | CPU torch | **7 passed** on macOS |
| 3 | model build + forward passes | pytorch3d, manotorch, `assets/mano_v1_2` | 9 passed (cfg/env/device), **8 pending** |
| 4 | accuracy on the released example data | + `POEM_CHECKPOINT`, `POEM_EXAMPLE_DATA` | **6 pending** |

Tiers 1-2 were run and pass on this laptop (`106 passed, 15 skipped` overall).
Tiers 3-4 have never executed — that is the main open item.

What tiers 1-2 already prove, against exact synthetic ground truth:

* DLT triangulation matches the repo's `batch_triangulate_dlt_torch` to 1e-6,
  and recovers synthetic 3D to 1e-7 m;
* `project(K_crop, X) == crop_affine(project(K_full, X))` — the crop intrinsics
  the network is handed are consistent with the crop it is shown;
* `target_cam_extr` really is camera→master, master → identity, verified by
  replaying the model's own reference-joint step with the repo's DLT code;
* the left-hand mirroring round-trips: mirrored frames + mirrored extrinsics in,
  original-world coordinates out, and its 2D keypoints land on unflipped pixels;
* single-view input is refused at both the packing and runner boundaries;
* a 3-view rig losing one view still packs (2 views), losing two is skipped;
* export: `keypoints.npz` / `keypoints.json` match ground truth to 1e-4 m
  through the whole pipeline, using an oracle predictor in place of the network.

## 4. Pending — in order

### A. Build the environment on a Linux + NVIDIA box
Follow `docs/installation.md` (conda env, torch 1.11.0+cu113, cu113 pytorch3d,
manotorch, MANO assets in `assets/mano_v1_2`, `sh prepare/download_hrnet.sh`),
then `pip install -r requirements-video-infer.txt` for mediapipe/ffmpeg-python.
Verify with:

```bash
python -m tool.infer_video --env-report --out unused
```

`missing` must be empty and `mano_assets` true.

### B. Run tier 3 (model forward) — 8 cases
```bash
scripts/testing/run_test_matrix.sh tier3
```
Covers: 2-view and 3-view forwards, a 2-of-3-view (occlusion) forward, output
shapes/finiteness, determinism, and the two mono guards — including
`test_single_view_batch_needs_ground_truth`, which asserts the GT dependency of
the single-view path rather than just describing it. Weights are random here, so
values are meaningless by design; only structure is asserted.

### C. Get the checkpoints and run tier 3's checkpoint cases
Download from the README's `ckpt_release` Drive folder into `./checkpoints`, then
```bash
export POEM_CHECKPOINT=$PWD/checkpoints/medium.pth.tar
scripts/testing/run_test_matrix.sh tier3
```
Adds a strict weight-load check and a "output is metric and in front of the
camera" check.

### D. Run tier 4 on the released example data — 6 cases
Download `example_data.tar.xz` from
`https://huggingface.co/kelvin34501/POEM-v2_example_data`, extract, then
```bash
export POEM_EXAMPLE_DATA=/path/to/extracted/example_data
export POEM_CHECKPOINT=$PWD/checkpoints/medium.pth.tar
scripts/testing/run_test_matrix.sh tier4        # add POEM_TEST_FRAMES=50 for more frames
```
This is the real accuracy gate: 2D/3D agreement under 20 px, metric hand span
(0.10-0.35 m), rigid bone lengths over time, wrist continuity under 5 cm/frame,
the left-hand path on real imagery, a 2-view-vs-all-views MPJPE comparison
(prints the number — **this is the answer to "how much do I lose with stereo?"**),
and a MediaPipe-vs-shipped-boxes IoU check, which decides whether the
detector-driven video-only path can replace the released mask pipeline.

### E. Synthetic smoke with real weights
```bash
scripts/testing/run_synthetic_smoke.sh /tmp/poem_smoke checkpoints/medium.pth.tar medium
```
Exercises stereo / 3-view-with-dropouts / left-hand end to end through the CLI
and writes overlay videos. Cartoon imagery, so plumbing only.

### F. Then, on your own capture
1. `python -m tool.make_calib stereo ... --out calib.json` (or `opencv` from
   your rectification work), then `make_calib check`.
2. `python -m tool.infer_video --dry-run` — fix every calibration warning
   before spending GPU time.
3. Full run with `--bbox-backend mediapipe --overlay`, then look at
   `report.json`'s `reproj_px_mean` and the overlay before trusting the numbers.

## 5. Open questions / risks

* **Stereo accuracy is unquantified.** Training mixes 2-8 views; a short-baseline
  pair gives the weakest DLT seed. Step D's 2-view-vs-N-view MPJPE is the
  measurement — run it before committing to a two-camera rig.
* **Working volume.** `POSITION_RANGE` is x,y ∈ [-0.6, 0.6] m, z ∈ [0, 1.2] m
  from the master camera; a rig framed further out is off-distribution.
* **Detector quality gates everything.** Crops come from the bbox, so a loose or
  jittery detector degrades 3D without any error. Step D's IoU check is the
  proxy; a hand-specific detector may beat MediaPipe on your footage.
* **Synchronisation.** Views are aligned by frame index only. Drift biases the
  3D silently — `check_alignment()` catches length/resolution/rate mismatches but
  cannot detect a constant temporal offset. Hardware-sync or verify externally.
* **Two hands.** One hand per run; a two-hand capture means two passes with
  per-hand boxes (`--bbox-backend json`).
* **Throughput** on real hardware is unmeasured (HRNet + a point transformer per
  frame, per view). Measure with `--max-frames` before planning a long capture.
* **Left-hand path on real data** has only been verified synthetically so far
  (exactly, but synthetically) — step D covers it.

## 6. Quick reference

```bash
# what is installed / which devices
python -m tool.infer_video --env-report --out unused

# validate a capture without torch
python -m tool.infer_video --calib calib.json --video-dir CAP --dry-run --out /tmp/check

# real run
python -m tool.infer_video --calib calib.json --video-dir CAP --hand-side rh \
    --bbox-backend mediapipe --cfg config/release/eval_single.yaml --model medium \
    --reload checkpoints/medium.pth.tar --device cuda:0 --out exp/cap --overlay

# tests
scripts/testing/run_test_matrix.sh [tier1|tier2|tier3|tier4]
```

Details: `docs/video_inference.md`.
