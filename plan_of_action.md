# Plan of action — POEM-v2 hand keypoints from video only

Working notes for picking this up later (or on another machine). Status as of
**2026-09-13**, branch `feature/video-only-inference`. Tiers 1-4 have now all been
executed on a Linux + NVIDIA A10G box; the numbers below are measured, not
estimated.

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
  kernels. The pipeline was written on a laptop with neither CUDA nor pytorch3d
  — hence the tiered test design below, which is what let tiers 1-2 be proven
  before a GPU box existed.

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
| `docs/pipeline_design.md` | how the pipeline is built and why: architecture, geometry, design decisions, test strategy |

Differences from `tool/infer_hand.py`: any number/naming of views, calibration
from several sources, pluggable detectors, no GUI, device flag, results written
to disk, and a torch-free `--dry-run` validation mode.

## 3. Test tiers and current results

`scripts/testing/run_test_matrix.sh` (skips are never failures):

| tier | scope | needs | status |
| --- | --- | --- | --- |
| 1 | calibration, geometry, boxes, crops, video IO, dry-run pipeline, CLI | numpy + opencv + pytest | **91 passed, 0 skipped** (A10G box, 10.3 s) |
| 2 | torch batch contract, DLT parity with the repo's implementation | CPU torch | **7 passed** (4.6 s) |
| 3 | model build + forward passes | pytorch3d, manotorch, `assets/mano_v1_2` | **17 passed, 0 skipped** with `medium.pth.tar` (70 s) |
| 4 | accuracy on the released example data | + `POEM_CHECKPOINT`, `POEM_EXAMPLE_DATA` | **6 passed, 0 skipped** (37.6 s) |

All four tiers now pass with no skips: **121 of 121 cases**. Measured on
Linux + NVIDIA A10G (23 GB), Python 3.8.0, torch 1.11.0+cu113, pytorch3d 0.7.2,
manotorch 0.0.2, mediapipe 0.10.9, checkpoint `medium.pth.tar`.

Tier 1's previously-skipped case now runs (it needed mediapipe). Tier 3 without
a checkpoint is 15 passed / 2 skipped; the 2 are the weight-load and
metric-plausibility checks, which pass once `POEM_CHECKPOINT` is set.

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

## 4. Done — with the measured numbers

All of A-E below have been executed on a Linux + NVIDIA A10G box
(2026-09-13). Commands are kept so they can be re-run.

### A. Environment — built and verified ✅
```bash
python -m tool.infer_video --env-report --out unused
# missing: []   mano_assets: true   cuda: true   devices: [cpu, cuda:0]
```
Python 3.8.0, torch 1.11.0+cu113, torchvision 0.12.0+cu113, pytorch3d 0.7.2,
manotorch 0.0.2, numpy 1.23.5, opencv 4.5.3, mediapipe 0.10.9, on an
NVIDIA A10G (23 GB).

Four deviations from `docs/installation.md` were needed; none touches the
inference path, and the pins in the doc are otherwise honoured:

1. **conda-forge only.** `conda env create -f environment.yml` pulls the
   `defaults` channel, which now refuses to install without an interactive
   Anaconda Terms-of-Service acceptance. The env is built with the identical
   package pins from conda-forge instead
   (`conda create -n POEM --override-channels -c conda-forge python==3.8
   "setuptools~=58.5" numpy==1.23.5 pip ipython pyembree`).
2. **`opendr` omitted.** It fails to compile (`gcc`/`ld` error) and its failure
   aborts the whole `pip install -r requirements.txt`. It is a legacy SMPL
   renderer and is not imported anywhere on the inference path.
3. **pytorch3d needs its deps from PyPI.** `pip install --no-index -f <wheel
   url>` cannot resolve `fvcore`, so install `fvcore~=0.1.5 iopath` first and
   then the wheel with `--no-deps`.
4. **mediapipe pinned to 0.10.9.** `mediapipe>=0.10.9` resolves to 1.x, which
   uses PEP-585 generics (`list[Category]`) and cannot be imported on Python
   3.8; 0.10.11 additionally wants a `jaxlib` that has no 3.8 wheel.

`sh prepare/download_hrnet.sh` was not needed: `--reload` supplies all weights
and `build_cfg()` blanks `BACKBONE.PRETRAINED`.

### B/C. Tier 3 — 17 passed, 0 skipped ✅
```bash
export POEM_CHECKPOINT=$PWD/checkpoints/medium.pth.tar
scripts/testing/run_test_matrix.sh tier3        # 17 passed in 70 s
```
Without a checkpoint: 15 passed, 2 skipped (the skips are exactly the
weight-load and metric-plausibility cases). With `medium.pth.tar` the strict
load reports `Loading SUCCEEDED` — every checkpoint tensor lands in the model
(373.5 M parameters). Both mono guards pass, including
`test_single_view_batch_needs_ground_truth`.

Checkpoints came from the README's Drive folder via `gdown --folder`, which
worked headlessly; all four (`small`, `medium`, `medium_MANO`, `large`) are in
`./checkpoints/` (gitignored).

### D. Tier 4 — 6 passed, 0 skipped ✅
```bash
export POEM_EXAMPLE_DATA=/path/to/example_data
export POEM_CHECKPOINT=$PWD/checkpoints/medium.pth.tar
scripts/testing/run_test_matrix.sh tier4        # 6 passed in 37.6 s
```

Two test-side fixes were needed before this tier could run at all — neither
relaxes a threshold:

* the release now ships its sequences under **`data_v2/`**, not `data/`, so the
  layout fixture accepts either;
* the released clips **do not start with the hand in shot**. The mask pipeline
  writes a file per frame but stores an empty array when it found nothing, so
  `pour__2025_0325_1115_55` has < 2 boxed views until frame 11 and
  `pour__2025_0325_1117_17` until frame 21. The tests sampled from frame 0 and
  so measured the dead zone (0-9 usable frames). They now begin at the first
  frame whose shipped boxes cover ≥ 2 views (`_first_usable_frame`), which
  makes them *stricter* — more real frames are graded, against the same gates.

Measured on `pour__2025_0325_1115_55`, 3 views, 100 frames from frame 11,
`medium.pth.tar`:

| metric | measured | gate | headroom |
| --- | --- | --- | --- |
| `reproj_px_mean` | **11.1 px** (p95 15.4) | < 20 px | 1.8× |
| median hand span | **0.178 m** | 0.10-0.35 m | comfortably inside |
| bone-length std over time | **1.53 mm** | < 6 mm | 3.9× |
| wrist step, p90 | **7.8 mm/frame** | < 50 mm | 6.4× |
| joint depth z | 0.57-0.81 m | trained 0-1.2 m | inside |
| frames predicted | 100/100, 0 skipped | — | — |

**2-view vs N-view (the stereo-rig question).** Over the same 100-frame window,
camera_1+camera_2 against all three cameras:

* **MPJPE 9.2 mm** (median 6.4, p95 27.8, max 52.7 mm)
* the tier-4 test's own shorter 10-frame window prints **13.4 mm**
* both runs are metric and stable; the 2-view run actually has a *lower*
  `reproj_px_mean` (8.5 px vs 11.1 px), since reprojection error is averaged
  over fewer, better-boxed views — it is an internal-consistency measure, not
  an accuracy measure, and should not be read as "2 views are more accurate".

**A stereo rig costs about 1 cm of joint accuracy** relative to a 3-camera rig
on this footage. That is usable for most manipulation work; it is not
negligible if you need sub-centimetre fingertips.

**MediaPipe vs the shipped boxes: median IoU 0.60** over 12 view-frames
(gate > 0.3). Good enough that the detector-driven, video-only path is a real
replacement for the released mask pipeline — but 0.60 is agreement, not
accuracy, and crop quality gates everything downstream, so check the overlay on
your own footage.

### E. Synthetic smoke with real weights — 3/3 cases ✅
```bash
scripts/testing/run_synthetic_smoke.sh /tmp/poem_smoke checkpoints/medium.pth.tar medium
```
Stereo, 3-view-with-dropouts and left-hand all completed 10/10 frames, finite
joints, `overlay.mp4` written, exit 0. `reproj_px_mean` is 36.8 / 51.7 / 40.0 px
— high by design: the imagery is cartoon hands well off the training
distribution, so this tier proves plumbing, not accuracy. Dry-run crop
consistency is ~7e-14 px.

### F. Throughput on real hardware — measured ✅
`pour__2025_0325_1115_55`, `medium.pth.tar`, A10G, 100 frames from frame 11:

| views | frames | seconds | **s/frame** | fps | peak GPU mem |
| --- | --- | --- | --- | --- | --- |
| 3 (all) | 100/100 | 12.59 | **0.126** | 7.9 | 2709 MiB |
| 2 (stereo) | 97/100 | 10.36 | **0.107** | 9.4 | 2709 MiB |

So roughly **8-9 fps**, i.e. a 30 fps capture costs ~3.5-4× realtime, and
GPU memory is a non-issue (< 3 GB of 23 GB) — batching frames would likely help
more than a bigger GPU. Cost scales weakly with view count: +1 view is ~+18 %
wall-clock, because per-view backbone work is a minority of the budget.

### G. Then, on your own capture
1. `python -m tool.make_calib stereo ... --out calib.json` (or `opencv` from
   your rectification work), then `make_calib check`.
2. `python -m tool.infer_video --dry-run` — fix every calibration warning
   before spending GPU time.
3. Full run with `--bbox-backend mediapipe --overlay`, then look at
   `report.json`'s `reproj_px_mean` and the overlay before trusting the numbers.

## 5. Open questions / risks

Resolved by measurement (2026-09-13):

* **~~Stereo accuracy is unquantified.~~** Measured: **9.2 mm MPJPE** between a
  2-view and a 3-view rig over 100 frames (13.4 mm on the test's shorter
  window), both metric and temporally stable. A two-camera rig is viable;
  budget ~1 cm of joint error against a three-camera one.
* **~~Throughput is unmeasured.~~** Measured on an A10G: **0.126 s/frame at 3
  views, 0.107 s/frame at 2 views** (7.9 / 9.4 fps), < 3 GB GPU memory.
* **~~Left-hand path verified only synthetically.~~** Now also passes on the
  released left-hand sequence (`pour__2025_0325_1117_17`) on real imagery.
* **~~Detector quality is unproven.~~** Partially resolved: MediaPipe agrees
  with the shipped boxes at **median IoU 0.60**, above the 0.3 gate. Still a
  live risk on *your* footage — see below.

Still open:

* **Detector quality gates everything.** Crops come from the bbox, so a loose or
  jittery detector degrades 3D with no error raised. IoU 0.60 against the
  released masks is agreement on easy, well-lit, hand-centred frames; a
  hand-specific detector may beat MediaPipe on real captures. Always watch the
  overlay.
* **Working volume.** `POSITION_RANGE` is x,y ∈ [-0.6, 0.6] m, z ∈ [0, 1.2] m
  from the master camera; a rig framed further out is off-distribution. The
  example data sits at z ≈ 0.57-0.81 m, comfortably inside.
* **Synchronisation.** Views are aligned by frame index only. Drift biases the
  3D silently — `check_alignment()` catches length/resolution/rate mismatches but
  cannot detect a constant temporal offset. Hardware-sync or verify externally.
* **Two hands.** One hand per run; a two-hand capture means two passes with
  per-hand boxes (`--bbox-backend json`).
* **A clip may not open with the hand in shot.** The released example data does
  not, and neither will most real captures. Frames with < 2 boxed views are
  skipped (correctly), but a run whose window lands entirely in a dead zone
  returns nothing rather than erroring — check `frames_predicted` and
  `bbox_stats` in `report.json`, not just the exit code.

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

Details: `docs/video_inference.md` (usage), `docs/pipeline_design.md` (design).
