# The video-only pipeline: what it is and how it was built

Design and implementation notes for the code added in
`tool/poemkit/`, `tool/infer_video.py`, `tool/make_calib.py`,
`scripts/testing/` and `tests/`.

`docs/video_inference.md` is the user manual ("which flags do I type").
`plan_of_action.md` is the status board ("what is still pending").
This document is the engineering record: what the pipeline does stage by
stage, the geometry it relies on, why each structural decision was made, and
how the tests are able to prove correctness on a laptop with no GPU.

Contents:

1. [Starting point and constraints](#1-starting-point-and-constraints)
2. [What the pipeline is](#2-what-the-pipeline-is)
3. [Architecture and import policy](#3-architecture-and-import-policy)
4. [Frame-by-frame data flow](#4-frame-by-frame-data-flow)
5. [The geometry, derived](#5-the-geometry-derived)
6. [Design decisions and why](#6-design-decisions-and-why)
7. [How the tests prove it without a GPU](#7-how-the-tests-prove-it-without-a-gpu)
8. [Failure-mode catalogue](#8-failure-mode-catalogue)
9. [Extending it](#9-extending-it)
10. [File reference](#10-file-reference)

---

## 1. Starting point and constraints

### What already existed

`tool/infer_hand.py` is the released real-world demo. It works, and everything
here follows its conventions, but it is written for exactly one capture:

| in the demo | consequence |
| --- | --- |
| `CAMERA_INFO` maps three fixed RealSense serials | other rigs need code edits |
| calibration read from `cam_intr/*.pkl`, `cam_extr/*.pkl` | no other calibration format |
| boxes read from `<mask>/<cam>/bbox/%05d.npy` | needs a pre-run mask pipeline |
| `cv2.imshow` per view + `cv2.waitKey` in the frame loop | cannot run over SSH; blocks on Enter |
| `device = torch.device("cuda:0")`, `CUDA_VISIBLE_DEVICES` set at import | no CPU path, no device choice |
| results only drawn, never written | nothing to consume downstream |
| `.mkv` via `ffmpeg-python` | requires ffmpeg |
| imports `lib.viztools.viz_o3d_utils` | pulls in open3d |

### What the model actually requires

Read off `lib/models/POEM.py` and `config/release/eval_single.yaml`:

* **RGB only.** No depth, IMU or point-cloud input exists anywhere on the
  inference path. Per view: a 256×256 crop, a 3×3 `K` and a 4×4 extrinsic.
* **Two or more views.** `_forward_impl` triangulates its own predicted 2D
  keypoints (DLT) to seed the transformer's reference joints. With one view per
  sample (`inputs_all_sv`, `lib/models/POEM.py:276-305`) it instead reads
  `batch["master_joints_3d"]` — ground truth. The release config's own
  `VIEW_RANGE` is `[2, 2]`.
* **Metric extrinsics.** Output is absolute, in meters, in the master camera's
  frame; `POSITION_RANGE` covers x,y ∈ [-0.6, 0.6] m and z ∈ [0, 1.2] m.
* **A hand box per view.** POEM-v2 is not a detector.
* **A hand side.** Trained on right hands; left hands work by mirroring images
  and extrinsics (`tool/flip_util.py`).

### Two environment facts that shaped the code

1. **`lib.utils.transform` imports `pytorch3d.transforms` at module scope**
   (`lib/utils/transform.py:10`), and `lib.viztools.draw` imports matplotlib.
   So the repo's own crop-affine and drawing helpers cannot be imported at all
   without a CUDA-flavoured environment — even though the math in them is pure
   numpy. Any laptop-testable code had to avoid those modules.
2. **The HRNet backbone loader calls `torch.load(pretrained)` with no
   `map_location`** (`lib/models/backbones/hrnet.py:433-434`), unlike
   `lib/utils/net_utils.load_weights`, which maps to CPU. On a CPU-only host
   that can fail. Since a full POEM checkpoint already contains backbone
   weights, the runner blanks `BACKBONE.PRETRAINED` whenever `--reload` is given.

## 2. What the pipeline is

One sentence: **synchronised RGB videos + calibration + per-view hand boxes →
metric 3D hand keypoints per frame, headless, with a report.**

```
  N videos            calib.json / OpenCV yml / released pickles
  (one per view)      (K and T_cw per camera, meters)
        |                        |
        v                        v
  MultiViewReader          CameraRig  --> sanity_check()  (metric? orthonormal?
  (frame-index sync)                                       co-located? sizes?)
        |                        |
        +----------+-------------+
                   v
           bbox provider (mediapipe | npy | json | full)
           wrapped in BBoxTracker (reuse last good box)
                   v
           prepare_views()   ------->  PackError if < 2 views survive
                   |                   (mono is refused, never faked)
                   v
              ViewPacket  (numpy: crops, K_crop, T_cw_used, T_mc, centers, scales, flipped)
                   |
                   v
           packet_to_batch()  (the only torch conversion)
                   v
      PtEmbedMultiviewStereoV2(..., mode="inference")
                   v
           unpack_prediction()
                   v
   joints/verts in master frame  --> un-mirror if left hand --> world frame
                   v
   keypoints.npz | keypoints.json | report.json | overlay.mp4
```

Compared with the demo: any number and naming of views, four calibration
sources, four box sources, no GUI, a device flag, results on disk, and a
`--dry-run` mode that validates a whole capture with **torch not installed**.

## 3. Architecture and import policy

The single most important structural rule: **the geometry is numpy, and torch
appears in exactly one function.**

| layer | module | imports | testable on a laptop |
| --- | --- | --- | --- |
| calibration | `tool/poemkit/calib.py` | numpy (cv2 lazily, for yml) | yes |
| geometry | `tool/poemkit/geometry.py` | numpy | yes |
| boxes | `tool/poemkit/bbox.py` | numpy (mediapipe lazily) | yes |
| video | `tool/poemkit/video.py` | numpy, cv2 / ffmpeg-python (lazy) | yes |
| view packing | `tool/poemkit/views.py` | numpy, cv2; torch **only inside** `packet_to_batch` | yes |
| overlays | `tool/poemkit/render.py` | cv2 | yes |
| model | `tool/poemkit/runner.py` | torch, pytorch3d, manotorch, MANO assets | no |
| CLI | `tool/infer_video.py` | the above; imports `runner` **only** for a real run | partly |

Consequences:

* `--dry-run` and `--env-report` work with no torch present, so a capture can be
  validated before renting a GPU.
* 106 of the 121 test cases execute on macOS with no CUDA.
* `runner.py` is imported lazily inside `main()`, so an import error in the
  heavy stack cannot break the light paths.

Because `lib.utils.transform` and `lib.viztools.draw` are unreachable without
pytorch3d/matplotlib, `geometry.py` and `render.py` re-implement the small parts
needed (`_get_affine_trans_no_rot`, `_affine_transform_post_rot` at `rot=0`, the
DLT, `plot_hand`'s bone list). **Duplication is a drift risk, so it is pinned by
tests**: `test_geometry.py::test_parity_with_repo_affine_helpers` and
`::test_parity_with_repo_triangulation` compare against the originals whenever
the full environment is importable.

## 4. Frame-by-frame data flow

What `run_sequence()` does per frame, and where each step lives.

1. **Read** one frame per view by index — `MultiViewReader.read(frame_id)`.
   Views are aligned by frame index only; `check_alignment()` reports differing
   lengths, resolutions or frame rates up front, and the run length is the
   shortest view.
2. **Detect / fetch a box** per view — `BBoxTracker.get(cam, frame_id, frame)`.
   On a miss it returns the last good box for up to `--bbox-max-age` frames,
   then drops the view. Counters land in `report.json` as `hit/reused/miss`.
3. **Pack** — `prepare_views()`:
   * square crop window per view: `bbox_get_center_scale(bbox, expand=2.0, mindim=200)`;
   * left hand → mirror the frame about the principal point, mirror the bbox
     center with it, mirror the extrinsics;
   * warp to 256×256 (`cv2.warpAffine`, zero padding, as in training);
   * crop intrinsics `K_crop = affine @ K`;
   * camera→master transforms, master = identity;
   * fewer than `min_views` survivors → `PackError`, frame skipped with a reason.
4. **Convert** — `packet_to_batch()`: `ToTensor` + `normalize(mean .5, std 1)`,
   stack to `(N, 3, 256, 256)`, `cam_view_num=[N]` so the model treats the N
   views as one sample.
5. **Forward** — `model(batch, 0, "inference", epoch_idx=0)` under `no_grad`.
6. **Unpack** — `unpack_prediction()`: joints/verts in the master frame, then
   `packet.master_to_world()` (which un-mirrors for a left hand).
7. **Score** — `_uv_vs_reproj_px()`: the network's own 2D keypoints, mapped back
   from crop pixels to the original frame, versus the reprojection of its 3D
   output. Stored per frame; the only accuracy signal available without ground
   truth.
8. **Draw** (optional) — reprojected skeleton per view, tiled, into `overlay.mp4`.
9. **Write** — `keypoints.npz` (`frame_ids`, `joints_world`, `joints_master`,
   `reproj_px`, `views`, optional `verts_world`), `keypoints.json`, `report.json`.

Skipped frames are *absent* from the arrays, never zero-filled — `frame_ids`
and `views` say exactly which frames and which views produced each row.

## 5. The geometry, derived

### 5.1 The crop invariant

The network sees a crop, so its camera must be the crop's camera. With
`center`, `scale` (square side) and output `(w, h)`, the image→crop affine is

```
A = [[ w/s,      0,   w(-cx/s + 1/2) ],
     [   0,  h/s·r,   h(-cy/s·r + 1/2) ],
     [   0,      0,                 1 ]]      r = w/h
```

which maps the crop window's top-left to `(0,0)`, its bottom-right to `(w,h)`
and its center to `(w/2, h/2)`. Since projection is linear in homogeneous
coordinates, cropping the projection equals projecting with `A·K`:

```
A · (K · X)  ==  (A · K) · X        =>   K_crop = A @ K
```

That identity is the pipeline's load-bearing invariant, and it is asserted
directly (`test_geometry.py::test_crop_intrinsics_match_cropping_the_projection`,
`test_views.py::test_crop_intrinsics_match_the_rendered_crop`) plus checked at
runtime in `--dry-run` (`crop_consistency_px_max`, ~1e-13 in practice).

The repo's own `_affine_transform_post_rot` additionally translates by the
optical center — which cancels exactly when `rot == 0`. Inference never rotates,
so `intrinsics_after_crop` accepts and ignores `optical_center`, and the parity
test confirms the two agree.

### 5.2 The master frame, and a naming trap

The model returns 3D in the **master** camera's frame (view 0). The batch key
that carries the extrinsics is `target_cam_extr`, but what the model wants there
is the **camera→master** transform, not world→camera: `_forward_impl` does
`T_c2m = inv(batch['target_cam_extr'])` and feeds *that* to the DLT as the
projection matrix. So, with `T_cw` the calibration's world→camera:

```
T_mc[i] = T_cw[master] @ inv(T_cw[i])        # camera i -> master
T_mc[master] = I
```

`ViewPacket` computes this once, and three tests pin the semantics: identity for
the master, correct point mapping in both directions, and — the strongest one —
`test_torch_batch.py::test_model_reference_triangulation_reproduces_the_truth`,
which replays the model's own reference-joint step by calling the repo's
`batch_triangulate_dlt_torch` on `K_crop` and `inv(T_mc)` and checking it returns
the ground-truth joints in the master frame.

### 5.3 Left hands: world mirroring

The model was trained on right hands. A left hand becomes a right hand under a
mirror, so for `--hand-side lh`:

* the frame is flipped horizontally about the principal point
  (`M = [[-1,0,2cx],[0,1,0]]`), which leaves `K` unchanged;
* the bbox center is mirrored with it: `cx' = 2·cx − cx`;
* the extrinsics are mirrored across the world YZ plane, delegated to the
  released `tool/flip_util.flip_cam_extr` (which operates on camera→world, hence
  `inv(flip(inv(T_cw)))`) so the behaviour is bit-identical to the demo;
* on the way out, `master_to_world()` negates x again.

Verified end to end against exact ground truth: the mirrored world projects onto
the flipped pixels (`test_views.py::test_left_hand_mirrored_projection_lands_on_the_flipped_pixels`),
the round trip returns the original world coordinates, and a full left-hand
sequence exports GT to 1e-4 m.

### 5.4 2D keypoints back to the original frame

`pred_joints_uv` comes out of the heatmap stage in **crop** pixels (the integral
heatmap is normalised to 0..1 then scaled by the input `W, H` —
`lib/models/POEM.py`, `heatmap_stage`). To compare it with anything in the
source video, `joints_uv_to_original()` applies `inv(A)` and then, for a left
hand, un-mirrors `u ← 2·cx − u`. This is what makes `reproj_px` meaningful.

### 5.5 Why mono cannot be patched around

With `N == batch_size` the model takes the `inputs_all_sv` branch and reads
`batch["master_joints_3d"]`. Supplying that means supplying the answer. Nothing
in this pipeline fabricates it: `prepare_views` raises `PackError` and
`PoemRunner.predict_packet` raises `ValueError` below two views, and the CLI
exits 2 with a message when fewer than two videos resolve. A monocular path
would mean replacing the DLT seed with a monocular 3D estimator — a model
change, not a flag.

## 6. Design decisions and why

**numpy-first, torch last.** Geometry bugs are the expensive ones and they need
no GPU to find. Keeping torch inside `packet_to_batch` made ~90 % of the code
testable on the laptop where it was written.

**`ViewPacket` as the boundary object.** The demo's `format_batch` mixed
cropping, mirroring, tensor building and `cv2.imshow` in one function, so none
of it could be inspected. `ViewPacket` carries the intermediate numpy state
(crops, `K_crop`, `T_cw_used`, `T_mc`, centers, scales, `flipped`, `dropped`),
which is what makes both the oracle-predictor tests and the runtime consistency
check possible.

**Refuse rather than degrade.** Below two views the code raises. The alternative
— quietly taking the model's single-view branch — yields confident, wrong,
GT-dependent output. Same principle for calibration: `sanity_check()` reports
non-metric translations, non-orthonormal rotations, co-located views and
`image_size` mismatches, and `--strict-calib` turns them into a hard stop.

**Boxes are pluggable, and sticky.** MediaPipe for a pure video capture, `npy`
for the released layout, `json` for any external detector, `full` for debugging.
`BBoxTracker` exists because a frame dies only when *fewer than two* views have
a box — reusing one stale box for a few frames rescues frames that would
otherwise be dropped, and the counters make that visible instead of implicit.

**A `--dry-run` that is honest.** It validates and reports; it never writes a
`keypoints.npz`. There is no mode in which this tool emits numbers that did not
come from the network.

**No in-place config rewriting.** `scripts/eval_single.py` mutates
`config/release/eval_single.yaml` on disk to switch model size. `build_cfg()`
applies the same overrides (embed dims, `PARAMETRIC_OUTPUT`, `PRETRAINED`) in
memory, so parallel runs cannot corrupt each other and the repo's configs stay
pristine.

**stdout is data, stderr is chatter.** Progress lines, calibration warnings and
the model banner go to stderr; stdout is a single JSON report. `... | jq` works,
and a test asserts it.

**Headless by construction.** No `imshow`, no open3d, no matplotlib anywhere on
the path — overlays are written as mp4 via cv2. Remote GPU boxes are the target.

**Follow the demo's numbers exactly.** `expand=2.0`, `mindim=200`, 256×256,
`normalize(0.5, 1.0)`, the `flip_util` mirroring, `cam_view_num` packing: all
copied deliberately, because any deviation puts the crops off-distribution
relative to training in a way no error message would reveal.

## 7. How the tests prove it without a GPU

121 cases, four tiers, gated by capability so the same command is meaningful
everywhere (`scripts/testing/run_test_matrix.sh`):

| tier | file(s) | needs | count |
| --- | --- | --- | --- |
| 1 | `test_calib_io`, `test_geometry`, `test_bbox`, `test_views`, `test_video_io`, `test_pipeline` | numpy, cv2, pytest | 91 |
| 2 | `test_torch_batch` | + CPU torch | 7 |
| 3 | `test_model_forward` | + pytorch3d, manotorch, MANO assets | 17 |
| 4 | `test_example_data` | + `POEM_CHECKPOINT`, `POEM_EXAMPLE_DATA` | 6 |

Skips always carry the reason (`conftest.py`: `requires_torch`,
`requires_torchvision`, `requires_config`, `requires_model_env`,
`requires_checkpoint`, `requires_example_data`). Nothing passes vacuously.

### The two ideas that make laptop testing worthwhile

**1. A synthetic capture with exact ground truth.**
`scripts/testing/make_synthetic_capture.py` renders an articulated 21-joint hand
(five 4-joint chains, curling over time, orbiting and rotating) into a 2-view
rectified stereo rig or an N-view ring, on a textured background, and writes:
videos per view, `calib.json`, boxes in both `json` and the released `npy`
layout, `hand_labels.json`, and `gt.npz` with the world-frame joints, per-camera
3D and per-view 2D. Because the projections are computed rather than measured,
the answer is known to floating-point precision — DLT triangulation of the
stored 2D recovers the stored 3D to ~1e-7 m. Fixtures build stereo, a 3-view
ring *with scripted detector failures*, and a left-hand capture.

**2. An oracle predictor in place of the network.**
`conftest.oracle_predictor` implements the same `predict_packet(packet)`
interface as `PoemRunner` and returns ground truth *shaped exactly as the model
would*: joints in the (possibly mirrored) master frame plus consistent
crop-space 2D keypoints. Substituting it turns "did the pipeline run?" into "is
the pipeline *correct*?" — every stage around the network (crops, master
bookkeeping, un-mirroring, the `reproj_px` metric, npz/json export) is checked
against GT to 1e-4 m, with no GPU involved.

### What each file locks down

* `test_calib_io.py` (15) — SE(3) inversion, rectified-stereo geometry
  (disparity sign, no vertical disparity), json round-trip, the released pickle
  layout, `look_at` correctness, and every sanity check firing: single view,
  millimeter translations, co-located views, non-orthonormal rotation,
  off-image principal point, duplicate names, malformed matrices.
* `test_geometry.py` (14) — the crop invariant, crop-window corner mapping, the
  non-square aspect convention, projection/transform round-trips, z≈0 clamping,
  DLT exactness and its 1 px-noise behaviour (< 1 cm at 0.6 m on a 12 cm
  baseline), mirroring as an involution, and **parity with the repo's own**
  affine helpers and `batch_triangulate_dlt_torch`.
* `test_bbox.py` (14) — every backend, multi-row and empty `npy` files, the
  tracker's reuse/expiry/reset and per-camera independence, factory validation,
  a clear error when MediaPipe is absent, and a check that the fixture's own
  boxes contain the GT keypoints.
* `test_views.py` (17) — packet shapes, master identity, camera→master
  correctness, crop-K vs rendered crop, non-blank crops, single-view refusal,
  2-of-3-view survival, 1-of-3 refusal, malformed-box reasons, master override,
  the world↔master round trip, both left-hand invariants, and `joints_uv`
  inversion for right and left hands.
* `test_video_io.py` (10) — frame geometry, RGB dtype/shape, random access
  matching sequential, index bounds, missing files, backend fallback, and the
  alignment warnings for truncated / rescaled / mismatched-rate views.
* `test_pipeline.py` (21) — dry-run over stereo and a 3-view rig with dropouts,
  reuse rescuing frames, GT recovery end to end, `joints_master` consistency,
  the left-hand sequence, frame-range selection, overlay writing, warnings
  reaching the report, and the CLI surface (video-arg forms, unknown/missing
  files, the mono exit code, `--reload` requirement, `--env-report`, clean JSON
  on stdout).
* `test_torch_batch.py` (7) — batch keys and shapes, `cam_view_num` grouping
  (asserting `BN != batch_size`, i.e. *not* the GT-seeded branch),
  normalisation recoverable from the uint8 crop, `target_cam_extr` semantics in
  both directions, the reference-triangulation replay, mirrored extrinsics in
  the left-hand batch, and float32 dtypes.
* `test_model_forward.py` (17, tier 3) — `environment_report`, `build_cfg` for
  all five model sizes plus checkpoint/backbone wiring, device resolution, then
  2-view / 3-view / 2-of-3-view forwards, determinism, both mono guards
  (including one that *demonstrates* the GT dependency by feeding a genuine
  single-view batch and expecting `KeyError`), strict checkpoint loading, and a
  metric-plausibility check on real weights.
* `test_example_data.py` (6, tier 4) — the released capture: layout and
  calibration, a dry run, then metric/temporal assertions (2D-3D agreement
  < 20 px, hand span 0.10–0.35 m, rigid bone lengths over time, wrist
  continuity < 5 cm/frame), the left-hand sequence on real imagery, a printed
  2-view vs N-view MPJPE, and MediaPipe-vs-shipped-box IoU.

### Result as of this commit

Tiers 1–2 run and pass on macOS (M-series, no CUDA, no pytorch3d):
**106 passed, 15 skipped**. Tiers 3–4 are pending a Linux + NVIDIA box; the
sequence to run them is `plan_of_action.md` §4.

`scripts/testing/run_synthetic_smoke.sh` is the companion end-to-end check:
stage A renders three captures and dry-runs them (no torch), stage B repeats
with a real checkpoint and writes overlays.

## 8. Failure-mode catalogue

Everything below produces *plausible* output if unnoticed, which is why each has
a designated detector.

| failure | symptom | caught by |
| --- | --- | --- |
| extrinsics in millimeters | 3D scaled 1000× | `sanity_check()` "non-metric" |
| rotation block not a rotation | skewed 3D | `sanity_check()` orthonormality/det |
| views co-located / tiny baseline | noisy depth | `sanity_check()` "co-located" |
| `K` for a different resolution than the video | systematic offset | `image_size` warning in `report.json` |
| video resized after calibration | same | same warning |
| unsynchronised views | biased 3D, no error | `check_alignment()` (length/size/rate); a constant offset is **not** detectable — sync in hardware |
| loose or jittery boxes | degraded, jittery 3D | `bbox_stats`, `reproj_px`, the overlay |
| detector misses a view | frame skipped | `frames_skipped` + `skipped_examples` |
| only one view usable | refused | `PackError` / exit 2 |
| wrong hand side | mirrored nonsense | `reproj_px` blows up; visible in overlay |
| crop math wrong | wrong 3D from right 2D | `crop_consistency_px_max`; the parity tests |
| checkpoint / `--model` size mismatch | load error | strict `load_state_dict`; tier-3 weight check |
| hand beyond ~1.2 m | off-distribution | documented; `joints_master[..., 2]` check in tier 4 |

## 9. Extending it

**A different detector.** Subclass `BBoxProvider` with
`get(cam_name, frame_id, frame) -> (4,) xyxy | None`, register it in
`build_bbox_provider`, add a `--bbox-backend` choice. Everything downstream is
unchanged. Simplest alternative: dump boxes to `bbox.json` and use `--bbox-backend json`.

**A different calibration source.** Return a `CameraRig` of `Camera(name, K,
T_cw, image_size, dist)`; add a loader to `calib.py` and a sub-command to
`make_calib.py`. Keep the first camera as the intended master.

**Two hands.** One hand per run today. Run twice with per-hand boxes
(`--bbox-backend json`) and `--hand-side rh` / `lh`; merge the two npz files
afterwards.

**Temporal smoothing.** Nothing here filters across time — deliberately, since
per-frame output plus `reproj_px` is the honest primitive. A filter belongs
downstream of `keypoints.npz`, where `frame_ids` marks the gaps.

**Throughput.** Frames are processed one at a time (batch size 1, N views).
The model supports several samples per batch via `cam_view_num`, so batching
consecutive frames is the natural speed-up: extend `packet_to_batch` to
concatenate packets and set `cam_view_num=[N0, N1, ...]`.

**Undistortion.** `Camera.dist` is carried but not applied. Feed rectified /
undistorted video and rectified `K`, or add an undistort step in `video.py` —
then `dist` must be zeroed so it cannot be applied twice.

## 10. File reference

| path | lines | role |
| --- | --- | --- |
| `tool/poemkit/calib.py` | 386 | `Camera`, `CameraRig`, `inv_se3`, four loaders, `sanity_check`, `baselines` |
| `tool/poemkit/geometry.py` | 176 | crop affine, crop intrinsics, projection, transforms, mirroring, numpy DLT, reprojection error |
| `tool/poemkit/bbox.py` | 290 | `bbox_from_points`, four providers, `BBoxTracker`, factory |
| `tool/poemkit/video.py` | 187 | `CV2FrameReader`, `FFmpegFrameReader`, `MultiViewReader`, `VideoWriter` |
| `tool/poemkit/views.py` | 323 | `prepare_views`, `ViewPacket`, `packet_to_batch`, `unpack_prediction`, `joints_uv_to_original`, `project_to_views` |
| `tool/poemkit/render.py` | 97 | `HAND_BONES`, skeleton/point drawing, view tiling |
| `tool/poemkit/runner.py` | 208 | `environment_report`, `resolve_device`, `build_cfg`, `PoemRunner` |
| `tool/infer_video.py` | 509 | CLI, `run_sequence`, exports, health metrics |
| `tool/make_calib.py` | 186 | `look_at_extrinsic`, `make_ring_rig`, five sub-commands |
| `scripts/testing/make_synthetic_capture.py` | 279 | synthetic hand + rig renderer with ground truth |
| `scripts/testing/run_test_matrix.sh` | 63 | tiered test runner |
| `scripts/testing/run_synthetic_smoke.sh` | 82 | end-to-end smoke, with or without a checkpoint |
| `tests/` | 2113 | 121 cases, `conftest.py` fixtures + capability gates |

Untouched: every pre-existing file. The pipeline is additive, and
`tool/infer_hand.py` still runs exactly as before.
