# Video-only inference (`tool/infer_video.py`)

How to get metric 3D hand keypoints out of POEM-v2 from **nothing but
synchronised RGB video plus calibration** — no depth, no IMU, no ground truth.

The released demo (`tool/infer_hand.py`) is hard-wired to one capture: fixed
camera serials, pickled calibration, precomputed hand masks, an on-screen
viewer, and `cuda:0`. This document covers the generalised path added on top of
it.

For how that path is built and why — architecture, the crop/master-frame/
mirroring geometry, design rationale and the test strategy — see
[`pipeline_design.md`](pipeline_design.md). For current status and what is
still pending, see [`../plan_of_action.md`](../plan_of_action.md).

---

## 1. What the model gives you

Per frame, from `pred_joints_3d` / `pred_verts_3d`:

| output | shape | frame | units |
| --- | --- | --- | --- |
| `joints_master` | `(21, 3)` | master camera (view 0) | meters |
| `joints_world` | `(21, 3)` | world frame of your extrinsics | meters |
| `verts_world` (opt.) | `(778, 3)` | world frame | meters |
| `joints_uv` | `(N_views, 21, 2)` | crop pixels | pixels |

Joint order is the repo's 21-joint convention: index 0 is the wrist, then five
4-joint chains (thumb, index, middle, ring, pinky) — see
`tool/poemkit/render.HAND_BONES` and `lib/viztools/draw.plot_hand`.

## 2. What it needs from you

1. **Two or more synchronised, calibrated views.** Not optional — see §6.
2. **`K` and `T_cw` per camera**, in meters, describing *the frames you feed in*.
3. **A hand bbox per view per frame.** POEM-v2 is not a detector.
4. **The hand side** (`rh` / `lh`); left hands are handled by mirroring.

## 3. Calibration

`tool/make_calib.py` writes the `calib.json` the runner reads:

```bash
# already-rectified stereo pair (baseline in METERS)
python -m tool.make_calib stereo --fx 700 --fy 700 --cx 640 --cy 360 \
    --baseline 0.12 --image-size 1280 720 --out data/cap/calib.json

# OpenCV stereoCalibrate / stereoRectify FileStorage
python -m tool.make_calib opencv --yml stereo.yml --rectified --out calib.json

# the released example data's pickles
python -m tool.make_calib poem --calib-dir example_data/calib/calib__2025_0319_1534_41 \
    --out calib.json

# validate anything
python -m tool.make_calib check --calib calib.json
```

`calib.json` schema (first camera is the master, whose frame the model predicts in):

```json
{"cameras": [
  {"name": "left",  "image_size": [1280, 720],
   "K": [[700,0,640],[0,700,360],[0,0,1]], "T_cw": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
  {"name": "right", "image_size": [1280, 720],
   "K": [[700,0,640],[0,700,360],[0,0,1]], "T_cw": [[1,0,0,-0.12],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
]}
```

`P_c = T_cw · P_w`. Rectified frames ⇒ `dist` is zeros and `K` comes from the
*rectified* `P1`/`P2`, not the raw intrinsics.

**Failure modes that produce plausible-but-wrong 3D**, all caught by
`CameraRig.sanity_check()` and reported in `report.json`:
millimeter translations, a non-orthonormal rotation block, co-located views,
a principal point outside the image, `image_size` disagreeing with the video.

## 4. Running it

```bash
python -m tool.infer_video \
    --calib data/cap/calib.json \
    --video left=data/cap/left.mp4 --video right=data/cap/right.mp4 \
    --hand-side rh \
    --bbox-backend mediapipe \
    --cfg config/release/eval_single.yaml --model medium \
    --reload checkpoints/medium.pth.tar \
    --device cuda:0 --out exp/cap --overlay --save-verts
```

`--video-dir DIR` is shorthand when the files are named `<camera>.mp4`.

Outputs in `--out`:

| file | contents |
| --- | --- |
| `keypoints.npz` | `frame_ids`, `joints_world`, `joints_master`, `reproj_px`, `views`, optional `verts_world` |
| `keypoints.json` | the same joints, plain json |
| `report.json` | frames kept/skipped, bbox stats, calibration warnings, 2D/3D agreement |
| `overlay.mp4` | reprojected skeleton per view (with `--overlay`) |

### bbox backends

| `--bbox-backend` | source | notes |
| --- | --- | --- |
| `mediapipe` | MediaPipe Hands landmarks | CPU, `pip install mediapipe`; the video-only default |
| `npy` | `<root>/<cam>/bbox/%05d.npy` | the released example-data layout |
| `json` | `{"<cam>": {"<frame>": [x0,y0,x1,y1]}}` | hand-off from any detector |
| `full` | whole frame | debug only; crops don't match training |

`--bbox-max-age N` reuses the last good box for up to N frames when a detector
misses, which keeps a frame alive as long as ≥2 views still have a box.

### validating a capture without a GPU

```bash
python -m tool.infer_video --calib ... --video-dir ... --dry-run --out /tmp/check
python -m tool.infer_video --env-report --out unused
```

`--dry-run` needs neither torch nor a checkpoint. It checks calibration,
view synchronisation, box availability, view counts per frame and the
crop-intrinsic bookkeeping (`crop_consistency_px_max` must be ~1e-13).

## 5. Reading the health metrics

* `reproj_px_mean` — mean pixel distance between the network's 2D keypoints and
  its reprojected 3D joints. The only no-ground-truth accuracy signal. A few
  pixels is healthy; tens of pixels means bad extrinsics, unsynchronised views,
  or wrong crops.
* `frames_skipped` + `skipped_examples` — frames with <2 boxed views.
* `bbox_stats` — `hit` / `reused` / `miss` per the tracker.
* `warnings` — calibration and synchronisation problems.

## 6. Why mono is not supported

`lib/models/POEM.py:276-305`: with more than one view the model triangulates its
own predicted 2D keypoints (DLT) to seed the transformer's reference joints. With
exactly one view per sample (`inputs_all_sv`) it instead reads
`batch["master_joints_3d"]` — **ground-truth 3D joints**, which do not exist at
inference time. The single-view numbers in the README are therefore GT-seeded and
Procrustes-aligned; absolute position is unrecoverable from one view anyway.

Both `prepare_views()` and `PoemRunner.predict_packet()` refuse fewer than two
views rather than silently taking that path.

Practical notes for a stereo rig:

* a short baseline weakens the DLT seed — the wider the better, as long as both
  cameras see the hand;
* `POSITION_RANGE` in the release config covers x,y ∈ [-0.6, 0.6] m and
  z ∈ [0, 1.2] m around the master camera, so keep the hand within ~1.2 m;
* rectification is not required, only calibration.

## 7. Hardware

Supported: **Linux + NVIDIA GPU**, Python 3.8, torch 1.11.0+cu113, and the cu113
pytorch3d wheel (`docs/installation.md`). `--device cpu` works — the model code
is written against `x.device` — but is seconds-per-frame, useful only for shape
and plumbing checks. `--device mps` is accepted and unverified: pytorch3d's
`knn_points`/`ball_query` have no Metal kernels.

`--reload` supplies all weights, so the HRNet ImageNet checkpoint is not needed
at inference; `build_cfg` blanks `BACKBONE.PRETRAINED` in that case (that loader
calls `torch.load` without `map_location`, which breaks on CPU-only hosts).

## 8. Tests

```bash
scripts/testing/run_test_matrix.sh          # every tier this machine supports
scripts/testing/run_synthetic_smoke.sh /tmp/poem_smoke [CKPT] [MODEL]
```

See `plan_of_action.md` for the tier layout and what is still pending.
