# Handoff brief: continue the POEM-v2 video-only pipeline on a GPU server

You are picking up work that was prepared on a laptop with no GPU. The code is
written and the GPU-independent half is tested; your job is to build the
environment, execute the tests that could not run, measure the numbers that are
still unknown, and record the results.

**Read these first, in this order** (they are in the repo, not here):

1. `plan_of_action.md` — status board; §4 is your task list, §5 the open risks.
2. `docs/video_inference.md` — how to run the tool.
3. `docs/pipeline_design.md` — architecture, geometry, test strategy.

Do not re-derive what those documents already establish, and do not redesign
the pipeline. Everything below assumes them.

---

## 0. Context you need up front

* Repo: `https://github.com/abirsinha-humynlabs/POEM-v2.git` (a fork of the
  official POEM-v2). Work happens on branch **`feature/video-only-inference`**,
  which is ahead of `release`.
* Goal: 3D hand keypoints from RGB video only — no depth, no IMU, no ground truth.
* Settled facts, not open questions:
  * **≥ 2 calibrated, synchronised views are mandatory.** Mono is impossible:
    `lib/models/POEM.py:276-305` seeds reference joints from
    `batch["master_joints_3d"]` (ground truth) when there is one view per sample.
  * Output is 21 joints + 778 MANO vertices, **metric meters**, in the master
    (first) camera's frame.
  * The pipeline needs a per-view hand bbox each frame (POEM-v2 is not a
    detector) and the hand side.
* What exists: `tool/poemkit/` (calibration, geometry, boxes, video, view
  packing, runner, overlays), `tool/infer_video.py` (headless CLI),
  `tool/make_calib.py`, `scripts/testing/` (synthetic capture generator, test
  matrix, smoke script), `tests/` (121 cases in 4 tiers).
* Test tiers and current state (from the laptop): tier 1 (91) and tier 2 (7)
  **pass** — 106 passed, 15 skipped. Tier 3 (17, model forward) and tier 4
  (6, released example data) have **never executed**.
* A skip is not a pass. Every gate prints why it skipped; if a tier you were
  asked to run reports skips, the environment is incomplete — fix that first.

## 1. Get the code

```bash
git clone https://github.com/abirsinha-humynlabs/POEM-v2.git   # or: cd into the existing clone
cd POEM-v2
git fetch origin
git checkout feature/video-only-inference
git pull origin feature/video-only-inference
git log --oneline -5      # expect: "Document the video-only pipeline design" at or near HEAD
```

Work on this branch. Commit here; do not merge into `release` without being
asked.

## 2. Build the environment

Follow `docs/installation.md` exactly — the pins matter:

```bash
conda env create -f environment.yml && conda activate POEM     # python 3.8
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
pip install git+https://github.com/lixiny/manotorch.git@v0.0.2
pip install transformers
pip install -r requirements-video-infer.txt                     # mediapipe, ffmpeg-python, pytest
sh prepare/download_hrnet.sh                                    # optional at inference; see below
```

Skip `neural_renderer`, `dex_ycb_toolkit` and `oikit` — none are on the
inference path (the repo's `neural_renderer` wrapper is unused by it).

**MANO assets are a manual step and a likely blocker.** The model constructs a
`ManoLayer` with `mano_assets_root="assets/mano_v1_2"`. Those files require
registration at https://mano.is.tue.mpg.de and cannot be downloaded
non-interactively. If `assets/mano_v1_2` is absent, stop and ask the user for
them rather than trying to work around it.

Verify before going further:

```bash
python -m tool.infer_video --env-report --out unused
```

`missing` must be `[]`, `mano_assets` must be `true`, `cuda` must be `true`.

Two known environment traps, already handled in code — do not "fix" them again:
`build_cfg()` blanks `BACKBONE.PRETRAINED` when `--reload` is given (the HRNet
loader calls `torch.load` without `map_location`), and configs are overridden in
memory rather than rewritten on disk the way `scripts/eval_single.py` does.

## 3. Task A — tier 3: model construction and forward passes

```bash
scripts/testing/run_test_matrix.sh tier3
```

17 cases: config overrides for all five model sizes, device resolution, then
2-view / 3-view / 2-of-3-view forwards, output shapes and finiteness,
determinism, and both mono guards — including
`test_single_view_batch_needs_ground_truth`, which feeds a genuine single-view
batch and expects a `KeyError`, demonstrating the ground-truth dependency rather
than asserting it in prose. Weights are random here by design; only structure is
checked.

If a case fails, decide whether it is the environment or the code, and say which
in your report. Fix code on this branch; do not weaken an assertion to make it
pass.

## 4. Task B — checkpoints, then tier 3 again

Download the checkpoints from the README's `ckpt_release` folder
(https://drive.google.com/drive/folders/16BRH8zJ7fbR7QNluHHEshZMJc1wMRr_k) into
`./checkpoints/` — `medium.pth.tar` is enough to start. If the Drive folder
cannot be fetched headlessly, ask the user rather than guessing at a mirror.

```bash
export POEM_CHECKPOINT=$PWD/checkpoints/medium.pth.tar
scripts/testing/run_test_matrix.sh tier3
```

This adds a strict weight-load check (every checkpoint tensor must land in the
model) and a metric-plausibility check on real weights.

## 5. Task C — tier 4: the released example data (the real gate)

```bash
# https://huggingface.co/kelvin34501/POEM-v2_example_data/blob/main/example_data.tar.xz
tar -xf example_data.tar.xz -C /path/to/data
export POEM_EXAMPLE_DATA=/path/to/data/example_data
export POEM_CHECKPOINT=$PWD/checkpoints/medium.pth.tar
scripts/testing/run_test_matrix.sh tier4          # POEM_TEST_FRAMES=50 for a longer run
```

Six cases. Capture these numbers from the output — they are the point of the
exercise:

| what | where it comes from | why it matters |
| --- | --- | --- |
| `reproj_px_mean` | the metric run | 2D/3D agreement; the only no-GT accuracy signal |
| median hand span (m) | same test | catches unit errors |
| bone-length std over time | same test | a rigid hand cannot change size |
| wrist step per frame | same test | temporal continuity |
| **2-view vs N-view MPJPE (mm)** | `test_view_count_changes_the_answer_but_not_the_scale` (prints it) | **how much accuracy a stereo rig costs** |
| MediaPipe vs shipped-box median IoU | `test_mediapipe_boxes_agree_with_the_shipped_boxes` | whether a detector can replace the mask pipeline |

Run that tier with `-s` (the matrix script already does) so the printed numbers
are visible.

## 6. Task D — synthetic smoke with real weights

```bash
scripts/testing/run_synthetic_smoke.sh /tmp/poem_smoke checkpoints/medium.pth.tar medium
```

Stereo, 3-view-with-dropouts and left-hand captures through the CLI end to end,
writing `overlay.mp4` per case. The imagery is cartoon hands: judge plumbing
(completes, right shapes, no NaNs), not accuracy.

## 7. Task E — measure throughput

Still unmeasured and listed as a risk in `plan_of_action.md` §5. On the example
data, time a bounded run and report **seconds per frame at 2 views and at all
views**, plus the GPU used:

```bash
python -m tool.infer_video --calib <example_data>/calib/calib__* \
    --video-dir <example_data>/data/<sequence> \
    --bbox-backend npy --bbox-root <example_data>/human_mask_hand/<sequence> \
    --hand-side rh --cfg config/release/eval_single.yaml --model medium \
    --reload checkpoints/medium.pth.tar --device cuda:0 \
    --max-frames 100 --out exp/throughput
```

`report.json` carries `seconds` and `frames_predicted`. Also note GPU memory.

## 8. Task F — report and record

Update `plan_of_action.md`:

* move §4's completed steps from pending to done, **with the measured numbers
  inline** (do not leave "pending" language behind);
* update §3's status table with real tier 3/4 results;
* revise §5: the stereo-accuracy and throughput risks should become measured
  facts, or stay risks with evidence.

Then commit on `feature/video-only-inference` and push:

```bash
git add -A && git commit && git push origin feature/video-only-inference
```

End commit messages with:

```
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

Do not commit checkpoints, example data, `exp/` or generated captures — they are
gitignored; keep it that way.

## 9. Ground rules

* **Additive only.** No pre-existing repo file has been modified, and
  `tool/infer_hand.py` still runs as shipped. Keep it that way unless a real bug
  in the existing code blocks you — then say so explicitly.
* **Report honestly.** If a tier fails or is skipped, say so with the output.
  Do not describe a skipped test as passing.
* **Do not relax tests to get green.** The thresholds in tier 4 (20 px, span
  0.10–0.35 m, 6 mm bone drift, 5 cm/frame wrist step) were chosen to catch
  silent failures. If one trips, investigate first; if you conclude a threshold
  is genuinely wrong, change it in a separate commit that explains why.
* **Environment is Python 3.8.** Keep any new code 3.8-compatible.
* If something needs a human — MANO registration, a Drive download, a GPU that
  is not there — stop and ask rather than improvising.

## 10. After the tests: the user's own capture

The endpoint is the user's stereo/multi-view footage, not the example data. When
the tiers are green:

1. `python -m tool.make_calib stereo|opencv ... --out calib.json`, then
   `python -m tool.make_calib check --calib calib.json` — fix every warning.
2. `python -m tool.infer_video --dry-run` on the capture, before spending GPU
   time: it validates calibration, synchronisation, boxes and crops.
3. Full run with `--bbox-backend mediapipe --overlay`, then read
   `report.json`'s `reproj_px_mean` and watch the overlay before trusting the
   numbers.

Constraints to keep in mind: the trained volume is x,y ∈ [-0.6, 0.6] m and
z ∈ [0, 1.2] m from the master camera; views are aligned by frame index only, so
a constant temporal offset between cameras cannot be detected by the tooling and
must be prevented in hardware.
