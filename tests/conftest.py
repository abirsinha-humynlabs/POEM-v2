"""Shared fixtures for the POEM-v2 video-inference test suite.

Test tiers (see ``plan_of_action.md``):

* **tier 1 - laptop, no torch**: calibration, geometry, boxes, crops, video IO,
  packing, dry-run pipeline. Needs numpy + opencv + pytest only.
* **tier 2 - laptop with CPU torch**: batch tensors, DLT parity with the repo's
  torch implementation. Marked ``torch``.
* **tier 3 - GPU box, full env**: model construction and forward passes.
  Marked ``model`` (needs pytorch3d, manotorch and ``assets/mano_v1_2``).
* **tier 4 - GPU box, checkpoint + real data**: accuracy on the released
  example data. Marked ``realdata`` (needs ``POEM_CHECKPOINT`` and
  ``POEM_EXAMPLE_DATA`` environment variables).

Anything unavailable is skipped with a reason, never silently passed.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a broken/partial install
        return False


HAVE_TORCH = _have("torch")
HAVE_TORCHVISION = HAVE_TORCH and _have("torchvision")
HAVE_MODEL_ENV = HAVE_TORCHVISION and _have("pytorch3d") and _have("manotorch")
MANO_ASSETS = os.path.isdir(os.path.join(REPO_ROOT, "assets", "mano_v1_2"))
CHECKPOINT = os.environ.get("POEM_CHECKPOINT")
EXAMPLE_DATA = os.environ.get("POEM_EXAMPLE_DATA")

HAVE_CONFIG = _have("yacs")

requires_config = pytest.mark.skipif(not HAVE_CONFIG, reason="yacs not installed (needed for lib.utils.config)")
requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
requires_torchvision = pytest.mark.skipif(not HAVE_TORCHVISION, reason="torchvision not installed")
requires_model_env = pytest.mark.skipif(
    not (HAVE_MODEL_ENV and MANO_ASSETS),
    reason="needs torch + pytorch3d + manotorch and assets/mano_v1_2 (see docs/installation.md)",
)
requires_checkpoint = pytest.mark.skipif(
    not (HAVE_MODEL_ENV and MANO_ASSETS and CHECKPOINT and os.path.isfile(CHECKPOINT or "")),
    reason="set POEM_CHECKPOINT=/path/to/checkpoints/<model>.pth.tar",
)
requires_example_data = pytest.mark.skipif(
    not (EXAMPLE_DATA and os.path.isdir(EXAMPLE_DATA or "")),
    reason="set POEM_EXAMPLE_DATA=/path/to/extracted/example_data",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: needs torch (CPU is enough)")
    config.addinivalue_line("markers", "model: needs the full model environment (pytorch3d, manotorch, MANO assets)")
    config.addinivalue_line("markers", "realdata: needs a checkpoint and the released example data")
    config.addinivalue_line("markers", "slow: renders video or runs the network")


# ---------------------------------------------------------------------------
# synthetic captures
# ---------------------------------------------------------------------------
def _generate(tmp_path_factory, name, **kwargs):
    from scripts.testing.make_synthetic_capture import generate

    out_dir = tmp_path_factory.mktemp(name)
    return generate(out_dir=str(out_dir), **kwargs)


@pytest.fixture(scope="session")
def stereo_capture(tmp_path_factory):
    """2-view rectified stereo, right hand, 8 frames."""
    return _generate(tmp_path_factory, "stereo", rig_kind="stereo", num_frames=8, seed=1)


@pytest.fixture(scope="session")
def ring_capture(tmp_path_factory):
    """3-view ring rig, right hand, with view drop-outs to exercise the skip logic.

    ``cam1`` loses its box on frames 2-3 (2 views remain -> still usable) and
    ``cam2`` loses frame 3 as well, so frame 3 has a single view and must be
    skipped rather than fed to the model.
    """
    return _generate(
        tmp_path_factory,
        "ring",
        rig_kind="ring",
        num_cams=3,
        num_frames=6,
        seed=2,
        drop_view_frames={"cam1": [2, 3], "cam2": [3]},
    )


@pytest.fixture(scope="session")
def left_hand_capture(tmp_path_factory):
    """2-view stereo capture of a LEFT hand: exercises the world-mirroring path."""
    return _generate(tmp_path_factory, "lefthand", rig_kind="stereo", num_frames=6, hand_side="lh", seed=3)


@pytest.fixture(scope="session")
def stereo_gt(stereo_capture):
    import numpy as np

    return np.load(stereo_capture["gt"])


@pytest.fixture
def oracle_predictor():
    """A stand-in for :class:`~tool.poemkit.runner.PoemRunner` that returns ground truth.

    It answers with the GT joints expressed exactly as the real model would
    (master camera frame, mirrored for a left hand) plus consistent 2D
    keypoints, so the pipeline around the network -- crops, master bookkeeping,
    the un-mirroring, the export -- can be tested for correctness rather than
    just for "it ran".
    """
    import numpy as np

    from tool.poemkit.geometry import mirror_points_x, project_points, transf_points

    class OraclePredictor:

        def __init__(self, joints_world_by_frame):
            self.joints_world_by_frame = joints_world_by_frame
            self.calls = 0

        def predict_packet(self, packet):
            self.calls += 1
            gt_world = np.asarray(self.joints_world_by_frame[packet.frame_id], dtype=np.float64)
            # the model works in the (possibly mirrored) master camera frame
            world_for_model = mirror_points_x(gt_world) if packet.flipped else gt_world
            joints_master = transf_points(packet.T_cw_used[0], world_for_model)
            joints_uv = np.stack([
                project_points(packet.K_crop[i], transf_points(packet.T_cw_used[i], world_for_model))
                for i in range(packet.num_views)
            ], axis=0)
            verts_master = np.repeat(joints_master[:1], 778, axis=0)  # shape-only placeholder

            pred = {
                "pred_joints_3d": joints_master[None],
                "pred_verts_3d": verts_master[None],
                "pred_joints_uv": joints_uv,
            }
            from tool.poemkit.views import unpack_prediction

            return unpack_prediction(pred, packet)

    return OraclePredictor
