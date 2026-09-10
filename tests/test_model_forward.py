"""T9 - model construction and forward passes (GPU box / full environment).

Everything here is skipped on a laptop: it needs pytorch3d, manotorch and
``assets/mano_v1_2``. Run it first on the rented GPU machine -- it is the
cheapest way to find out whether the environment is really assembled, before
burning time on a long sequence.

With random weights the *values* are meaningless, so the assertions are about
shapes, devices, finiteness and the view-count contract. The value-level
assertions live in ``test_example_data.py``, which needs a checkpoint and the
released capture.
"""

import numpy as np
import pytest

from conftest import CHECKPOINT, requires_checkpoint, requires_config, requires_model_env, requires_torch
from tool.poemkit.bbox import NpyBBoxProvider
from tool.poemkit.calib import load_calib_json
from tool.poemkit.video import MultiViewReader
from tool.poemkit.views import prepare_views

CFG = "config/release/eval_single.yaml"


def _packet(manifest, frame_id=0, min_views=2, drop=()):
    rig = load_calib_json(manifest["calib"])
    provider = NpyBBoxProvider(manifest["bbox_root"])
    with MultiViewReader({n: manifest["videos"][n] for n in rig.names}, backend="cv2") as reader:
        frames = reader.read(frame_id)
    boxes = {name: (None if name in drop else provider.get(name, frame_id)) for name in rig.names}
    return rig, prepare_views(frames=frames, bboxes=boxes, rig=rig, frame_id=frame_id, min_views=min_views)


# ---------------------------------------------------------------------------
# environment / config (cheap, no model build)
# ---------------------------------------------------------------------------
def test_environment_report_is_informative():
    from tool.poemkit.runner import environment_report

    report = environment_report(".")
    assert set(report) >= {"modules", "missing", "mano_assets", "cuda", "devices"}
    assert "cpu" in report["devices"]
    for name in ("torch", "pytorch3d", "manotorch"):
        assert name in report["modules"]


@requires_config
@pytest.mark.parametrize("size,embed", [("small", 128), ("medium", 256), ("large", 512), ("huge", 1024),
                                        ("medium_MANO", 256)])
def test_build_cfg_applies_the_model_size(size, embed):
    """Same overrides ``scripts/eval_single.py`` writes into the yaml, in memory."""
    from tool.poemkit.runner import build_cfg

    cfg = build_cfg(CFG, model_size=size)
    assert cfg.MODEL.HEAD.EMBED_DIMS == embed
    assert cfg.MODEL.HEAD.POINTS_FEAT_DIM == embed
    assert cfg.MODEL.HEAD.TRANSFORMER.INPUT_FEAT_DIM == embed
    assert cfg.MODEL.HEAD.POSITIONAL_ENCODING.NUM_FEATS == embed // 2
    assert cfg.MODEL.HEAD.TRANSFORMER.PARAMETRIC_OUTPUT == (size == "medium_MANO")


@requires_config
def test_build_cfg_wires_the_checkpoint_and_skips_the_backbone_load():
    from tool.poemkit.runner import build_cfg

    cfg = build_cfg(CFG, checkpoint="/tmp/whatever.pth.tar")
    assert cfg.MODEL.PRETRAINED == "/tmp/whatever.pth.tar"
    assert cfg.MODEL.BACKBONE.PRETRAINED == "", \
        "a full checkpoint already carries backbone weights; the HRNet loader has no map_location"

    cfg_no_ckpt = build_cfg(CFG)
    assert cfg_no_ckpt.MODEL.BACKBONE.PRETRAINED.endswith(".pth")


@requires_config
def test_build_cfg_validates_inputs():
    from tool.poemkit.runner import build_cfg

    with pytest.raises(ValueError, match="unknown model size"):
        build_cfg(CFG, model_size="enormous")
    with pytest.raises(FileNotFoundError):
        build_cfg("config/release/does_not_exist.yaml")


@requires_torch
def test_resolve_device():
    import torch

    from tool.poemkit.runner import resolve_device

    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
            resolve_device("cuda:0")
    else:
        assert resolve_device("cuda:0").type == "cuda"


# ---------------------------------------------------------------------------
# forward passes
# ---------------------------------------------------------------------------
@requires_model_env
@pytest.mark.model
@pytest.mark.slow
def test_two_view_forward_shapes(stereo_capture):
    """A stereo frame through the real network: 21 joints + 778 verts, no NaNs."""
    from tool.poemkit.runner import PoemRunner

    runner = PoemRunner(cfg_path=CFG, checkpoint=None, model_size="small", device="cpu", verbose=False)
    _, packet = _packet(stereo_capture)
    out = runner.predict_packet(packet)

    assert out["joints_master"].shape == (21, 3)
    assert out["verts_master"].shape == (778, 3)
    assert out["joints_world"].shape == (21, 3)
    assert out["joints_uv"].shape == (2, 21, 2)
    for key, value in out.items():
        assert np.all(np.isfinite(value)), f"{key} contains non-finite values"
    assert runner.hand_faces.shape[1] == 3, "MANO faces should be triangles"


@requires_model_env
@pytest.mark.model
@pytest.mark.slow
def test_three_view_forward(ring_capture):
    """Any number of views >= 2 is accepted; the extra view only changes N."""
    from tool.poemkit.runner import PoemRunner

    runner = PoemRunner(cfg_path=CFG, checkpoint=None, model_size="small", device="cpu", verbose=False)
    _, packet = _packet(ring_capture, frame_id=0)
    assert packet.num_views == 3

    out = runner.predict_packet(packet)
    assert out["joints_uv"].shape == (3, 21, 2)
    assert out["joints_master"].shape == (21, 3)


@requires_model_env
@pytest.mark.model
@pytest.mark.slow
def test_two_of_three_views_still_runs(ring_capture):
    """Occlusion: drop a view and the same code path must carry the frame."""
    from tool.poemkit.runner import PoemRunner

    runner = PoemRunner(cfg_path=CFG, checkpoint=None, model_size="small", device="cpu", verbose=False)
    _, packet = _packet(ring_capture, frame_id=0, drop=("cam1",))
    assert packet.num_views == 2
    assert runner.predict_packet(packet)["joints_master"].shape == (21, 3)


@requires_model_env
@pytest.mark.model
def test_single_view_is_refused_by_the_runner(stereo_capture):
    """The mono limitation, at the runner boundary."""
    from tool.poemkit.runner import PoemRunner

    runner = PoemRunner(cfg_path=CFG, checkpoint=None, model_size="small", device="cpu", verbose=False)
    _, packet = _packet(stereo_capture, min_views=1, drop=("right",))
    assert packet.num_views == 1

    with pytest.raises(ValueError, match=">= 2 views"):
        runner.predict_packet(packet)


@requires_model_env
@pytest.mark.model
@pytest.mark.slow
def test_single_view_batch_needs_ground_truth(stereo_capture):
    """Prove *why* mono is unsupported: with one view per sample the model reads
    ``master_joints_3d`` (ground truth) to seed its reference joints, so a
    genuine single-view batch cannot be built at inference time."""
    from tool.poemkit.runner import PoemRunner
    from tool.poemkit.views import packet_to_batch

    runner = PoemRunner(cfg_path=CFG, checkpoint=None, model_size="small", device="cpu", verbose=False)
    _, packet = _packet(stereo_capture, min_views=1, drop=("right",))
    batch = packet_to_batch(packet, device="cpu")
    assert batch["image"].shape[0] == len(batch["cam_view_num"]) == 1  # the inputs_all_sv branch

    with pytest.raises(KeyError):
        runner.predict_batch_dict(batch)


@requires_model_env
@pytest.mark.model
@pytest.mark.slow
def test_forward_is_deterministic(stereo_capture):
    from tool.poemkit.runner import PoemRunner

    runner = PoemRunner(cfg_path=CFG, checkpoint=None, model_size="small", device="cpu", verbose=False)
    _, packet = _packet(stereo_capture, frame_id=1)
    first = runner.predict_packet(packet)["joints_master"]
    second = runner.predict_packet(packet)["joints_master"]
    np.testing.assert_allclose(first, second, atol=1e-6)


@requires_checkpoint
@pytest.mark.model
@pytest.mark.slow
def test_checkpoint_actually_loads():
    """Every tensor in the checkpoint must land in the model (strict load), and the
    model-size flag must match the checkpoint it is used with."""
    import torch

    from tool.poemkit.runner import PoemRunner

    size = next((s for s in ("small", "medium", "large", "huge") if s in CHECKPOINT), "medium")
    runner = PoemRunner(cfg_path=CFG, checkpoint=CHECKPOINT, model_size=size, device="cpu", verbose=False)

    raw = torch.load(CHECKPOINT, map_location="cpu")
    state = raw.get("state_dict", raw)
    model_state = runner.model.state_dict()
    checked = 0
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if key in model_state and value.shape == model_state[key].shape:
            assert torch.allclose(model_state[key].float(), value.float(), atol=1e-6), f"{key} was not loaded"
            checked += 1
    assert checked > 50, f"only {checked} tensors compared; the checkpoint layout may have changed"


@requires_checkpoint
@pytest.mark.model
@pytest.mark.slow
def test_checkpointed_forward_stays_in_the_trained_volume(stereo_capture):
    """With real weights the output must at least be metric and in front of the
    master camera -- ``POSITION_RANGE`` is x,y in [-0.6, 0.6], z in [0, 1.2]."""
    from tool.poemkit.runner import PoemRunner

    size = next((s for s in ("small", "medium", "large", "huge") if s in CHECKPOINT), "medium")
    runner = PoemRunner(cfg_path=CFG, checkpoint=CHECKPOINT, model_size=size, device="auto", verbose=False)
    _, packet = _packet(stereo_capture, frame_id=2)
    joints = runner.predict_packet(packet)["joints_master"]

    assert np.all(np.isfinite(joints))
    assert np.all(np.abs(joints[:, :2]) < 1.5), "x/y far outside the trained volume"
    assert np.all(joints[:, 2] > 0.0), "the hand must be in front of the master camera"
    assert np.all(joints[:, 2] < 3.0)
    # a hand is ~15-25 cm across: this catches unit errors even on synthetic imagery
    span = float(np.linalg.norm(joints.max(axis=0) - joints.min(axis=0)))
    assert 0.02 < span < 0.6, f"implausible hand span {span:.3f} m"
