"""T8 - the torch batch handed to the network (CPU torch is enough).

``packet_to_batch`` is the contract with ``PtEmbedMultiviewStereoV2._forward_impl``.
Everything asserted here is something that code reads directly: tensor shapes,
the ``cam_view_num`` bookkeeping that groups views into one sample, the
normalisation, and -- most importantly -- that ``target_cam_extr`` holds
**camera->master** transforms, because the model inverts it before
triangulating its reference joints.
"""

import numpy as np
import pytest

from conftest import requires_torch, requires_torchvision
from tool.poemkit.bbox import NpyBBoxProvider
from tool.poemkit.calib import load_calib_json
from tool.poemkit.geometry import project_points, transf_points
from tool.poemkit.video import MultiViewReader
from tool.poemkit.views import prepare_views

pytestmark = pytest.mark.torch


def _packet(manifest, frame_id=0, hand_side="rh"):
    rig = load_calib_json(manifest["calib"])
    provider = NpyBBoxProvider(manifest["bbox_root"])
    with MultiViewReader({n: manifest["videos"][n] for n in rig.names}, backend="cv2") as reader:
        frames = reader.read(frame_id)
    boxes = {name: provider.get(name, frame_id) for name in rig.names}
    return rig, prepare_views(frames=frames, bboxes=boxes, rig=rig, hand_side=hand_side, frame_id=frame_id)


@requires_torchvision
def test_batch_shapes_and_keys(stereo_capture):
    from tool.poemkit.views import packet_to_batch

    _, packet = _packet(stereo_capture)
    batch = packet_to_batch(packet, device="cpu")

    assert set(batch) >= {"image", "cam_serial", "cam_view_num", "target_cam_intr", "target_cam_extr",
                          "master_id", "master_serial"}
    assert tuple(batch["image"].shape) == (2, 3, 256, 256)
    assert tuple(batch["target_cam_intr"].shape) == (1, 2, 3, 3)
    assert tuple(batch["target_cam_extr"].shape) == (1, 2, 4, 4)
    assert batch["cam_serial"] == [packet.names]
    assert batch["master_serial"] == [packet.names[0]]
    assert int(batch["master_id"][0]) == 0


@requires_torchvision
def test_cam_view_num_groups_views_into_one_sample(ring_capture):
    """``BN == batch_size`` is how the model detects single-view input; with N
    views and one sample it must be N != 1."""
    from tool.poemkit.views import packet_to_batch

    _, packet = _packet(ring_capture, frame_id=0)
    batch = packet_to_batch(packet, device="cpu")

    batch_size = len(batch["cam_view_num"])
    assert batch_size == 1
    assert int(batch["cam_view_num"][0]) == packet.num_views == 3
    assert batch["image"].shape[0] == int(np.sum(batch["cam_view_num"]))
    assert batch["image"].shape[0] != batch_size, "would trigger the GT-seeded single-view path"


@requires_torchvision
def test_image_normalisation_matches_training(stereo_capture):
    """ToTensor + normalize(mean .5, std 1) -> pixels land in [-0.5, 0.5]."""
    from tool.poemkit.views import packet_to_batch

    _, packet = _packet(stereo_capture)
    batch = packet_to_batch(packet, device="cpu")
    image = batch["image"]

    assert image.dtype.is_floating_point
    assert float(image.min()) >= -0.5 - 1e-6
    assert float(image.max()) <= 0.5 + 1e-6
    # exactly recoverable from the uint8 crop
    expected = packet.crops[0].astype(np.float64) / 255.0 - 0.5
    np.testing.assert_allclose(image[0].permute(1, 2, 0).numpy(), expected, atol=1e-6)


@requires_torchvision
def test_target_cam_extr_is_camera_to_master(stereo_capture):
    """The model does ``inv(target_cam_extr)`` and treats the result as world->camera
    for DLT, with the master as world. Verify both directions on real geometry."""
    from tool.poemkit.views import packet_to_batch

    rig, packet = _packet(stereo_capture, frame_id=1)
    batch = packet_to_batch(packet, device="cpu")
    T_mc = batch["target_cam_extr"][0].numpy().astype(np.float64)

    np.testing.assert_allclose(T_mc[0], np.eye(4), atol=1e-6)

    point_world = np.array([[0.01, -0.02, 0.6]])
    in_master = transf_points(rig[packet.names[0]].T_cw, point_world)
    for i, name in enumerate(packet.names):
        in_cam = transf_points(rig[name].T_cw, point_world)
        np.testing.assert_allclose(transf_points(T_mc[i], in_cam), in_master, atol=1e-5)
        # inv(T_mc) is master->camera, which is what the model feeds to DLT
        np.testing.assert_allclose(transf_points(np.linalg.inv(T_mc[i]), in_master), in_cam, atol=1e-5)


@requires_torch
def test_model_reference_triangulation_reproduces_the_truth(stereo_capture, stereo_gt):
    """Replay the model's own reference-joint step: ``batch_triangulate_dlt_torch``
    on crop-space 2D keypoints and ``inv(target_cam_extr)`` must return the GT
    joints in the master camera frame."""
    import torch

    from lib.utils.triangulation import batch_triangulate_dlt_torch

    rig, packet = _packet(stereo_capture, frame_id=2)
    gt_world = stereo_gt["joints_world"][2]

    uv_crop = np.stack([
        project_points(packet.K_crop[i], transf_points(rig[name].T_cw, gt_world))
        for i, name in enumerate(packet.names)
    ], axis=0)

    ref = batch_triangulate_dlt_torch(
        torch.as_tensor(uv_crop, dtype=torch.float64)[None],
        torch.as_tensor(packet.K_crop, dtype=torch.float64)[None],
        torch.as_tensor(np.linalg.inv(packet.T_mc), dtype=torch.float64)[None],
    )[0].numpy()

    expected_master = transf_points(rig[packet.names[0]].T_cw, gt_world)
    np.testing.assert_allclose(ref, expected_master, atol=1e-6)


@requires_torchvision
def test_left_hand_batch_uses_mirrored_extrinsics(left_hand_capture):
    from tool.poemkit.views import packet_to_batch

    rig, packet = _packet(left_hand_capture, frame_id=1, hand_side="lh")
    batch = packet_to_batch(packet, device="cpu")

    assert packet.flipped
    np.testing.assert_allclose(batch["target_cam_extr"][0][0].numpy(), np.eye(4), atol=1e-6)
    # the mirrored right camera ends up on the opposite side of the master
    center_used = np.linalg.inv(packet.T_cw_used[1])[:3, 3]
    center_orig = rig[packet.names[1]].center
    assert np.sign(center_used[0]) == -np.sign(center_orig[0])


@requires_torchvision
def test_dtypes_are_float32(stereo_capture):
    import torch

    from tool.poemkit.views import packet_to_batch

    _, packet = _packet(stereo_capture)
    batch = packet_to_batch(packet, device="cpu")
    assert batch["image"].dtype == torch.float32
    assert batch["target_cam_intr"].dtype == torch.float32
    assert batch["target_cam_extr"].dtype == torch.float32
