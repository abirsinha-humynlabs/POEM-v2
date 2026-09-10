"""Pack synchronised multi-view frames into a POEM-v2 batch (and unpack results).

This is a refactor of ``format_batch`` / ``extract_pred`` in
``tool/infer_hand.py`` with three differences that matter for video-only use:

1. **numpy first.** All geometry happens in numpy (:class:`ViewPacket`);
   torch is touched only in :func:`packet_to_batch`. So the crop math,
   left-hand mirroring and master-frame bookkeeping are testable on a laptop
   with no torch/CUDA/pytorch3d present.
2. **No GUI.** The original calls ``cv2.imshow`` per view inside the packing
   function; here nothing is drawn.
3. **Explicit view-count contract.** A frame with fewer than ``min_views``
   boxed views is refused with a reason, instead of returning ``None``.
   Single-view input is a hard error, not a fallback: with one view
   ``lib/models/POEM.py`` seeds its reference joints from
   ``batch["master_joints_3d"]`` -- ground truth -- which does not exist at
   inference time.

Frame conventions
-----------------
* Input frames are **RGB uint8** ``(H, W, 3)``.
* The model input is a ``256x256`` crop, ``ToTensor`` then normalised with
  mean 0.5 / std 1.0 (matching training).
* View 0 of the packet is the master; predictions come back in its camera
  frame and are re-expressed in the rig's world frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .calib import CameraRig, inv_se3
from .geometry import (
    affine_crop_transform,
    apply_affine_2d,
    bbox_get_center_scale,
    intrinsics_after_crop,
    mirror_points_x,
    project_points,
    transf_points,
)

__all__ = [
    "ViewPacket",
    "PackError",
    "prepare_views",
    "packet_to_batch",
    "unpack_prediction",
    "project_to_views",
    "joints_uv_to_original",
]

IMAGENET_LIKE_MEAN = (0.5, 0.5, 0.5)
IMAGENET_LIKE_STD = (1.0, 1.0, 1.0)
MIN_VIEWS = 2


class PackError(RuntimeError):
    """Raised when a frame cannot be turned into a valid multi-view batch."""


def _flip_extrinsic_world_x(T_cw: np.ndarray) -> np.ndarray:
    """Mirror a world->camera extrinsic across the world YZ plane.

    Delegates to :func:`tool.flip_util.flip_cam_extr` (which works on the
    camera->world matrix) so the left-hand path is bit-identical to the
    released demo.
    """
    from tool.flip_util import flip_cam_extr

    return inv_se3(flip_cam_extr(inv_se3(np.asarray(T_cw, dtype=np.float64))))


class ViewPacket:
    """Everything about one frame's usable views, in numpy.

    Attributes:
        names: camera names, ordered; ``names[0]`` is the master.
        crops: ``(N, out_h, out_w, 3)`` uint8 RGB crops fed to the network.
        K_crop: ``(N, 3, 3)`` intrinsics *of the crops*.
        T_cw_used: ``(N, 4, 4)`` world->camera actually used (mirrored for a left hand).
        T_mc: ``(N, 4, 4)`` camera->master transforms; ``T_mc[0]`` is identity.
        centers / scales: crop windows in original-image pixels.
        flipped: whether the world-mirroring (left-hand) path was applied.
    """

    def __init__(
        self,
        names: Sequence[str],
        crops: np.ndarray,
        K_crop: np.ndarray,
        T_cw_used: np.ndarray,
        centers: np.ndarray,
        scales: np.ndarray,
        flipped: bool,
        frame_id: int = -1,
        dropped: Optional[Dict[str, str]] = None,
    ):
        self.names = list(names)
        self.crops = crops
        self.K_crop = K_crop
        self.T_cw_used = T_cw_used
        self.centers = centers
        self.scales = scales
        self.flipped = bool(flipped)
        self.frame_id = int(frame_id)
        self.dropped = dropped or {}

        T_wc = np.stack([inv_se3(T) for T in self.T_cw_used], axis=0)  # camera->world
        T_cw_master = self.T_cw_used[0]
        self.T_mc = np.stack([T_cw_master @ T for T in T_wc], axis=0)  # camera->master

    @property
    def num_views(self) -> int:
        return len(self.names)

    def master_to_world(self, points_master: np.ndarray) -> np.ndarray:
        """Master-camera-frame points -> rig world frame (undoing any mirroring)."""
        pts = transf_points(inv_se3(self.T_cw_used[0]), points_master)
        return mirror_points_x(pts) if self.flipped else pts

    def __repr__(self) -> str:
        return f"ViewPacket(frame={self.frame_id}, views={self.names}, flipped={self.flipped})"


def prepare_views(
    frames: Dict[str, np.ndarray],
    bboxes: Dict[str, Optional[np.ndarray]],
    rig: CameraRig,
    hand_side: str = "rh",
    out_res: Sequence[int] = (256, 256),
    bbox_expand: float = 2.0,
    bbox_mindim: float = 200.0,
    frame_id: int = -1,
    min_views: int = MIN_VIEWS,
    master: Optional[str] = None,
) -> ViewPacket:
    """Crop, mirror (if left hand) and pack the views that have a bbox.

    Raises:
        PackError: if fewer than ``min_views`` views survive.
    """
    import cv2

    if hand_side not in ("rh", "lh"):
        raise ValueError(f"hand_side must be 'rh' or 'lh', got {hand_side!r}")
    req_flip = hand_side == "lh"
    out_w, out_h = int(out_res[0]), int(out_res[1])

    order = list(rig.names)
    if master is not None:
        if master not in order:
            raise ValueError(f"master {master!r} not in rig {order}")
        order = [master] + [n for n in order if n != master]

    names: List[str] = []
    crops: List[np.ndarray] = []
    K_list: List[np.ndarray] = []
    T_list: List[np.ndarray] = []
    centers: List[np.ndarray] = []
    scales: List[float] = []
    dropped: Dict[str, str] = {}

    for cam_name in order:
        if cam_name not in frames or frames[cam_name] is None:
            dropped[cam_name] = "no frame"
            continue
        bbox = bboxes.get(cam_name)
        if bbox is None:
            dropped[cam_name] = "no bbox"
            continue
        bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if bbox.shape[0] < 4 or not np.all(np.isfinite(bbox[:4])):
            dropped[cam_name] = "malformed bbox"
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            dropped[cam_name] = "degenerate bbox"
            continue

        cam = rig[cam_name]
        img = frames[cam_name]
        img_h, img_w = img.shape[:2]
        center, scale = bbox_get_center_scale(bbox, expand=bbox_expand, mindim=bbox_mindim)
        K_full = cam.K.copy()
        T_cw = cam.T_cw.copy()

        if req_flip:
            # world-mirroring: flip the image about the principal point, mirror
            # the bbox center with it, and mirror the extrinsics accordingly.
            cx = float(K_full[0, 2])
            center[0] = 2.0 * cx - center[0]
            M = np.array([[-1.0, 0.0, 2.0 * cx], [0.0, 1.0, 0.0]], dtype=np.float32)
            img = cv2.warpAffine(img, M, (img_w, img_h))
            T_cw = _flip_extrinsic_world_x(T_cw)

        affine = affine_crop_transform(center, scale, (out_w, out_h))
        crop = cv2.warpAffine(img, affine[:2, :].astype(np.float32), (out_w, out_h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        K_crop = intrinsics_after_crop(K_full, center, scale, (out_w, out_h), cam.optical_center)

        names.append(cam_name)
        crops.append(crop)
        K_list.append(K_crop)
        T_list.append(T_cw)
        centers.append(center)
        scales.append(float(scale))

    if len(names) < min_views:
        raise PackError(f"frame {frame_id}: only {len(names)} usable view(s) "
                        f"(need >= {min_views}); dropped={dropped}")

    return ViewPacket(
        names=names,
        crops=np.stack(crops, axis=0),
        K_crop=np.stack(K_list, axis=0),
        T_cw_used=np.stack(T_list, axis=0),
        centers=np.stack(centers, axis=0),
        scales=np.asarray(scales, dtype=np.float64),
        flipped=req_flip,
        frame_id=frame_id,
        dropped=dropped,
    )


def packet_to_batch(packet: ViewPacket, device="cpu") -> Dict:
    """Convert a :class:`ViewPacket` into the dict ``PtEmbedMultiviewStereoV2`` expects.

    Batch size is 1: ``image`` is ``(N, 3, H, W)`` for the N views of this
    frame, and ``cam_view_num`` tells the model they all belong to one sample.
    ``target_cam_extr`` holds **camera->master** transforms (the name is
    inherited from the training code, where it is inverted again inside
    ``_forward_impl``).
    """
    import torch
    import torchvision.transforms.functional as tvF

    images = []
    for crop in packet.crops:
        image = tvF.to_tensor(np.ascontiguousarray(crop))
        if image.shape[0] != 3:
            raise ValueError(f"expected a 3-channel crop, got {tuple(image.shape)}")
        images.append(tvF.normalize(image, list(IMAGENET_LIKE_MEAN), list(IMAGENET_LIKE_STD)))

    image_th = torch.stack(images, dim=0).to(device)
    cam_intr_th = torch.as_tensor(packet.K_crop, dtype=torch.float32).to(device)
    cam_mc_th = torch.as_tensor(packet.T_mc, dtype=torch.float32).to(device)

    return {
        "image": image_th,  # (N, 3, H, W)
        "cam_serial": [list(packet.names)],
        "cam_view_num": np.array([packet.num_views]),  # (1,)
        "target_cam_intr": cam_intr_th[None],  # (1, N, 3, 3)
        "target_cam_extr": cam_mc_th[None],  # (1, N, 4, 4) camera->master
        "master_id": torch.as_tensor([0]).to(device),
        "master_serial": [packet.names[0]],
    }


def unpack_prediction(pred: Dict, packet: ViewPacket) -> Dict[str, np.ndarray]:
    """Pull joints/verts out of a model output dict and express them in world coords.

    Returns numpy arrays:
        ``joints_master`` (21,3), ``verts_master`` (778,3) -- master camera frame
        ``joints_world``  (21,3), ``verts_world``  (778,3) -- rig world frame
        ``joints_uv``     (N,21,2) -- per-view 2D heatmap keypoints, in crop pixels
    """
    def _np(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    for key in ("pred_joints_3d", "pred_verts_3d"):
        if key not in pred:
            raise KeyError(f"model output has no {key!r}; got {sorted(pred.keys())}")

    joints_master = _np(pred["pred_joints_3d"])[0].astype(np.float64)
    verts_master = _np(pred["pred_verts_3d"])[0].astype(np.float64)

    out = {
        "joints_master": joints_master,
        "verts_master": verts_master,
        "joints_world": packet.master_to_world(joints_master),
        "verts_world": packet.master_to_world(verts_master),
    }
    if "pred_joints_uv" in pred:
        out["joints_uv"] = _np(pred["pred_joints_uv"]).astype(np.float64)
    return out


def joints_uv_to_original(packet: ViewPacket, joints_uv: np.ndarray, rig: CameraRig) -> Dict[str, np.ndarray]:
    """Map the network's 2D keypoints from crop pixels back to original-frame pixels.

    ``pred_joints_uv`` comes out of the heatmap stage in *crop* coordinates
    (0..256). Undoing the crop affine puts them back on the full frame; for a
    left hand the frame itself was mirrored about the principal point, so that
    mirroring is undone too. Comparing these against the reprojected 3D joints
    is the one end-to-end consistency signal available without ground truth.
    """
    joints_uv = np.asarray(joints_uv, dtype=np.float64)
    if joints_uv.shape[0] != packet.num_views:
        raise ValueError(f"joints_uv has {joints_uv.shape[0]} views, packet has {packet.num_views}")

    out: Dict[str, np.ndarray] = {}
    out_res = (packet.crops.shape[2], packet.crops.shape[1])  # (w, h)
    for i, name in enumerate(packet.names):
        affine = affine_crop_transform(packet.centers[i], packet.scales[i], out_res)
        uv = apply_affine_2d(np.linalg.inv(affine), joints_uv[i])
        if packet.flipped:
            uv = uv.copy()
            uv[..., 0] = 2.0 * float(rig[name].K[0, 2]) - uv[..., 0]
        out[name] = uv
    return out


def project_to_views(points_world: np.ndarray, rig: CameraRig,
                     names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """Project world-frame points into each (original, uncropped) view."""
    names = list(rig.names) if names is None else list(names)
    out = {}
    for name in names:
        cam = rig[name]
        out[name] = project_points(cam.K, transf_points(cam.T_cw, points_world))
    return out
