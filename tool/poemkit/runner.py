"""Model construction + a device-agnostic prediction wrapper.

Everything torch-heavy lives here. Importing this module pulls in torch,
pytorch3d (via ``lib.models``) and manotorch, and constructing
:class:`PoemRunner` also needs the MANO assets in ``assets/mano_v1_2``. Use
:func:`environment_report` to find out what is missing *before* paying for a
model build -- that is what makes the failure modes legible on a laptop.

Device notes
------------
* ``cuda`` is the supported path (what the released scripts hardcode).
* ``cpu`` works because the model code is written against ``x.device``
  throughout; it is only useful for shape/plumbing checks -- expect seconds
  per frame, not fps.
* ``mps`` is accepted but unverified: pytorch3d's ``knn_points`` / ``ball_query``
  have no Metal kernels, so it will most likely fall back or fail. Kept
  because it costs nothing and makes the failure explicit rather than silent.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "MODEL_CATEGORY",
    "EMBED_SIZE",
    "environment_report",
    "resolve_device",
    "build_cfg",
    "PoemRunner",
]

# Mirrors scripts/eval_single.py: the model size only changes the embedding width.
MODEL_CATEGORY: List[str] = ["small", "medium", "large", "huge", "medium_MANO"]
EMBED_SIZE: List[int] = [128, 256, 512, 1024, 256]

_REQUIRED_MODULES = ("torch", "torchvision", "pytorch3d", "manotorch", "yacs", "cv2", "numpy")
_MANO_ASSET_DIR = os.path.join("assets", "mano_v1_2")


def environment_report(repo_root: str = ".") -> Dict[str, object]:
    """What is present / missing for a real inference run.

    Returns a dict with ``modules`` (name -> version or None), ``missing``,
    ``mano_assets``, ``cuda`` and ``devices``.
    """
    modules: Dict[str, Optional[str]] = {}
    for name in _REQUIRED_MODULES:
        if importlib.util.find_spec(name) is None:
            modules[name] = None
            continue
        try:
            mod = importlib.import_module(name)
            modules[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:  # import present but broken (e.g. missing CUDA libs)
            modules[name] = f"import failed: {type(exc).__name__}"

    cuda_available = False
    devices: List[str] = ["cpu"]
    if modules.get("torch") and not str(modules["torch"]).startswith("import failed"):
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices.append("mps")

    mano_dir = os.path.join(repo_root, _MANO_ASSET_DIR)
    return {
        "modules": modules,
        "missing": [k for k, v in modules.items() if v is None],
        "mano_assets": os.path.isdir(mano_dir),
        "mano_assets_path": mano_dir,
        "cuda": cuda_available,
        "devices": devices,
    }


def resolve_device(spec: str = "auto"):
    """Turn ``auto|cpu|cuda|cuda:N|mps`` into a ``torch.device``."""
    import torch

    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            spec = "cuda:0"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            spec = "mps"
        else:
            spec = "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False. "
                           "Run on a Linux box with an NVIDIA GPU, or pass --device cpu for a "
                           "plumbing-only (very slow) run.")
    return torch.device(spec)


def build_cfg(
    cfg_path: str,
    model_size: str = "medium",
    checkpoint: Optional[str] = None,
    backbone_pretrained: Optional[str] = None,
):
    """Load a release config and apply the model-size / checkpoint overrides.

    ``scripts/eval_single.py`` performs the same overrides by rewriting the
    yaml in place; doing it in memory keeps the repo's configs pristine.

    Args:
        backbone_pretrained: path to the HRNet ImageNet weights, or ``""`` to
            skip that load. Defaults to ``""`` when ``checkpoint`` is given,
            since a full POEM checkpoint already contains backbone weights
            (and the HRNet loader calls ``torch.load`` without ``map_location``,
            which can break on a CPU-only host).
    """
    from lib.utils.config import get_config

    if model_size not in MODEL_CATEGORY:
        raise ValueError(f"unknown model size {model_size!r}; expected one of {MODEL_CATEGORY}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    cfg = get_config(cfg_path, arg=None, merge=True)
    embed = EMBED_SIZE[MODEL_CATEGORY.index(model_size)]

    cfg.defrost()
    cfg.MODEL.HEAD.POSITIONAL_ENCODING.NUM_FEATS = embed // 2
    cfg.MODEL.HEAD.TRANSFORMER.INPUT_FEAT_DIM = embed
    cfg.MODEL.HEAD.POINTS_FEAT_DIM = embed
    cfg.MODEL.HEAD.EMBED_DIMS = embed
    cfg.MODEL.HEAD.TRANSFORMER.PARAMETRIC_OUTPUT = (model_size == "medium_MANO")
    if checkpoint is not None:
        cfg.MODEL.PRETRAINED = checkpoint
    if backbone_pretrained is None:
        backbone_pretrained = "" if checkpoint else cfg.MODEL.BACKBONE.PRETRAINED
    cfg.MODEL.BACKBONE.PRETRAINED = backbone_pretrained
    cfg.freeze()
    return cfg


class PoemRunner:
    """Builds the model once, then maps a :class:`~tool.poemkit.views.ViewPacket` to predictions."""

    def __init__(
        self,
        cfg_path: str = os.path.join("config", "release", "eval_single.yaml"),
        checkpoint: Optional[str] = None,
        model_size: str = "medium",
        device: str = "auto",
        backbone_pretrained: Optional[str] = None,
        verbose: bool = True,
    ):
        import torch

        import lib.models  # noqa: F401  (registers the model classes)
        from lib.external import EXT_PACKAGE
        from lib.utils import builder

        self.cfg = build_cfg(cfg_path, model_size=model_size, checkpoint=checkpoint,
                             backbone_pretrained=backbone_pretrained)
        self.device = resolve_device(device)
        self.model_size = model_size
        self.checkpoint = checkpoint
        self._torch = torch

        if self.cfg.MODEL.TYPE in EXT_PACKAGE:  # parity with the released scripts
            importlib.import_module(f"lib.external.{EXT_PACKAGE[self.cfg.MODEL.TYPE]}")

        self.model = builder.build_model(self.cfg.MODEL, data_preset=self.cfg.DATA_PRESET, train=self.cfg.TRAIN)
        self.model.setup(summary_writer=None, log_freq=100)
        self.model.to(self.device)
        self.model.eval()

        self.num_joints = int(self.cfg.DATA_PRESET.NUM_JOINTS)
        self.image_size = tuple(int(v) for v in self.cfg.DATA_PRESET.IMAGE_SIZE)
        self.faces = self.model.face.detach().cpu().numpy()
        if verbose:
            import sys

            print(f"[poemkit] model={self.cfg.MODEL.TYPE} size={model_size} device={self.device} "
                  f"ckpt={checkpoint or 'NONE (random weights)'} input={self.image_size}", file=sys.stderr)

    @property
    def hand_faces(self) -> np.ndarray:
        """MANO triangle faces ``(1538, 3)``, for meshing the 778 vertices."""
        return self.faces

    def predict_packet(self, packet) -> Dict[str, np.ndarray]:
        """Run one frame's views and return world/master-frame joints + verts."""
        from .views import packet_to_batch, unpack_prediction

        if packet.num_views < 2:
            raise ValueError("POEM-v2 inference needs >= 2 views in the batch; "
                             "single-view forward seeds reference joints from ground truth")
        batch = packet_to_batch(packet, device=self.device)
        with self._torch.no_grad():
            pred = self.model(batch, 0, "inference", epoch_idx=0)
        return unpack_prediction(pred, packet)

    def predict_batch_dict(self, batch: Dict) -> Dict:
        """Escape hatch: run a hand-built batch dict (used by the smoke tests)."""
        with self._torch.no_grad():
            return self.model(batch, 0, "inference", epoch_idx=0)
