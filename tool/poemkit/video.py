"""Video reading/writing for multi-view inference.

Two readers, one interface (``__len__`` + ``__getitem__`` -> RGB uint8 HWC):

``cv2``     ``cv2.VideoCapture``; sequential reads are cheap, random access is
            emulated by seeking. Works anywhere OpenCV works, including macOS.
``ffmpeg``  the repo's :class:`video_tool.ffmpeg_util.FFMPEGFrameLoader`, which
            decodes a windowed cache via ffmpeg pipes. Needs ffmpeg-python plus
            an ffmpeg binary and is what the released demo uses for ``.mkv``.

Views are read strictly by frame index, so the videos must be frame-synchronised
(same start instant, same frame rate, same length). Any drift shows up as a
constant-offset error in the 3D output, not as a crash -- see
``check_alignment``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["FrameReader", "CV2FrameReader", "FFmpegFrameReader", "open_reader", "MultiViewReader", "VideoWriter"]


class FrameReader:
    """Interface for a decoded RGB frame sequence."""

    num_frame: int = 0
    frame_size: Sequence[int] = (0, 0)  # (w, h)
    fps: float = 0.0

    def __len__(self) -> int:
        return self.num_frame

    def __getitem__(self, frame_id: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CV2FrameReader(FrameReader):
    """OpenCV-backed reader. Returns RGB (OpenCV decodes BGR)."""

    def __init__(self, path: str):
        import cv2

        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self._cv2 = cv2
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open {path}")
        self.num_frame = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self._next_id = 0

    def __getitem__(self, frame_id: int) -> np.ndarray:
        if frame_id < 0 or (self.num_frame > 0 and frame_id >= self.num_frame):
            raise IndexError(f"frame_id out of range: {frame_id} of [0, {self.num_frame})")
        if frame_id != self._next_id:
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame_bgr = self._cap.read()
        if not ok:
            raise RuntimeError(f"failed to decode frame {frame_id} of {self.path}")
        self._next_id = frame_id + 1
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self):
        self._cap.release()


class FFmpegFrameReader(FrameReader):
    """Wrapper over the repo's ffmpeg frame loader (used by the released demo)."""

    def __init__(self, path: str, cache_size: int = 64):
        from video_tool.ffmpeg_util import FFMPEGFrameLoader

        self._loader = FFMPEGFrameLoader(path, pix_fmt="rgb24", cache_size=cache_size)
        self.num_frame = self._loader.num_frame
        self.frame_size = self._loader.frame_size
        self.fps = 0.0

    def __getitem__(self, frame_id: int) -> np.ndarray:
        return np.asarray(self._loader[frame_id])


def open_reader(path: str, backend: str = "auto", cache_size: int = 64) -> FrameReader:
    """Open ``path`` with the requested backend; ``auto`` prefers ffmpeg, falls back to cv2."""
    backend = backend.lower()
    if backend == "cv2":
        return CV2FrameReader(path)
    if backend == "ffmpeg":
        return FFmpegFrameReader(path, cache_size=cache_size)
    if backend == "auto":
        try:
            return FFmpegFrameReader(path, cache_size=cache_size)
        except Exception:
            return CV2FrameReader(path)
    raise ValueError(f"unknown video backend {backend!r}; expected auto|cv2|ffmpeg")


class MultiViewReader:
    """Frame-index-synchronised access to one video per camera."""

    def __init__(self, video_paths: Dict[str, str], backend: str = "auto", cache_size: int = 64):
        if len(video_paths) == 0:
            raise ValueError("no videos given")
        self.readers: Dict[str, FrameReader] = {
            name: open_reader(path, backend=backend, cache_size=cache_size) for name, path in video_paths.items()
        }
        self.names: List[str] = list(video_paths.keys())

    @property
    def num_frame(self) -> int:
        """Length of the shortest view: frames beyond it exist in no full set."""
        return min(len(r) for r in self.readers.values())

    def frame_size(self, name: Optional[str] = None):
        return self.readers[name if name is not None else self.names[0]].frame_size

    def check_alignment(self) -> List[str]:
        """Warn about mismatched lengths / resolutions across views."""
        problems = []
        lengths = {n: len(r) for n, r in self.readers.items()}
        if len(set(lengths.values())) > 1:
            problems.append(f"views have different frame counts {lengths}; using the shortest ({self.num_frame}). "
                            "Un-synchronised views bias the 3D output without any error being raised.")
        sizes = {n: tuple(r.frame_size) for n, r in self.readers.items()}
        if len(set(sizes.values())) > 1:
            problems.append(f"views have different resolutions {sizes}; each K must match its own video")
        rates = {n: round(r.fps, 2) for n, r in self.readers.items() if r.fps}
        if len(set(rates.values())) > 1:
            problems.append(f"views have different frame rates {rates}")
        return problems

    def read(self, frame_id: int) -> Dict[str, np.ndarray]:
        return {name: reader[frame_id] for name, reader in self.readers.items()}

    def close(self):
        for reader in self.readers.values():
            reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class VideoWriter:
    """Lazy mp4 writer for overlay debug output (BGR in, per OpenCV)."""

    def __init__(self, path: str, fps: float = 30.0, fourcc: str = "mp4v"):
        import cv2

        self._cv2 = cv2
        self.path = path
        self.fps = fps if fps and fps > 0 else 30.0
        self.fourcc = fourcc
        self._writer = None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write_bgr(self, frame_bgr: np.ndarray):
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self._writer = self._cv2.VideoWriter(self.path, self._cv2.VideoWriter_fourcc(*self.fourcc), self.fps,
                                                 (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot open VideoWriter for {self.path}")
        self._writer.write(frame_bgr)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
