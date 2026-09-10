"""poemkit: video-only inference utilities for POEM-v2.

The modules here are deliberately split so that everything that does *not*
need a GPU (calibration IO, bbox handling, crop/intrinsic math, view
packing) can be imported and unit-tested on a laptop, while the parts that
need torch / pytorch3d / MANO assets stay isolated in ``runner``.

Import policy:
  - ``calib``, ``bbox``, ``video``, ``views``  -> numpy (+cv2) only.
  - ``runner``                                 -> torch + pytorch3d + MANO.
"""
