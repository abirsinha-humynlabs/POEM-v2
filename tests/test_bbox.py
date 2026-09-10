"""T3 - hand boxes: the one input POEM-v2 needs that video alone does not give.

A frame is usable only if >= 2 views have a box, so box handling decides how
much of a sequence survives. These tests cover each backend, the reuse
("tracking") policy for dropped detections, and the failure shapes.
"""

import json

import numpy as np
import pytest

from tool.poemkit.bbox import (
    BBoxTracker,
    FullFrameBBoxProvider,
    JsonBBoxProvider,
    NpyBBoxProvider,
    bbox_from_points,
    build_bbox_provider,
    clip_bbox,
)


class _ScriptedProvider:
    """Returns whatever the script says for each (cam, frame): a box or None."""

    name = "scripted"

    def __init__(self, script):
        self.script = script
        self.calls = []

    def get(self, cam_name, frame_id, frame=None):
        self.calls.append((cam_name, frame_id))
        return self.script.get((cam_name, frame_id))

    def close(self):
        pass


def test_bbox_from_points_is_padded_and_ordered():
    pts = np.array([[100.0, 200.0], [140.0, 260.0]])
    box = bbox_from_points(pts, pad_ratio=0.0)
    np.testing.assert_allclose(box, [100.0, 200.0, 140.0, 260.0])

    padded = bbox_from_points(pts, pad_ratio=0.5)
    pad = 0.5 * 60.0  # larger side is 60 px
    np.testing.assert_allclose(padded, [100 - pad, 200 - pad, 140 + pad, 260 + pad])
    assert padded[0] < padded[2] and padded[1] < padded[3]


def test_bbox_from_points_rejects_empty():
    with pytest.raises(ValueError):
        bbox_from_points(np.zeros((0, 2)))


def test_clip_bbox_stays_inside_the_frame():
    box = clip_bbox([-50.0, -20.0, 5000.0, 4000.0], (640, 480))
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= 640 and box[3] <= 480


def test_npy_backend_reads_released_layout(ring_capture):
    provider = NpyBBoxProvider(ring_capture["bbox_root"])
    box = provider.get("cam0", 0)
    assert box is not None and box.shape == (4,)
    assert box[2] > box[0] and box[3] > box[1]
    # cam1 has no file on frame 2 (simulated detector failure)
    assert provider.get("cam1", 2) is None
    assert provider.get("cam0", 10_000) is None, "a missing frame must return None, not raise"


def test_npy_backend_takes_the_first_row_of_a_multi_box_file(tmp_path):
    cam_dir = tmp_path / "camX" / "bbox"
    cam_dir.mkdir(parents=True)
    np.save(cam_dir / "00000.npy", np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32))
    np.save(cam_dir / "00001.npy", np.zeros((0, 4), dtype=np.float32))

    provider = NpyBBoxProvider(str(tmp_path))
    np.testing.assert_allclose(provider.get("camX", 0), [10, 20, 30, 40])
    assert provider.get("camX", 1) is None, "an empty array means no detection"


def test_json_backend_roundtrip(tmp_path, ring_capture):
    provider = JsonBBoxProvider(ring_capture["bbox_json"])
    assert provider.get("cam0", 0) is not None
    assert provider.get("cam1", 2) is None, "explicit null must be reported as a miss"
    assert provider.get("nosuchcam", 0) is None

    path = tmp_path / "b.json"
    path.write_text(json.dumps({"cam0": {"0": [1, 2, 3, 4]}}))  # bare mapping, no "boxes" wrapper
    np.testing.assert_allclose(JsonBBoxProvider(str(path)).get("cam0", 0), [1, 2, 3, 4])


def test_full_frame_backend_uses_the_frame_shape():
    provider = FullFrameBBoxProvider(image_size=(640, 480))
    np.testing.assert_allclose(provider.get("cam", 0), [0, 0, 640, 480])
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    np.testing.assert_allclose(provider.get("cam", 0, frame), [0, 0, 200, 120])


def test_tracker_reuses_the_last_box_within_max_age():
    script = {("cam", 0): np.array([1.0, 2.0, 3.0, 4.0])}  # frames 1..n miss
    tracker = BBoxTracker(_ScriptedProvider(script), max_age=2)

    np.testing.assert_allclose(tracker.get("cam", 0), [1, 2, 3, 4])
    np.testing.assert_allclose(tracker.get("cam", 1), [1, 2, 3, 4])  # age 1: reused
    np.testing.assert_allclose(tracker.get("cam", 2), [1, 2, 3, 4])  # age 2: reused
    assert tracker.get("cam", 3) is None, "beyond max_age the view must be dropped"
    assert tracker.stats == {"hit": 1, "reused": 2, "miss": 1}


def test_tracker_reuse_can_be_disabled():
    tracker = BBoxTracker(_ScriptedProvider({("cam", 0): np.array([1.0, 2, 3, 4])}), max_age=0)
    assert tracker.get("cam", 0) is not None
    assert tracker.get("cam", 1) is None


def test_tracker_keeps_cameras_independent():
    script = {("a", 0): np.array([1.0, 1, 2, 2]), ("b", 1): np.array([3.0, 3, 4, 4])}
    tracker = BBoxTracker(_ScriptedProvider(script), max_age=3)
    tracker.get("a", 0)
    assert tracker.get("b", 0) is None, "camera b has never been seen yet"
    np.testing.assert_allclose(tracker.get("a", 1), [1, 1, 2, 2])
    np.testing.assert_allclose(tracker.get("b", 1), [3, 3, 4, 4])


def test_tracker_resets_age_after_a_fresh_detection():
    script = {("cam", 0): np.array([1.0, 1, 2, 2]), ("cam", 2): np.array([5.0, 5, 6, 6])}
    tracker = BBoxTracker(_ScriptedProvider(script), max_age=1)
    tracker.get("cam", 0)
    tracker.get("cam", 1)  # reused
    np.testing.assert_allclose(tracker.get("cam", 2), [5, 5, 6, 6])
    np.testing.assert_allclose(tracker.get("cam", 3), [5, 5, 6, 6]), "age must have reset"


def test_factory_validates_its_arguments(ring_capture):
    assert isinstance(build_bbox_provider("npy", bbox_root=ring_capture["bbox_root"]), BBoxTracker)
    assert isinstance(build_bbox_provider("json", bbox_json=ring_capture["bbox_json"]), BBoxTracker)
    assert isinstance(build_bbox_provider("full", image_size=(640, 480)), BBoxTracker)

    with pytest.raises(ValueError, match="--bbox-root"):
        build_bbox_provider("npy")
    with pytest.raises(ValueError, match="--bbox-json"):
        build_bbox_provider("json")
    with pytest.raises(ValueError, match="unknown bbox backend"):
        build_bbox_provider("magic")


def test_synthetic_boxes_actually_contain_the_hand(ring_capture, tmp_path):
    """Sanity-check the fixture itself: every stored box contains its GT keypoints."""
    gt = np.load(ring_capture["gt"])
    provider = NpyBBoxProvider(ring_capture["bbox_root"])
    for ci, cam in enumerate(gt["cameras"].tolist()):
        for frame in range(int(gt["num_frames"])):
            box = provider.get(cam, frame)
            if box is None:
                continue
            uv = gt["joints_2d"][ci, frame]
            # boxes carry a few px of jitter, so allow a small margin
            assert uv[:, 0].min() >= box[0] - 8 and uv[:, 0].max() <= box[2] + 8
            assert uv[:, 1].min() >= box[1] - 8 and uv[:, 1].max() <= box[3] + 8


def test_mediapipe_backend_reports_a_clear_error_when_absent():
    """MediaPipe is optional; asking for it without it installed must say so."""
    pytest.importorskip("numpy")
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="mediapipe"):
            build_bbox_provider("mediapipe")
    else:
        provider = build_bbox_provider("mediapipe")
        assert provider.provider.name == "mediapipe"
        with pytest.raises(ValueError, match="needs the RGB frame"):
            provider.get("cam", 0, None)
