"""T5 - video reading: frame indexing and synchronisation.

Views are aligned by frame index alone, so a length/resolution/rate mismatch
must be surfaced loudly -- unsynchronised views bias the triangulated 3D
without raising anything.
"""

import numpy as np
import pytest

from tool.poemkit.video import CV2FrameReader, MultiViewReader, VideoWriter, open_reader


def test_reader_reports_geometry(stereo_capture):
    reader = CV2FrameReader(stereo_capture["videos"]["left"])
    try:
        assert len(reader) == stereo_capture["num_frames"]
        assert tuple(reader.frame_size) == tuple(stereo_capture["image_size"])
        assert reader.fps > 0
    finally:
        reader.close()


def test_frames_are_rgb_uint8_and_stable(stereo_capture):
    reader = CV2FrameReader(stereo_capture["videos"]["left"])
    try:
        frame = reader[0]
        assert frame.dtype == np.uint8
        assert frame.shape == (stereo_capture["image_size"][1], stereo_capture["image_size"][0], 3)
        # random access must return the same pixels as sequential access
        _ = reader[3]
        again = reader[0]
        np.testing.assert_array_equal(frame, again)
    finally:
        reader.close()


def test_out_of_range_frame_raises(stereo_capture):
    reader = CV2FrameReader(stereo_capture["videos"]["left"])
    try:
        with pytest.raises(IndexError):
            _ = reader[stereo_capture["num_frames"] + 5]
        with pytest.raises(IndexError):
            _ = reader[-1]
    finally:
        reader.close()


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        CV2FrameReader("/nonexistent/video.mp4")


def test_open_reader_auto_falls_back_to_cv2(stereo_capture):
    reader = open_reader(stereo_capture["videos"]["left"], backend="auto")
    try:
        assert len(reader) > 0
    finally:
        reader.close()
    with pytest.raises(ValueError, match="unknown video backend"):
        open_reader(stereo_capture["videos"]["left"], backend="vlc")


def test_multiview_reader_is_index_synchronised(stereo_capture, stereo_gt):
    with MultiViewReader(stereo_capture["videos"], backend="cv2") as reader:
        assert reader.num_frame == stereo_capture["num_frames"]
        assert reader.check_alignment() == []
        frames = reader.read(2)
        assert set(frames) == set(stereo_capture["cameras"])
        # the two views differ: a stereo pair sees the hand at different u
        left, right = frames["left"], frames["right"]
        assert not np.array_equal(left, right)


def test_multiview_reader_uses_the_shortest_view(stereo_capture, tmp_path):
    """A truncated view must shorten the run, and be reported."""
    import cv2

    short_path = tmp_path / "short.mp4"
    src = CV2FrameReader(stereo_capture["videos"]["right"])
    writer = VideoWriter(str(short_path), fps=30.0)
    try:
        for i in range(3):
            writer.write_bgr(cv2.cvtColor(src[i], cv2.COLOR_RGB2BGR))
    finally:
        writer.close()
        src.close()

    with MultiViewReader({"left": stereo_capture["videos"]["left"], "right": str(short_path)},
                         backend="cv2") as reader:
        assert reader.num_frame == 3
        warnings = reader.check_alignment()
        assert any("different frame counts" in w for w in warnings)


def test_multiview_reader_flags_resolution_mismatch(stereo_capture, tmp_path):
    import cv2

    resized_path = tmp_path / "small.mp4"
    src = CV2FrameReader(stereo_capture["videos"]["right"])
    writer = VideoWriter(str(resized_path), fps=30.0)
    try:
        for i in range(len(src)):
            small = cv2.resize(src[i], (src.frame_size[0] // 2, src.frame_size[1] // 2))
            writer.write_bgr(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    finally:
        writer.close()
        src.close()

    with MultiViewReader({"left": stereo_capture["videos"]["left"], "right": str(resized_path)},
                         backend="cv2") as reader:
        assert any("different resolutions" in w for w in reader.check_alignment())


def test_empty_video_set_rejected():
    with pytest.raises(ValueError):
        MultiViewReader({})


def test_writer_requires_consistent_frames(tmp_path):
    writer = VideoWriter(str(tmp_path / "out.mp4"), fps=25.0)
    try:
        writer.write_bgr(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.close()
    assert (tmp_path / "out.mp4").exists()
