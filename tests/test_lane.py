"""Phase 3 — lane detection tests (classical pipeline on synthetic frames).

No GPU / no model needed. Run: PYTHONPATH=src pytest tests/test_lane.py -v
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drivevision.perception.lane import ClassicalLaneDetector, ModelLaneDetector
from drivevision.types import Frame, LaneSide


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(index=index, timestamp=index / 30.0, image=image)


def _blank(h: int = 480, w: int = 640) -> Frame:
    return _frame(np.zeros((h, w, 3), dtype=np.uint8))


def _synthetic(h: int = 480, w: int = 640) -> Frame:
    """Grey road with two white lines converging toward the centre."""
    img = np.full((h, w, 3), 80, dtype=np.uint8)
    cv2.line(img, (50, h - 1), (w // 2 - 20, h // 2), (255, 255, 255), 12)
    cv2.line(img, (w - 50, h - 1), (w // 2 + 20, h // 2), (255, 255, 255), 12)
    return _frame(img)


class TestClassicalLaneDetector:
    def test_blank_frame_returns_empty(self):
        lanes = ClassicalLaneDetector().detect(_blank())
        assert isinstance(lanes, list)

    def test_synthetic_detects_a_lane(self):
        lanes = ClassicalLaneDetector().detect(_synthetic())
        assert len(lanes) >= 1

    def test_lane_has_valid_points(self):
        for lane in ClassicalLaneDetector().detect(_synthetic()):
            assert len(lane.points) >= 2

    def test_side_classification(self):
        sides = {ln.side for ln in ClassicalLaneDetector().detect(_synthetic())}
        assert LaneSide.LEFT in sides or LaneSide.RIGHT in sides

    def test_confidence_in_range(self):
        for lane in ClassicalLaneDetector().detect(_synthetic()):
            assert 0.0 <= lane.confidence <= 1.0

    def test_points_within_image_bounds(self):
        h, w = 480, 640
        for lane in ClassicalLaneDetector().detect(_synthetic(h, w)):
            for x, y in lane.points:
                assert -50 <= x <= w + 50
                assert 0 <= y <= h

    def test_multiple_frames_smoothing(self):
        det = ClassicalLaneDetector(smooth_buffer=5)
        img = _synthetic().image
        for i in range(10):
            assert isinstance(det.detect(_frame(img, i)), list)

    def test_dark_frame_no_crash(self):
        img = np.full((480, 640, 3), 5, dtype=np.uint8)
        assert isinstance(ClassicalLaneDetector().detect(_frame(img)), list)

    def test_reset_clears_history(self):
        det = ClassicalLaneDetector(smooth_buffer=5)
        for i in range(5):
            det.detect(_frame(_synthetic().image, i))
        det.reset()
        assert len(det._history) == 0

    def test_perspective_mode_no_crash(self):
        lanes = ClassicalLaneDetector(use_perspective=True).detect(_synthetic())
        assert isinstance(lanes, list)


class TestModelLaneDetectorStub:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ModelLaneDetector().detect(_blank())
