"""Phase 2 — tracking tests. ID consistency + Hungarian matching.

No GPU / no ultralytics needed for SimpleTracker tests (numpy + scipy only).
Run: PYTHONPATH=src pytest tests/test_tracking.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from drivevision.perception.tracking import SimpleTracker
from drivevision.types import BoundingBox, Detection, Frame


# ----------------------------------------------------------------- helpers ---

def _frame(idx: int = 0) -> Frame:
    return Frame(index=idx, timestamp=idx / 30.0, image=np.zeros((480, 640, 3), dtype=np.uint8))


def _det(x1, y1, x2, y2, cls="car", conf=0.9) -> Detection:
    return Detection(BoundingBox(x1, y1, x2, y2), class_id=2, class_name=cls, confidence=conf)


def _move(det: Detection, dx=5.0, dy=0.0) -> Detection:
    b = det.bbox
    return _det(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy, det.class_name, det.confidence)


def _has_ultralytics() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


# ------------------------------------------------------ ID consistency -------

class TestSimpleTrackerIDConsistency:
    def test_single_object_stable_id(self):
        # min_hits=1 -> a track is confirmed on its first *match* (frame 1 onward),
        # so allow the spawn frame to be empty; the id must then stay constant.
        tracker = SimpleTracker(max_age=5, iou_threshold=0.3, min_hits=1)
        det = _det(100, 100, 200, 200)
        seen, appearances = set(), 0
        for i in range(30):
            tracks = tracker.update(_frame(i), [det])
            if tracks:
                seen.add(tracks[0].track_id)
                appearances += 1
            det = _move(det, dx=2.0)
        assert seen == {0}, f"ID switch! ids={seen}"
        assert appearances >= 28, "track should be visible on nearly every frame"

    def test_two_objects_no_swap(self):
        tracker = SimpleTracker(max_age=5, iou_threshold=0.3, min_hits=1)
        det_a = _det(50, 100, 150, 200)
        det_b = _det(400, 100, 500, 200)
        tracker.update(_frame(0), [det_a, det_b])  # spawn (age 0, not yet shown)
        tracks = tracker.update(_frame(0), [det_a, det_b])  # confirmed
        assert len(tracks) == 2
        id_left = next(t.track_id for t in tracks if t.detection.bbox.x1 < 200)
        id_right = next(t.track_id for t in tracks if t.detection.bbox.x1 > 300)
        for i in range(1, 20):
            det_a = _move(det_a, dx=3.0)
            det_b = _move(det_b, dx=-3.0)
            tracks = tracker.update(_frame(i), [det_a, det_b])
            if len(tracks) == 2:
                cur_left = next((t.track_id for t in tracks if t.detection.bbox.x1 < 300), None)
                cur_right = next((t.track_id for t in tracks if t.detection.bbox.x1 >= 300), None)
                if cur_left is not None:
                    assert cur_left == id_left, f"Frame {i}: left ID swapped"
                if cur_right is not None:
                    assert cur_right == id_right, f"Frame {i}: right ID swapped"

    def test_min_hits_filters_ghost(self):
        tracker = SimpleTracker(max_age=10, iou_threshold=0.3, min_hits=3)
        det = _det(100, 100, 200, 200)
        assert tracker.update(_frame(0), [det]) == []  # age 0
        assert tracker.update(_frame(1), [det]) == []  # age 1
        assert tracker.update(_frame(2), [det]) == []  # age 2
        assert len(tracker.update(_frame(3), [det])) == 1  # age 3 >= min_hits

    def test_track_survives_occlusion(self):
        tracker = SimpleTracker(max_age=5, iou_threshold=0.3, min_hits=1)
        det = _det(100, 100, 200, 200)
        tracker.update(_frame(0), [det])
        for i in range(1, 5):
            assert tracker.update(_frame(i), []) == []  # not visible
        assert tracker.track_count == 1  # but still alive
        assert len(tracker.update(_frame(5), [det])) == 1  # re-links

    def test_track_dies_after_max_age(self):
        tracker = SimpleTracker(max_age=3, iou_threshold=0.3, min_hits=1)
        tracker.update(_frame(0), [_det(100, 100, 200, 200)])
        for i in range(1, 6):
            tracker.update(_frame(i), [])
        assert tracker.track_count == 0

    def test_no_detection_no_crash(self):
        tracker = SimpleTracker(min_hits=1)
        for i in range(5):
            assert tracker.update(_frame(i), []) == []

    def test_velocity_converges(self):
        tracker = SimpleTracker(max_age=10, iou_threshold=0.3, min_hits=1)
        det = _det(100, 100, 200, 200)
        tracker.update(_frame(0), [det])
        tracks = tracker.update(_frame(1), [_move(det, dx=10.0)])
        assert len(tracks) == 1
        vx, vy = tracks[0].velocity
        assert vx != 0.0 or vy != 0.0

    def test_history_length_capped(self):
        tracker = SimpleTracker(max_age=200, iou_threshold=0.3, min_hits=1, history_len=10)
        det = _det(100, 100, 200, 200)
        for i in range(50):
            det = _move(det, dx=1.0)
            tracker.update(_frame(i), [det])
        for t in tracker.all_tracks:
            assert len(t.history) <= 10

    def test_reset_clears_state(self):
        tracker = SimpleTracker(min_hits=1)
        tracker.update(_frame(0), [_det(100, 100, 200, 200)])
        assert tracker.track_count == 1
        tracker.reset()
        assert tracker.track_count == 0
        # ids restart from 0 (check the live track; min_hits delays output by 1 frame)
        tracker.update(_frame(0), [_det(100, 100, 200, 200)])
        assert tracker.all_tracks[0].track_id == 0


# ------------------------------------------------------ matching quality -----

class TestHungarianMatching:
    def test_matching_assigns_correct_track(self):
        tracker = SimpleTracker(max_age=5, iou_threshold=0.1, min_hits=1)
        tracker.update(_frame(0), [_det(100, 100, 200, 200), _det(210, 100, 310, 200)])
        tracks = tracker.update(_frame(1), [_det(105, 100, 205, 200), _det(215, 100, 315, 200)])
        assert len(tracks) == 2
        assert sorted(t.track_id for t in tracks) == [0, 1]

    def test_no_double_assignment(self):
        """One detection must not be claimed by two tracks."""
        tracker = SimpleTracker(max_age=5, iou_threshold=0.1, min_hits=1)
        tracker.update(_frame(0), [_det(100, 100, 200, 200), _det(130, 100, 230, 200)])
        tracks = tracker.update(_frame(1), [_det(115, 100, 215, 200)])
        # Only one of the two tracks can match the single detection.
        assert len([t for t in tracks if t.time_since_update == 0]) <= 1


# ------------------------------------------------------ ByteTracker smoke ----

class TestByteTrackerSmoke:
    @pytest.mark.skipif(not _has_ultralytics(), reason="ultralytics not installed")
    def test_returns_list_of_tracks(self):
        from drivevision.perception.tracking import ByteTracker
        from drivevision.types import Track

        tracker = ByteTracker(weights="models/weights/yolo.pt", conf=0.5)
        result = tracker.update(_frame(0), [])
        assert isinstance(result, list)
        assert all(isinstance(t, Track) for t in result)
