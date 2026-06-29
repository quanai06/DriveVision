"""Tests for Phase 1: Object Detection + Visualization + Output pipeline.

Run: PYTHONPATH=src pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from drivevision.types import (
    BoundingBox,
    Detection,
    Frame,
    PipelineResult,
    SceneState,
)


# ─────────────────────────── Fixtures ────────────────────────────

@pytest.fixture
def dummy_frame() -> Frame:
    """A 640x480 black frame."""
    return Frame(index=42, timestamp=1.4, image=np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.fixture
def sample_detections() -> list[Detection]:
    return [
        Detection(BoundingBox(10, 20, 200, 150), class_id=2, class_name="car", confidence=0.87),
        Detection(BoundingBox(300, 50, 420, 300), class_id=0, class_name="person", confidence=0.91),
        Detection(BoundingBox(500, 200, 620, 450), class_id=7, class_name="truck", confidence=0.62),
    ]


@pytest.fixture
def pipeline_result(dummy_frame, sample_detections) -> PipelineResult:
    scene = SceneState(
        frame_index=dummy_frame.index,
        timestamp=dummy_frame.timestamp,
        detections=sample_detections,
    )
    return PipelineResult(frame=dummy_frame, scene=scene)


# ─────────────────────────── BoundingBox ─────────────────────────

class TestBoundingBox:
    def test_as_int(self):
        assert BoundingBox(10.7, 20.3, 200.9, 150.1).as_int() == (10, 20, 200, 150)

    def test_width_height(self):
        bb = BoundingBox(0, 0, 100, 50)
        assert bb.width == 100
        assert bb.height == 50

    def test_area(self):
        assert BoundingBox(0, 0, 10, 10).area == 100.0

    def test_center(self):
        assert BoundingBox(0, 0, 100, 60).center == (50.0, 30.0)

    def test_iou_identical(self):
        bb = BoundingBox(0, 0, 10, 10)
        assert abs(bb.iou(bb) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        assert a.iou(b) == 0.0

    def test_iou_half_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 15, 10)
        # inter=50, union=150 -> 1/3
        assert abs(a.iou(b) - (50 / 150)) < 1e-6


# ─────────────────────────── Annotator ───────────────────────────

class TestAnnotator:
    def test_draw_returns_same_shape(self, pipeline_result, dummy_frame):
        from drivevision.viz.annotator import Annotator

        out = Annotator().draw(pipeline_result, fps=15.0)
        assert isinstance(out, np.ndarray)
        assert out.shape == dummy_frame.image.shape
        assert out.dtype == np.uint8

    def test_draw_does_not_modify_original(self, pipeline_result):
        from drivevision.viz.annotator import Annotator

        original = pipeline_result.frame.image.copy()
        Annotator().draw(pipeline_result, fps=0.0)
        assert np.array_equal(pipeline_result.frame.image, original)

    def test_draw_actually_paints_pixels(self, pipeline_result):
        """With detections, the output must differ from the black input."""
        from drivevision.viz.annotator import Annotator

        out = Annotator().draw(pipeline_result, fps=10.0)
        assert out.sum() > 0  # something was drawn

    def test_draw_empty_detections(self, dummy_frame):
        from drivevision.viz.annotator import Annotator

        result = PipelineResult(
            frame=dummy_frame, scene=SceneState(frame_index=0, timestamp=0.0)
        )
        out = Annotator().draw(result, fps=30.0)
        assert out.shape == dummy_frame.image.shape

    def test_draw_many_detections(self, dummy_frame):
        from drivevision.viz.annotator import Annotator

        dets = [
            Detection(
                BoundingBox(i * 10, i * 5, i * 10 + 80, i * 5 + 60),
                class_id=i % 12, class_name=f"obj{i}", confidence=0.5 + i * 0.02,
            )
            for i in range(20)
        ]
        result = PipelineResult(
            frame=dummy_frame,
            scene=SceneState(frame_index=0, timestamp=0.0, detections=dets),
        )
        assert Annotator().draw(result).shape == dummy_frame.image.shape

    def test_draw_box_at_top_edge(self, dummy_frame):
        """A box flush with y=0 must not crash (label flips below)."""
        from drivevision.viz.annotator import Annotator

        det = Detection(BoundingBox(5, 0, 100, 40), class_id=2, class_name="car", confidence=0.9)
        result = PipelineResult(
            frame=dummy_frame,
            scene=SceneState(frame_index=0, timestamp=0.0, detections=[det]),
        )
        assert Annotator().draw(result).shape == dummy_frame.image.shape

    def test_draw_fps_zero(self, pipeline_result):
        from drivevision.viz.annotator import Annotator

        assert Annotator().draw(pipeline_result, fps=0.0) is not None

    def test_draw_returns_new_array(self, pipeline_result):
        from drivevision.viz.annotator import Annotator

        out = Annotator().draw(pipeline_result)
        assert out is not pipeline_result.frame.image


# ─────────────────────────── YOLODetector (mock) ─────────────────

class TestYOLODetectorMock:
    """Exercise the result-parsing logic without a real model or GPU."""

    def test_detect_empty_results(self, dummy_frame):
        from unittest.mock import MagicMock

        from drivevision.perception.detection import YOLODetector

        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.names = {}

        det = YOLODetector.__new__(YOLODetector)
        det.conf, det.iou, det.classes, det.imgsz = 0.35, 0.5, None, 640
        det.model = MagicMock()
        det.model.predict.return_value = [mock_result]

        assert det.detect(dummy_frame) == []

    def test_detect_parses_boxes(self, dummy_frame):
        from unittest.mock import MagicMock

        from drivevision.perception.detection import YOLODetector

        class _Val:
            """Mimics a torch tensor element supporting .tolist()/int()/float()."""

            def __init__(self, v):
                self._v = v

            def tolist(self):
                return self._v

            def __float__(self):
                return float(self._v)

            def __int__(self):
                return int(self._v)

        mock_box = MagicMock()
        mock_box.xyxy = [_Val([10.0, 20.0, 200.0, 150.0])]
        mock_box.cls = [_Val(2.0)]
        mock_box.conf = [_Val(0.87)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {2: "car"}

        det = YOLODetector.__new__(YOLODetector)
        det.conf, det.iou, det.classes, det.imgsz = 0.35, 0.5, None, 640
        det.model = MagicMock()
        det.model.predict.return_value = [mock_result]

        dets = det.detect(dummy_frame)
        assert len(dets) == 1
        assert dets[0].class_name == "car"
        assert abs(dets[0].confidence - 0.87) < 0.001
        assert dets[0].bbox.as_int() == (10, 20, 200, 150)


# ─────────────────────────── Pipeline end-to-end ─────────────────

class TestPipelinePhase1:
    def test_pipeline_no_detector(self, dummy_frame):
        from drivevision.pipeline.pipeline import Pipeline

        result = Pipeline().process(dummy_frame)
        assert result.scene.detections == []
        assert result.scene.tracks == []
        assert result.risk is None
        assert result.decision is None

    def test_pipeline_with_mock_detector(self, dummy_frame, sample_detections):
        from drivevision.pipeline.pipeline import Pipeline
        from drivevision.perception.base import Detector

        class MockDetector(Detector):
            def detect(self, frame):
                return sample_detections

        result = Pipeline(detector=MockDetector()).process(dummy_frame)
        assert len(result.scene.detections) == 3
        assert result.scene.detections[0].class_name == "car"

    def test_annotator_integrates_with_pipeline(self, dummy_frame, sample_detections):
        from drivevision.pipeline.pipeline import Pipeline
        from drivevision.perception.base import Detector
        from drivevision.viz.annotator import Annotator

        class MockDetector(Detector):
            def detect(self, frame):
                return sample_detections

        result = Pipeline(detector=MockDetector()).process(dummy_frame)
        out = Annotator().draw(result, fps=25.0)
        assert out.shape == dummy_frame.image.shape


# ─────────────────────────── FPS counter ─────────────────────────

class TestFPSCounter:
    def test_fps_zero_on_single_tick(self):
        from drivevision.cli import _FPSCounter

        c = _FPSCounter(window=30)
        c.tick()
        assert c.fps == 0.0

    def test_fps_positive_after_two_ticks(self):
        import time

        from drivevision.cli import _FPSCounter

        c = _FPSCounter(window=30)
        c.tick()
        time.sleep(0.05)
        c.tick()
        assert c.fps > 5.0


# ─────────────────────────── Config ──────────────────────────────

class TestConfig:
    def test_load_default_config(self):
        from drivevision.config import load_config

        cfg = load_config()
        assert cfg["source"]["type"] == "video"
        assert cfg["output"]["display"] is False
        assert cfg["output"]["window_name"] == "DriveVision"
        assert cfg["logging"]["log_interval"] == 30

    def test_yaml_and_defaults_in_sync(self):
        """The shipped YAML must merge cleanly and expose the new Phase 1 keys."""
        from drivevision.config import load_config

        cfg = load_config("configs/default.yaml")
        assert cfg["perception"]["detection"]["device"] is None
        assert cfg["output"]["window_name"] == "DriveVision"
        assert "log_interval" in cfg["logging"]

    def test_get_path(self):
        from drivevision.config import get_path

        cfg = {"perception": {"detection": {"conf": 0.35}}}
        assert get_path(cfg, "perception.detection.conf") == 0.35
        assert get_path(cfg, "perception.detection.missing", 99) == 99

    def test_get_path_missing_key_returns_default(self):
        from drivevision.config import get_path

        assert get_path({}, "a.b.c", "default") == "default"
