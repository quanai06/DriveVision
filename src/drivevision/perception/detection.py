"""Object detection via Ultralytics YOLO.

Heavy deps (``ultralytics``/``torch``) are imported lazily inside ``__init__`` so
that simply importing the package never requires them. If the configured weights
file is missing, we fall back to a pretrained COCO model so the demo still runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..types import BoundingBox, Detection, Frame
from .base import Detector

_PRETRAINED_FALLBACK = "yolov8n.pt"  # auto-downloaded by ultralytics on first use


class YOLODetector(Detector):
    def __init__(
        self,
        weights: str = "models/weights/yolo.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        classes: Optional[List[int]] = None,
        imgsz: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLODetector. "
                "Install with: pip install ultralytics"
            ) from exc

        model_path = weights if Path(weights).exists() else _PRETRAINED_FALLBACK
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.imgsz = imgsz

    def detect(self, frame: Frame) -> List[Detection]:
        results = self.model.predict(
            frame.image,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        res = results[0]
        names = res.names
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            detections.append(
                Detection(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=float(box.conf[0]),
                )
            )
        return detections
