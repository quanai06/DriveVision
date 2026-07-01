"""Multi-Object Tracking — association on top of the detector.

Tracking is **not** a deep model: the detector says *what is where in this frame*,
the tracker answers *is this the same object as one in the previous frame?* Each
frame runs predict → match → update:

* **predict**  — SimpleTracker keeps the last bbox (constant-position model).
* **match**    — build an IoU cost matrix and solve the optimal 1-1 assignment
                 with the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``);
                 falls back to greedy matching if SciPy is missing.
* **update**   — matched tracks absorb the detection; unmatched tracks age;
                 unmatched detections spawn new tracks.

Two implementations share the :class:`Tracker` interface so the pipeline never
knows which one it is using:

* :class:`SimpleTracker` — IoU + Hungarian, dependency-light, easy to debug.
* :class:`ByteTracker`   — ByteTrack via ultralytics, fewer ID switches.

``Track.history`` (centres) and ``Track.velocity`` feed Phase 6 TTC estimation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..types import BoundingBox, Detection, Frame, Track
from .base import Tracker

log = logging.getLogger("drivevision.tracking")

# SciPy is optional — without it we fall back to greedy matching.
try:
    from scipy.optimize import linear_sum_assignment as _lsa

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only on machines without scipy
    _HAS_SCIPY = False
    log.warning(
        "scipy not found — SimpleTracker falls back to greedy matching. "
        "Install scipy for optimal Hungarian assignment: pip install scipy"
    )

_MatchResult = Tuple[List[Tuple[int, int]], List[int], List[int]]


def _build_iou_cost_matrix(tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    """Cost matrix ``(n_tracks, n_dets)`` where ``cost = 1 - IoU`` (0 = perfect)."""
    cost = np.ones((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            cost[i, j] = 1.0 - track.detection.bbox.iou(det.bbox)
    return cost


def _hungarian_match(cost: np.ndarray, iou_threshold: float) -> _MatchResult:
    """Optimal 1-1 assignment, then drop pairs whose IoU < ``iou_threshold``.

    Hungarian *forces* an assignment along the shorter dimension, so the
    post-filter is essential to reject low-quality matches.
    """
    n_t, n_d = cost.shape
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    row_ind, col_ind = _lsa(cost)
    matched: List[Tuple[int, int]] = []
    matched_t, matched_d = set(), set()
    for r, c in zip(row_ind, col_ind):
        if (1.0 - cost[r, c]) >= iou_threshold:
            matched.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))

    unmatched_tracks = [i for i in range(n_t) if i not in matched_t]
    unmatched_dets = [j for j in range(n_d) if j not in matched_d]
    return matched, unmatched_tracks, unmatched_dets


def _greedy_match(
    tracks: List[Track], detections: List[Detection], iou_threshold: float
) -> _MatchResult:
    """Greedy fallback: each track grabs its best remaining detection.

    Order-dependent and can ID-switch when two objects pass close together —
    which is exactly why Hungarian is preferred when SciPy is available.
    """
    unmatched_dets = list(range(len(detections)))
    matched: List[Tuple[int, int]] = []
    matched_t = set()
    for i, track in enumerate(tracks):
        best_iou, best_j = iou_threshold, -1
        for j in unmatched_dets:
            score = track.detection.bbox.iou(detections[j].bbox)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_j >= 0:
            matched.append((i, best_j))
            matched_t.add(i)
            unmatched_dets.remove(best_j)
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
    return matched, unmatched_tracks, unmatched_dets


class SimpleTracker(Tracker):
    """IoU tracker with Hungarian matching and ghost-track suppression.

    Parameters
    ----------
    max_age:
        Frames a track survives unmatched before deletion (``time_since_update``).
        Lets a track re-link after a short occlusion.
    iou_threshold:
        Minimum IoU for a (track, detection) pair to count as a match.
    min_hits:
        A track must be seen this many frames before it appears in the output —
        suppresses ghost tracks from single-frame false positives.
    history_len:
        Max number of centres kept in ``Track.history`` (for trajectory + velocity).
    """

    def __init__(
        self,
        max_age: int = 30,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        history_len: int = 30,
    ) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.history_len = history_len
        self._tracks: List[Track] = []
        self._next_id = 0

    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        """Advance the tracker one frame and return confirmed, visible tracks."""
        # 1 & 2 — match existing tracks to new detections.
        if _HAS_SCIPY:
            cost = _build_iou_cost_matrix(self._tracks, detections)
            matched, unmatched_t, unmatched_d = _hungarian_match(cost, self.iou_threshold)
        else:
            matched, unmatched_t, unmatched_d = _greedy_match(
                self._tracks, detections, self.iou_threshold
            )

        # 3a — matched tracks absorb their detection.
        for t_idx, d_idx in matched:
            track = self._tracks[t_idx]
            det = detections[d_idx]
            track.detection = det
            track.age += 1
            track.time_since_update = 0
            track.history.append(det.bbox.center)
            if len(track.history) > self.history_len:
                track.history = track.history[-self.history_len :]

        # 3b — unmatched tracks age (possible occlusion).
        for t_idx in unmatched_t:
            self._tracks[t_idx].time_since_update += 1

        # 3c — unmatched detections spawn new tracks.
        for d_idx in unmatched_d:
            det = detections[d_idx]
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    detection=det,
                    age=0,
                    time_since_update=0,
                    history=[det.bbox.center],
                )
            )
            self._next_id += 1

        # 4 — prune stale tracks.
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # 5 — return only confirmed (age >= min_hits) and visible (tsu == 0) tracks.
        return [
            t for t in self._tracks if t.time_since_update == 0 and t.age >= self.min_hits
        ]

    # -- introspection (handy for tests / debugging) ----------------------------

    @property
    def all_tracks(self) -> List[Track]:
        """Every live track, including unconfirmed and currently-occluded ones."""
        return list(self._tracks)

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all state (call between independent videos)."""
        self._tracks.clear()
        self._next_id = 0


class ByteTracker(Tracker):
    """ByteTrack via ultralytics — the higher-quality option.

    ByteTrack runs its OWN detector internally through ``model.track``, so the
    ``detections`` argument from the pipeline is ignored. The builder disables
    the standalone detector when this backend is selected to avoid running YOLO
    twice. ``persist=True`` keeps tracker state between calls — create a fresh
    instance per video.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        history_len: int = 30,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ByteTracker requires ultralytics: pip install ultralytics") from exc

        self._model = YOLO(weights)
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._classes = classes
        self.history_len = history_len
        self._histories: dict[int, list[tuple[float, float]]] = {}

    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        """Run ByteTrack on the frame; ``detections`` is ignored (internal detector)."""
        results = self._model.track(
            source=frame.image,
            persist=True,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            classes=self._classes,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        tracks: List[Track] = []
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return tracks  # tracker not warmed up yet / nothing to track

        boxes = results[0].boxes
        names = results[0].names
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for track_id, box, conf, cls_id in zip(ids, xyxy, confs, clss):
            bbox = BoundingBox(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            det = Detection(
                bbox=bbox,
                class_id=int(cls_id),
                class_name=names.get(int(cls_id), str(cls_id)),
                confidence=float(conf),
            )
            hist = self._histories.get(int(track_id), [])
            hist.append(bbox.center)
            if len(hist) > self.history_len:
                hist = hist[-self.history_len :]
            self._histories[int(track_id)] = hist
            tracks.append(
                Track(
                    track_id=int(track_id),
                    detection=det,
                    age=len(hist),
                    time_since_update=0,
                    history=list(hist),
                )
            )

        # Forget histories of tracks not present this frame.
        active = set(int(i) for i in ids.tolist())
        self._histories = {k: v for k, v in self._histories.items() if k in active}
        return tracks
