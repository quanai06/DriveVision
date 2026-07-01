"""Lane detection.

Phase 3 ships a full **classical** OpenCV pipeline — colour (HLS white/yellow) +
Canny edges, trapezoidal ROI, Hough segments split into left/right, a polynomial
fit ``x = f(y)`` per side, and temporal smoothing so the lanes don't jitter
frame-to-frame. An optional ``use_perspective`` mode warps to a bird's-eye view
and uses a sliding-window fit (better on curves / dashed lines).

Lanes are a different shape of output than detections — continuous polylines, not
rigid boxes — so this is its own ``LaneDetector`` (not reusing ``Detector``).
A learned :class:`ModelLaneDetector` (YOLOP / HybridNets / UFLD) plugs into the
same interface in Phase 8; for now it is a clean stub.

``detect`` never raises — on any failure (or empty/dark frame) it returns ``[]``
so the pipeline keeps running.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..types import Frame, Lane, LaneSide
from .base import LaneDetector

log = logging.getLogger("drivevision.lane")

_N_POINTS = 20          # points per fitted lane (at fixed y positions -> easy averaging)
_Y_TOP_RATIO = 0.60     # top of the lane region (fraction of image height)

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Image-processing helpers
# --------------------------------------------------------------------------- #
def _hls_color_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask emphasising white and yellow lane paint (robust to lighting)."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    white = (l > 200) & (s < 60)
    yellow = (h >= 15) & (h <= 35) & (s > 80) & (l > 80)
    return ((white | yellow).astype(np.uint8)) * 255


def _roi_trapezoid(h: int, w: int, top_ratio: float = _Y_TOP_RATIO,
                   top_width_ratio: float = 0.20, bottom_pad_ratio: float = 0.05) -> np.ndarray:
    """Trapezoidal ROI mask over the road area ahead of the car."""
    cx = w / 2.0
    y_top, y_bot = int(h * top_ratio), h
    pts = np.array([[
        (int(cx - (0.5 - bottom_pad_ratio) * w), y_bot),
        (int(cx - top_width_ratio * w), y_top),
        (int(cx + top_width_ratio * w), y_top),
        (int(cx + (0.5 - bottom_pad_ratio) * w), y_bot),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def _separate_lines(lines: np.ndarray, w: int, slope_min: float = 0.4,
                    slope_max: float = 5.0) -> Tuple[List, List]:
    """Split Hough segments into left/right by slope sign and x position."""
    cx = w / 2.0
    left, right = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not (slope_min <= abs(slope) <= slope_max):
            continue
        mid_x = (x1 + x2) / 2.0
        if slope < 0 and mid_x < cx:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > cx:
            right.append((x1, y1, x2, y2))
    return left, right


def _fit_points(segments: List, h: int, deg: int) -> Optional[List[Point]]:
    """Fit ``x = poly(y)`` over all segment endpoints; sample at fixed y's.

    Fitting x as a function of y (not y of x) avoids the vertical-line blow-up.
    Always returns exactly ``_N_POINTS`` at the SAME y positions so temporal
    averaging across frames is a trivial per-row mean. ``None`` if underdetermined.
    """
    if not segments:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    if len(xs) <= deg:
        return None
    try:
        coeffs = np.polyfit(ys, xs, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None
    y_vals = np.linspace(int(h * _Y_TOP_RATIO), h, _N_POINTS)
    x_vals = np.polyval(coeffs, y_vals)
    return [(float(x), float(y)) for x, y in zip(x_vals, y_vals)]


# --------------------------------------------------------------------------- #
# Perspective (bird's-eye) helpers
# --------------------------------------------------------------------------- #
def _perspective_matrices(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Forward/inverse warp for a bird's-eye view (tuned for a generic front cam)."""
    src = np.float32([[w * 0.43, h * 0.65], [w * 0.57, h * 0.65],
                      [w * 0.90, h * 0.95], [w * 0.10, h * 0.95]])
    dst = np.float32([[w * 0.25, 0], [w * 0.75, 0], [w * 0.75, h], [w * 0.25, h]])
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def _sliding_window_fit(binary: np.ndarray, n_windows: int = 9, margin: int = 100,
                        min_pix: int = 50, deg: int = 2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Sliding-window lane fit on a bird's-eye binary image -> (left, right) coeffs."""
    h, w = binary.shape
    histogram = np.sum(binary[h // 2:, :], axis=0)
    mid = w // 2
    leftx_base = int(np.argmax(histogram[:mid]))
    rightx_base = int(np.argmax(histogram[mid:])) + mid

    win_h = h // n_windows
    nz = binary.nonzero()
    nz_y, nz_x = np.array(nz[0]), np.array(nz[1])
    left_cur, right_cur = leftx_base, rightx_base
    left_inds, right_inds = [], []

    for win in range(n_windows):
        y_lo, y_hi = h - (win + 1) * win_h, h - win * win_h
        gl = ((nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= left_cur - margin) & (nz_x < left_cur + margin)).nonzero()[0]
        gr = ((nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= right_cur - margin) & (nz_x < right_cur + margin)).nonzero()[0]
        left_inds.append(gl)
        right_inds.append(gr)
        if len(gl) > min_pix:
            left_cur = int(np.mean(nz_x[gl]))
        if len(gr) > min_pix:
            right_cur = int(np.mean(nz_x[gr]))

    li = np.concatenate(left_inds) if left_inds else np.array([])
    ri = np.concatenate(right_inds) if right_inds else np.array([])
    left = np.polyfit(nz_y[li], nz_x[li], deg) if len(li) > min_pix * 2 else None
    right = np.polyfit(nz_y[ri], nz_x[ri], deg) if len(ri) > min_pix * 2 else None
    return left, right


# --------------------------------------------------------------------------- #
# Classical detector
# --------------------------------------------------------------------------- #
class ClassicalLaneDetector(LaneDetector):
    """HLS + Canny → ROI → Hough → polyfit → temporal smoothing.

    Parameters
    ----------
    use_perspective:
        Bird's-eye view + sliding window instead of raw Hough (better on curves).
    smooth_alpha:
        EMA factor for the perspective mode (higher = snappier, lower = smoother).
    smooth_buffer:
        Number of past frames averaged in the Hough mode (deque length).
    poly_deg:
        Polynomial degree (1 = straight, 2 = curved).
    """

    def __init__(self, use_perspective: bool = False, smooth_alpha: float = 0.30,
                 smooth_buffer: int = 7, poly_deg: int = 1) -> None:
        self.use_perspective = use_perspective
        self.alpha = smooth_alpha
        self.poly_deg = poly_deg
        self._history: deque = deque(maxlen=smooth_buffer)        # (left_pts|None, right_pts|None)
        self._left_coeffs: Optional[np.ndarray] = None            # perspective EMA state
        self._right_coeffs: Optional[np.ndarray] = None
        self._M: Optional[np.ndarray] = None
        self._M_inv: Optional[np.ndarray] = None

    def detect(self, frame: Frame) -> List[Lane]:
        """Return detected lanes; never raises (returns ``[]`` on failure)."""
        try:
            return self._detect_impl(frame)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("ClassicalLaneDetector error (frame %d): %s", frame.index, exc)
            return []

    def _detect_impl(self, frame: Frame) -> List[Lane]:
        img = frame.image
        h, w = img.shape[:2]
        color = _hls_color_mask(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 150)
        combined = cv2.bitwise_or(edges, color)
        masked = cv2.bitwise_and(combined, _roi_trapezoid(h, w))
        if self.use_perspective:
            return self._detect_perspective(masked, h, w)
        return self._detect_hough(masked, h, w)

    # -- Hough mode -------------------------------------------------------------
    def _detect_hough(self, masked: np.ndarray, h: int, w: int) -> List[Lane]:
        lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=40,
                                minLineLength=40, maxLineGap=100)
        if lines is None:
            raw_left = raw_right = None
        else:
            left_segs, right_segs = _separate_lines(lines, w)
            raw_left = _fit_points(left_segs, h, self.poly_deg)
            raw_right = _fit_points(right_segs, h, self.poly_deg)

        self._history.append((raw_left, raw_right))
        sm_left = self._avg_history(side=0)
        sm_right = self._avg_history(side=1)

        lanes: List[Lane] = []
        if sm_left:
            lanes.append(Lane(points=sm_left, side=LaneSide.LEFT,
                              confidence=0.8 if raw_left else 0.4))
        if sm_right:
            lanes.append(Lane(points=sm_right, side=LaneSide.RIGHT,
                              confidence=0.8 if raw_right else 0.4))
        return lanes

    def _avg_history(self, side: int) -> Optional[List[Point]]:
        """Per-row mean of x across buffered frames (all share the same y's)."""
        valid = [item[side] for item in self._history if item[side] is not None]
        if not valid:
            return None
        arr = np.array(valid, dtype=np.float32)          # (n_frames, _N_POINTS, 2)
        mean = arr.mean(axis=0)
        return [(float(x), float(y)) for x, y in mean]

    # -- Perspective mode -------------------------------------------------------
    def _detect_perspective(self, masked: np.ndarray, h: int, w: int) -> List[Lane]:
        if self._M is None:
            self._M, self._M_inv = _perspective_matrices(h, w)
        warped = cv2.warpPerspective(masked, self._M, (w, h))
        left_c, right_c = _sliding_window_fit(warped, deg=max(self.poly_deg, 2))
        self._left_coeffs = self._ema(self._left_coeffs, left_c)
        self._right_coeffs = self._ema(self._right_coeffs, right_c)

        lanes: List[Lane] = []
        y_vals = np.linspace(int(h * _Y_TOP_RATIO), h, _N_POINTS)
        for coeffs, side, raw in [(self._left_coeffs, LaneSide.LEFT, left_c),
                                  (self._right_coeffs, LaneSide.RIGHT, right_c)]:
            if coeffs is None:
                continue
            x_vals = np.polyval(coeffs, y_vals)
            warped_pts = np.column_stack([x_vals, y_vals]).astype(np.float32).reshape(-1, 1, 2)
            img_pts = cv2.perspectiveTransform(warped_pts, self._M_inv).reshape(-1, 2)
            pts = [(float(p[0]), float(p[1])) for p in img_pts]
            lanes.append(Lane(points=pts, side=side, confidence=0.85 if raw is not None else 0.5))
        return lanes

    def _ema(self, prev: Optional[np.ndarray], curr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if curr is None:
            return prev
        if prev is None or prev.shape != curr.shape:
            return curr
        return self.alpha * curr + (1 - self.alpha) * prev

    def reset(self) -> None:
        """Clear smoothing state (use on scene cuts / new clips)."""
        self._history.clear()
        self._left_coeffs = self._right_coeffs = None


class ModelLaneDetector(LaneDetector):
    """Learned lane detector — clean stub, implemented in Phase 8.

    Weights are trained/fine-tuned on Kaggle and saved to ``models/weights/lane.pt``.
    Planned backends (``model_type``):

    * ``yolov8-seg`` — ``from ultralytics import YOLO; YOLO(weights)``; lane-class
      mask → ``findContours`` → polylines.
    * ``yolop`` — multitask (det + lane + drivable); threshold lane mask →
      connected components → one ``Lane`` each.
    * ``ufld`` — Ultra-Fast-Lane-Detection v2 row anchors → reconstruct (x, y).
    """

    def __init__(self, weights: str = "models/weights/lane.pt",
                 model_type: str = "yolov8-seg", device: str = "cpu", conf: float = 0.5) -> None:
        self.weights = weights
        self.model_type = model_type
        self.device = device
        self.conf = conf
        self._model = None
        log.info("ModelLaneDetector created (weights=%s, type=%s) — not implemented until Phase 8.",
                 weights, model_type)

    def detect(self, frame: Frame) -> List[Lane]:  # pragma: no cover - stub
        raise NotImplementedError(
            "ModelLaneDetector.detect() is not implemented yet.\n"
            f"  weights : {self.weights} (model_type={self.model_type})\n"
            "  datasets: TuSimple / CULane / BDD100K lane\n"
            "  train   : Kaggle notebook (Phase 8); see phase_8.md\n"
            "  Use ClassicalLaneDetector (method: classical) in the meantime."
        )
