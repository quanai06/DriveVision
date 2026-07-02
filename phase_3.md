# Phase 3 — Phát hiện Làn Đường (Lane Detection)

**Mục tiêu: Xây dựng thành phần phát hiện làn đường hoàn chỉnh — gồm pipeline cổ điển OpenCV được hoàn thiện (fit đa thức, làm mượt theo thời gian, tuỳ chọn bird's-eye view) và lộ trình tích hợp model học máy — để `SceneState.lanes` luôn có dữ liệu đáng tin cậy phục vụ Phase 5 gán ego-lane.**

> **Tham chiếu lộ trình:**
> - Trước Phase 3: P0 (nền tảng, skeleton), P1 (detection YOLO), P2 (tracking).
> - Sau Phase 3: P4 (traffic light), P5 (scene understanding — dùng `lanes` để xác định ego-lane), P6 (risk+decision — dùng ego-lane để phán đoán nguy hiểm), P8 (fine-tuning model trên Kaggle), P9 (CARLA).
> - Phase 3 **không** thay đổi nội dung các phase khác; chỉ lấp đầy `ClassicalLaneDetector` và phác thảo `ModelLaneDetector`.

---

## 1. Tổng quan & vị trí trong lộ trình

### 1.1 Lane detection là gì và tại sao cần module riêng

Phát hiện làn đường là **task cơ bản nhất** trong perception xe tự hành: hệ thống phải biết xe đang chạy trên làn nào, ranh giới làn ở đâu, từ đó mới có thể đánh giá xe có lệch làn không, ai đang cắt làn, và đề xuất hành động lái.

Điểm khác biệt quan trọng so với object detection (Phase 1):

| Đặc điểm | Object Detection (Phase 1) | Lane Detection (Phase 3) |
|---|---|---|
| Đầu ra | Bounding box (hình chữ nhật) | Polyline/đường cong (danh sách điểm) |
| Đối tượng | Xe, người, biển báo | Vạch sơn đường (thường thiếu, nhòe, đứt) |
| Hình dạng | Rigid, đặc trưng rõ | Mảnh, dài, bị che khuất nhiều |
| Phương pháp | YOLO → bbox | Segmentation / Hough / polynomial fit |
| Đơn vị | "object" riêng rẽ | Cấu trúc hình học liên tục |

Vì vậy, `LaneDetector` là ABC riêng biệt (`perception/base.py`), **không** tái sử dụng `Detector`. Tuy nhiên, model đa nhiệm (YOLOP / HybridNets) có thể cùng lúc làm cả hai — được nêu trong hướng (B) dưới đây.

### 1.2 Vị trí trong pipeline

```
VideoSource → Frame → Pipeline.process()
                          ├─ Detector (YOLO)  → detections
                          ├─ Tracker          → tracks
                          ├─ LaneDetector ←── Phase 3 tập trung vào đây
                          │       └─ detect(frame) → List[Lane]
                          ├─ TrafficLightDetector
                          └─ SceneBuilder
                                └─ SceneState { lanes: List[Lane], ... }
                                        ↓
                              Phase 5 (ego-lane assignment)
```

`Pipeline.process()` tại `src/drivevision/pipeline/pipeline.py` dòng 47 đã gọi:

```python
lanes = self.lane_detector.detect(frame) if self.lane_detector else []
```

`build_pipeline()` tại `src/drivevision/pipeline/builder.py` dòng 67–71 đã khởi tạo detector theo config. Phase 3 **không sửa** các file này — chỉ hoàn thiện nội dung `perception/lane.py` và `viz/annotator.py`.

---

## 2. Mục tiêu (đo được)

| # | Mục tiêu | Cách đo | Ngưỡng chấp nhận |
|---|---|---|---|
| 2.1 | Classical detector chạy không crash trên video BDD100K | Xử lý 100 frame liên tiếp không raise exception | 100% |
| 2.2 | Phân loại đúng trái/phải | Kiểm tra thủ công trên 20 ảnh có vạch rõ | ≥ 85% frame phân loại đúng cả hai bên |
| 2.3 | Fit đường bậc 1 ổn định | Không nhảy frame-to-frame khi vạch rõ | Độ lệch trung bình x < 15 px giữa frame liền kề |
| 2.4 | Temporal smoothing giảm rung | So sánh trước/sau smoothing | Jitter giảm ≥ 50% trên đoạn thẳng |
| 2.5 | Annotator vẽ được lane | Hiển thị polyline / vùng tô màu trên output frame | Nhìn thấy rõ trên demo video |
| 2.6 | Edge case không crash | Test ảnh tối, không có vạch, vạch đứt | Trả về `[]` một cách tường minh, không exception |
| 2.7 | ModelLaneDetector có stub sạch | `NotImplementedError` với message hướng dẫn rõ | Ghi lại path weight, dataset, bước train |

---

## 3. Phạm vi

### Trong phạm vi Phase 3

- Hoàn thiện `ClassicalLaneDetector` với HLS color filter, polynomial fit bậc 1, temporal smoothing.
- Thêm tuỳ chọn `use_perspective` (perspective transform + sliding window) vào `ClassicalLaneDetector`.
- Cập nhật `ModelLaneDetector` stub với hướng dẫn tích hợp model học máy.
- Vẽ kết quả lane trong `Annotator.draw()` (polyline + optional filled polygon).
- Viết test `tests/test_lane.py` trên ảnh tĩnh (tạo ảnh synthetic hoặc dùng sample BDD100K).
- Cập nhật `configs/default.yaml` với thêm các tham số lane cụ thể.
- Tài liệu hoá edge case.

### Ngoài phạm vi Phase 3

- Ego-lane assignment (Phase 5).
- Fine-tuning model thật trên Kaggle (Phase 8).
- Tích hợp CARLA semantic lane data (Phase 9).
- Real-time performance tuning / TensorRT.
- 3D lane reconstruction.

---

## 4. Điều kiện tiên quyết

| Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|
| Python 3.12 với venv | Sẵn sàng | `/home/quan/DriveVision/.venv` |
| OpenCV (`opencv-python`) | Đã có | Dùng cho Canny, HLS, HoughLinesP, warpPerspective |
| NumPy | Đã có | Dùng cho polyfit, averaging |
| `src/drivevision/types.py` có `Lane`, `LaneSide`, `Frame` | Đã có (xem §6.1) | Không thay đổi interface |
| `src/drivevision/perception/base.py` có `LaneDetector` ABC | Đã có | Không thay đổi |
| `src/drivevision/perception/lane.py` có skeleton | Đã có | Phase 3 hoàn thiện |
| `src/drivevision/pipeline/builder.py` wire lane detector | Đã có | Không thay đổi |
| `configs/default.yaml` có `perception.lane` | Đã có | Thêm sub-keys |
| Video mẫu BDD100K hoặc KITTI | Cần chuẩn bị | Tải về `data/samples/` |

---

## 5. Công nghệ & thư viện

### 5.1 Hướng (A) — Classical (OpenCV)

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Canny, HLS, HoughLinesP, warpPerspective, fillPoly, polylines |
| `numpy` | ≥ 1.26 | polyfit (bậc 1/2), mảng điểm, mean/ewm |
| `collections.deque` | stdlib | Buffer temporal smoothing |

Không cần GPU. Chạy được trên bất kỳ máy nào có OpenCV.

### 5.2 Hướng (B) — Model học máy

| Model | Loại | Dataset | Ghi chú |
|---|---|---|---|
| **YOLOv8-seg** (Ultralytics) | Segmentation | BDD100K lane, TuSimple | Cắm vào `ModelLaneDetector`; cùng `ultralytics` dependency đã có ở Phase 1 |
| **Ultra-Fast-Lane-Detection v2** | Row-anchor classification | CULane, TuSimple | Nhẹ, nhanh, xử lý tốt cua gắt |
| **LaneNet** (lanenet-lane-detection) | Instance segmentation | TuSimple | Tách từng làn riêng bằng embedding |
| **YOLOP** (Huawei Noah's Ark) | Multitask: det + lane + drivable area | BDD100K | **Ưu điểm gộp**: 1 lần forward pass sinh đồng thời boxes + lane mask + drivable area mask — tiết kiệm compute, tốt cho portfolio vì demo 3-in-1 |
| **HybridNets** | Multitask: det + lane + drivable area | BDD100K | Nhẹ hơn YOLOP, phù hợp Kaggle P100 |

**Lý do ưu tiên multitask model cho Phase 8:**
- Một lần inference sinh cả `detections` (thay YOLO riêng) + `lanes` + `drivable_area`.
- Giảm latency tổng thể dù mỗi stage không tối ưu bằng model chuyên biệt.
- Demo portfolio ấn tượng: 1 model → 3 output cùng lúc.
- YOLOP và HybridNets đều có pretrained trên BDD100K, fine-tune nhẹ trên Kaggle.

### 5.3 Dataset cho Phase 8 (train/fine-tune)

| Dataset | Size | URL/Kaggle | Loại nhãn |
|---|---|---|---|
| **TuSimple** | 6.408 video clips | `kaggle datasets download -d manideep1st/tuslane` | Polyline JSON |
| **CULane** | 133.235 frames | CULane GitHub (tải thủ công) | Text file dòng điểm |
| **BDD100K lane** | 100K frames | `bdd100k/bdd100k-lane-detection` trên Kaggle | JSON polygon |

---

## 6. Thiết kế chi tiết

### 6.1 Interface bất biến (KHÔNG thay đổi)

Các type dưới đây **đã khoá** tại `src/drivevision/types.py`. Phase 3 chỉ đọc, không sửa:

```python
# src/drivevision/types.py (tham khảo — không sửa)

class LaneSide(str, Enum):
    LEFT    = "left"
    RIGHT   = "right"
    UNKNOWN = "unknown"

@dataclass
class Lane:
    points: List[Tuple[float, float]]  # [(x0,y0), (x1,y1), ...] pixel coords
    side: LaneSide = LaneSide.UNKNOWN
    confidence: float = 1.0

@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray   # HxWx3, BGR, OpenCV convention
    depth: Optional[np.ndarray] = None
    semantic: Optional[np.ndarray] = None

    @property
    def height(self) -> int: return self.image.shape[0]
    @property
    def width(self)  -> int: return self.image.shape[1]

@dataclass
class SceneState:
    frame_index: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track]        = field(default_factory=list)
    lanes: List[Lane]          = field(default_factory=list)   # ← Phase 3 điền vào đây
    traffic_lights: List[TrafficLight] = field(default_factory=list)
```

ABC cũng **không thay đổi**:

```python
# src/drivevision/perception/base.py (tham khảo — không sửa)

class LaneDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> List[Lane]:
        ...
```

### 6.2 Sơ đồ luồng xử lý — Hướng (A) Classical

```
Frame.image (BGR HxWx3)
        │
        ▼
┌───────────────────────────────┐
│  1. Tiền xử lý ảnh            │
│     a) BGR → HLS              │
│     b) Mask trắng (S thấp,    │
│        L cao) + Mask vàng     │
│        (H trong dải vàng)     │
│     c) Kết hợp mask → color_  │
│        filtered               │
│     d) BGR → Grayscale        │
│     e) Gaussian blur          │
│     f) Canny edge             │
│     g) Kết hợp Canny + color  │
└──────────────┬────────────────┘
               │ combined_edges
               ▼
┌───────────────────────────────┐
│  2. ROI — Hình thang          │
│     Đỉnh trên: (0.40w,0.60h) │
│              (0.60w,0.60h)    │
│     Đáy dưới: (0.05w, h)     │
│              (0.95w, h)       │
│     fillPoly → masked_edges   │
└──────────────┬────────────────┘
               │ masked_edges
               ▼
┌───────────────────────────────────────────────────────┐
│  3a. Không dùng perspective (mặc định)                │
│      HoughLinesP(rho=1, theta=π/180, thresh=40,       │
│                  minLen=40, maxGap=100)               │
│      → raw segments                                   │
│      Lọc theo slope |m| ∈ [0.4, 5.0]                 │
│      Tách trái (m<0, x<cx) / phải (m>0, x>cx)        │
│      Collect points từng bên                          │
│      polyfit(y, x, deg=1) cho mỗi bên                 │
│      → coefficients (a, b) với x = a*y + b            │
│      Reconstruct điểm từ y_bottom → y_top             │
└──────────────┬────────────────────────────────────────┘
               │   (hoặc)
┌──────────────▼────────────────────────────────────────┐
│  3b. Dùng perspective transform (use_perspective=True) │
│      a) warpPerspective → bird's-eye view              │
│      b) Histogram ở nửa dưới → tìm peak trái/phải     │
│      c) Sliding window (9 cửa sổ, margin=100px)        │
│         thu thập pixel nonzero trong mỗi cửa sổ       │
│      d) polyfit(y_nonzero, x_nonzero, deg=2)           │
│         → a, b, c với x = a*y² + b*y + c              │
│      e) warpPerspective inverse → trở về image coords  │
└──────────────┬────────────────────────────────────────┘
               │ raw_lanes (List[Lane], points chưa mượt)
               ▼
┌───────────────────────────────┐
│  4. Temporal Smoothing        │
│     deque(maxlen=BUFFER_SIZE) │
│     EMA: coeff_smooth =       │
│       α * coeff_new +         │
│       (1-α) * coeff_prev      │
│     Reconstruct điểm từ       │
│     coeff_smooth              │
└──────────────┬────────────────┘
               │ smoothed_lanes (List[Lane])
               ▼
           RETURN List[Lane]
```

### 6.3 Sơ đồ luồng — Hướng (B) Model

```
Frame.image (BGR)
        │
        ▼
┌─────────────────────────────────┐
│  Tiền xử lý theo yêu cầu model  │
│  (resize, normalize, BGR→RGB)   │
└──────────────┬──────────────────┘
               │ tensor [1, 3, H, W]
               ▼
┌─────────────────────────────────┐
│  ModelLaneDetector.model        │
│  (YOLOP / HybridNets / UFLD)    │
│  Tải từ models/weights/lane.pt  │
└──────────────┬──────────────────┘
               │ raw output (mask hoặc anchor)
               ▼
┌─────────────────────────────────────────────────────┐
│  Hậu xử lý tuỳ theo loại model:                    │
│  - Segmentation mask: findContours → polyline       │
│  - UFLD row-anchor: reconstruct (x, y) từ anchor   │
│  - YOLOP lane mask: threshold → connected component │
│    → mỗi component là một Lane                     │
└──────────────┬──────────────────────────────────────┘
               │ List[Lane] (cùng interface)
               ▼
           RETURN List[Lane]
```

### 6.4 Tại sao Lane không phải Detection

```
Detection.bbox  → hình chữ nhật [x1,y1,x2,y2]  → rigid object
Lane.points     → [(x0,y0),(x1,y1),...,(xN,yN)] → đường liên tục

YOLO tìm đối tượng có texture/shape rõ ràng.
Lane detection cần:
  - Phân tích màu (trắng/vàng) theo không gian màu HLS
  - Hình học đường thẳng/cong dài
  - Xử lý đứt đoạn (vạch đứt, bị che)
  → Cần pipeline hoàn toàn khác.

Điểm gộp duy nhất: multitask model (YOLOP/HybridNets) xử lý
cả hai trong 1 forward pass. Nhưng interface ra vẫn tách biệt:
- SceneState.detections (từ bbox head)
- SceneState.lanes (từ lane head)
```

---

## 7. Công việc chi tiết

### Task 3.1 — Hoàn thiện `ClassicalLaneDetector`

**Mục đích:** Nâng cấp bản skeleton hiện tại (chỉ có grayscale + Canny cơ bản) thành pipeline đầy đủ với HLS filter, polynomial fit, và temporal smoothing.

**File:** `src/drivevision/perception/lane.py`

**Các bước:**

1. Thêm HLS color masking để tăng khả năng phát hiện vạch trắng và vàng trong điều kiện ánh sáng khác nhau.
2. Kết hợp color mask với Canny edges.
3. Thay thế việc lưu raw Hough segments bằng `np.polyfit` degree 1 (hoặc 2 với perspective).
4. Implement temporal smoothing dùng `collections.deque` và Exponential Moving Average.
5. Thêm flag `use_perspective` để bật bird's-eye view + sliding window.

**Pseudocode / skeleton đầy đủ:**

```python
# src/drivevision/perception/lane.py

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..types import Frame, Lane, LaneSide
from .base import LaneDetector

log = logging.getLogger("drivevision.lane")

# Số frame giữ trong buffer smoothing
_SMOOTH_BUFFER = 7
# Hệ số EMA: α cao → phản ứng nhanh nhưng rung nhiều; α thấp → mượt nhưng lag
_EMA_ALPHA = 0.3


def _hls_color_mask(bgr: np.ndarray) -> np.ndarray:
    """Trả về binary mask nhấn mạnh pixel trắng và vàng (vạch làn đường).

    Trắng: Lightness cao (> 200), Saturation thấp (< 60).
    Vàng:  Hue trong dải vàng (15–35 trên thang 0–180 của OpenCV),
           Saturation > 80, Lightness > 80.
    """
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    H, L, S = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]

    # Mask trắng
    white_mask = (L > 200) & (S < 60)

    # Mask vàng
    yellow_mask = (H >= 15) & (H <= 35) & (S > 80) & (L > 80)

    combined = (white_mask | yellow_mask).astype(np.uint8) * 255
    return combined


def _roi_trapezoid(
    shape: Tuple[int, int],
    top_ratio: float = 0.60,
    top_width_ratio: float = 0.20,
    bottom_pad_ratio: float = 0.05,
) -> np.ndarray:
    """Tạo mask hình thang (ROI) cho phần đường phía trước.

    top_ratio:         y của đỉnh trên tính theo chiều cao ảnh (0=trên, 1=dưới).
    top_width_ratio:   nửa chiều rộng đỉnh trên / chiều rộng ảnh.
    bottom_pad_ratio:  padding ngang ở đáy / chiều rộng ảnh.
    """
    h, w = shape
    cx = w / 2.0
    y_top = int(h * top_ratio)
    y_bot = h

    pts = np.array([[
        (int(cx - (0.5 - bottom_pad_ratio) * w), y_bot),
        (int(cx - top_width_ratio * w),           y_top),
        (int(cx + top_width_ratio * w),           y_top),
        (int(cx + (0.5 - bottom_pad_ratio) * w), y_bot),
    ]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def _separate_lines(
    lines: np.ndarray,
    width: int,
    slope_min: float = 0.4,
    slope_max: float = 5.0,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Tách Hough segments thành trái và phải dựa trên slope và vị trí x.

    Quy ước ảnh OpenCV: y tăng xuống dưới.
    - Làn trái: slope âm (đường chạy từ trái dưới → phải trên), x trung bình < cx.
    - Làn phải: slope dương, x trung bình > cx.
    """
    cx = width / 2.0
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < slope_min or abs(slope) > slope_max:
            continue  # lọc nhiễu ngang và đường gần thẳng đứng
        mid_x = (x1 + x2) / 2.0
        if slope < 0 and mid_x < cx:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > cx:
            right.append((x1, y1, x2, y2))
    return left, right


def _fit_lane_points(
    segments: List[Tuple[int, int, int, int]],
    h: int,
    y_top_ratio: float = 0.60,
    deg: int = 1,
) -> Optional[List[Tuple[float, float]]]:
    """Gom tất cả điểm đầu/cuối từ segments, polyfit theo chiều dọc.

    Fit theo dạng x = f(y) thay vì y = f(x) để tránh vấn đề đường thẳng đứng.
    Trả về None nếu không đủ điểm.
    """
    if not segments:
        return None

    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]

    if len(xs) < 2:
        return None

    try:
        coeffs = np.polyfit(ys, xs, deg)  # x = poly(y)
    except np.linalg.LinAlgError:
        return None

    y_bot = h
    y_top = int(h * y_top_ratio)
    y_vals = np.linspace(y_top, y_bot, num=20, dtype=np.float32)
    x_vals = np.polyval(coeffs, y_vals)

    # Lọc điểm ngoài biên ảnh
    valid = (x_vals >= 0) & (x_vals <= 2000)  # lỏng, clip sẽ xử lý ở annotator
    if valid.sum() < 2:
        return None

    return list(zip(x_vals[valid].tolist(), y_vals[valid].tolist()))


# ---------------------------------------------------------------------------
# Perspective transform helpers (dùng khi use_perspective=True)
# ---------------------------------------------------------------------------

def _get_perspective_matrices(
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính M và M_inv cho warpPerspective (bird's-eye view).

    Nguồn (src): hình thang mặt đường nhìn từ camera.
    Đích (dst):  hình chữ nhật tương ứng nhìn từ trên xuống.

    Các tỉ lệ được chọn phù hợp với camera trước xe thông thường.
    Cần hiệu chỉnh lại cho camera cụ thể.
    """
    h, w = shape
    src = np.float32([
        [w * 0.43, h * 0.65],
        [w * 0.57, h * 0.65],
        [w * 0.90, h * 0.95],
        [w * 0.10, h * 0.95],
    ])
    dst = np.float32([
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.75, h],
        [w * 0.25, h],
    ])
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


def _sliding_window_fit(
    binary_warped: np.ndarray,
    n_windows: int = 9,
    margin: int = 100,
    min_pix: int = 50,
    deg: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Sliding window lane fit trên bird's-eye binary image.

    Trả về (left_coeffs, right_coeffs) dạng x = poly(y), hoặc None nếu không tìm được.
    """
    h, w = binary_warped.shape

    # Histogram để tìm điểm xuất phát
    histogram = np.sum(binary_warped[h // 2:, :], axis=0)
    midpoint  = w // 2
    left_x_base  = int(np.argmax(histogram[:midpoint]))
    right_x_base = int(np.argmax(histogram[midpoint:])) + midpoint

    win_height = h // n_windows
    nonzero    = binary_warped.nonzero()
    nonzero_y  = np.array(nonzero[0])
    nonzero_x  = np.array(nonzero[1])

    left_current  = left_x_base
    right_current = right_x_base
    left_lane_inds:  List[np.ndarray] = []
    right_lane_inds: List[np.ndarray] = []

    for win in range(n_windows):
        y_low  = h - (win + 1) * win_height
        y_high = h - win * win_height

        # Cửa sổ trái
        xl_low, xl_high = left_current  - margin, left_current  + margin
        # Cửa sổ phải
        xr_low, xr_high = right_current - margin, right_current + margin

        good_left  = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                      (nonzero_x >= xl_low) & (nonzero_x < xl_high)).nonzero()[0]
        good_right = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                      (nonzero_x >= xr_low) & (nonzero_x < xr_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left)  > min_pix:
            left_current  = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > min_pix:
            right_current = int(np.mean(nonzero_x[good_right]))

    left_inds  = np.concatenate(left_lane_inds)  if left_lane_inds  else np.array([])
    right_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

    left_coeffs = right_coeffs = None
    if len(left_inds)  > min_pix * 2:
        left_coeffs  = np.polyfit(nonzero_y[left_inds],  nonzero_x[left_inds],  deg)
    if len(right_inds) > min_pix * 2:
        right_coeffs = np.polyfit(nonzero_y[right_inds], nonzero_x[right_inds], deg)

    return left_coeffs, right_coeffs


# ---------------------------------------------------------------------------
# ClassicalLaneDetector
# ---------------------------------------------------------------------------

class ClassicalLaneDetector(LaneDetector):
    """Pipeline cổ điển: HLS filter → Canny → ROI → Hough → polyfit → EMA smoothing.

    Tham số:
        use_perspective: Bật bird's-eye view + sliding window thay vì Hough thô.
        smooth_alpha:    Hệ số EMA (0..1). Lớn → nhanh nhạy, nhỏ → mượt hơn.
        smooth_buffer:   Số frame tối đa trong buffer (dùng để khởi tạo deque).
        poly_deg:        Bậc đa thức (1 = đường thẳng, 2 = cong).
    """

    def __init__(
        self,
        use_perspective: bool = False,
        smooth_alpha: float = _EMA_ALPHA,
        smooth_buffer: int = _SMOOTH_BUFFER,
        poly_deg: int = 1,
    ) -> None:
        self.use_perspective = use_perspective
        self.alpha   = smooth_alpha
        self.poly_deg = poly_deg

        # Lưu coefficients đã làm mượt cho từng bên
        self._left_coeffs:  Optional[np.ndarray] = None
        self._right_coeffs: Optional[np.ndarray] = None

        # Buffer lịch sử (tuỳ chọn dùng cho debug / window-average thay vì EMA)
        self._history: deque = deque(maxlen=smooth_buffer)

        # Cache perspective matrices (tính 1 lần)
        self._M: Optional[np.ndarray]     = None
        self._M_inv: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> List[Lane]:
        """Phát hiện làn đường trong một frame.

        Trả về list rỗng nếu không tìm được làn nào (không raise exception).
        """
        try:
            return self._detect_impl(frame)
        except Exception as exc:           # pragma: no cover
            log.warning("ClassicalLaneDetector error (frame %d): %s", frame.index, exc)
            return []

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _detect_impl(self, frame: Frame) -> List[Lane]:
        img = frame.image
        h, w = img.shape[:2]

        # 1. Tiền xử lý ảnh
        color_mask = _hls_color_mask(img)
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 150)
        # Kết hợp: ưu tiên màu hơn cạnh thuần
        combined = cv2.bitwise_or(edges, color_mask)

        # 2. ROI
        roi_mask = _roi_trapezoid((h, w))
        masked   = cv2.bitwise_and(combined, roi_mask)

        if self.use_perspective:
            return self._detect_perspective(img, masked, h, w)
        else:
            return self._detect_hough(masked, h, w)

    def _detect_hough(
        self, masked: np.ndarray, h: int, w: int
    ) -> List[Lane]:
        """Hough lines → polyfit → EMA smoothing."""
        lines = cv2.HoughLinesP(
            masked, rho=1, theta=np.pi / 180,
            threshold=40, minLineLength=40, maxLineGap=100,
        )
        if lines is None:
            # Không tìm thấy đường: reset coefficients để tránh drift
            # (không xoá ngay — giữ để hiển thị thêm vài frame)
            return self._build_from_coeffs(h, w)

        left_segs, right_segs = _separate_lines(lines, w)

        raw_left  = _fit_lane_points(left_segs,  h, poly_deg=self.poly_deg) if left_segs  else None  # type: ignore[call-arg]
        raw_right = _fit_lane_points(right_segs, h, poly_deg=self.poly_deg) if right_segs else None  # type: ignore[call-arg]

        # Không dùng poly_deg trong _fit_lane_points signature hiện tại,
        # truyền qua deg= parameter đã có trong hàm.
        # (Xem _fit_lane_points ở trên — có tham số deg=1)
        raw_left  = _fit_lane_points(left_segs,  h) if left_segs  else None
        raw_right = _fit_lane_points(right_segs, h) if right_segs else None

        # Cập nhật EMA coefficients (được tính ngược từ points)
        # Ở đây dùng approach đơn giản hơn: smooth points trực tiếp
        self._history.append((raw_left, raw_right))
        smoothed_left, smoothed_right = self._smooth_from_history()

        lanes: List[Lane] = []
        if smoothed_left:
            lanes.append(Lane(points=smoothed_left,  side=LaneSide.LEFT,  confidence=0.8))
        if smoothed_right:
            lanes.append(Lane(points=smoothed_right, side=LaneSide.RIGHT, confidence=0.8))
        return lanes

    def _detect_perspective(
        self, img: np.ndarray, masked: np.ndarray, h: int, w: int
    ) -> List[Lane]:
        """Bird's-eye view + sliding window + inverse warp."""
        if self._M is None:
            self._M, self._M_inv = _get_perspective_matrices((h, w))

        warped = cv2.warpPerspective(masked, self._M, (w, h))
        left_coeffs, right_coeffs = _sliding_window_fit(warped, deg=self.poly_deg)

        # EMA smoothing trên coefficients
        self._left_coeffs  = self._ema_coeffs(self._left_coeffs,  left_coeffs)
        self._right_coeffs = self._ema_coeffs(self._right_coeffs, right_coeffs)

        lanes: List[Lane] = []
        y_vals = np.linspace(int(h * 0.60), h, 20)

        for coeffs, side in [
            (self._left_coeffs,  LaneSide.LEFT),
            (self._right_coeffs, LaneSide.RIGHT),
        ]:
            if coeffs is None:
                continue
            x_vals = np.polyval(coeffs, y_vals)
            # Chuyển từ bird's-eye về image coordinates
            pts_warped = np.column_stack([x_vals, y_vals]).astype(np.float32)
            pts_warped = pts_warped.reshape(-1, 1, 2)
            pts_img    = cv2.perspectiveTransform(pts_warped, self._M_inv)
            pts_img    = pts_img.reshape(-1, 2)
            points = [(float(p[0]), float(p[1])) for p in pts_img]
            lanes.append(Lane(points=points, side=side, confidence=0.85))

        return lanes

    # ------------------------------------------------------------------
    # Smoothing helpers
    # ------------------------------------------------------------------

    def _ema_coeffs(
        self,
        prev: Optional[np.ndarray],
        curr: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Exponential moving average trên vector coefficients."""
        if curr is None:
            return prev  # giữ nguyên nếu frame này không detect được
        if prev is None:
            return curr
        return self.alpha * curr + (1 - self.alpha) * prev

    def _smooth_from_history(
        self,
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[Tuple[float, float]]]]:
        """Trung bình hoá danh sách points từ buffer lịch sử.

        Chỉ tính trung bình trên các frame có points hợp lệ.
        """
        left_valid  = [pts for pts, _ in self._history if pts is not None]
        right_valid = [pts for _, pts in self._history if pts is not None]

        def avg_points(
            point_lists: List[List[Tuple[float, float]]],
        ) -> Optional[List[Tuple[float, float]]]:
            if not point_lists:
                return None
            # Lấy danh sách ngắn nhất để căn chỉnh
            min_len = min(len(p) for p in point_lists)
            arr = np.array([[p[:min_len] for p in plist] for plist in point_lists])
            # arr shape: (n_frames, min_len, 2)
            mean = arr.mean(axis=0)  # shape (min_len, 2)
            return [(float(x), float(y)) for x, y in mean]

        return avg_points(left_valid), avg_points(right_valid)

    def _build_from_coeffs(self, h: int, w: int) -> List[Lane]:
        """Trả về lanes từ coefficients đã lưu (khi frame hiện tại không detect được).

        Cho phép hiển thị ổn định trong vài frame khi mất vạch tạm thời.
        """
        # Với Hough mode, coefficients không được lưu riêng,
        # nên dùng history smoothing
        smoothed_left, smoothed_right = self._smooth_from_history()
        lanes: List[Lane] = []
        if smoothed_left:
            lanes.append(Lane(points=smoothed_left,  side=LaneSide.LEFT,  confidence=0.4))
        if smoothed_right:
            lanes.append(Lane(points=smoothed_right, side=LaneSide.RIGHT, confidence=0.4))
        return lanes

    def reset(self) -> None:
        """Xoá lịch sử smoothing (dùng khi chuyển clip hoặc scene thay đổi đột ngột)."""
        self._history.clear()
        self._left_coeffs  = None
        self._right_coeffs = None
```

**Lưu ý & edge cases:**

- **Không có vạch:** `lines is None` → trả về lanes từ history (confidence thấp) hoặc `[]` nếu history cũng trống. **Không crash.**
- **Ban đêm / thiếu sáng:** HLS mask white sẽ bắt vạch phản quang. Nếu hoàn toàn tối, Canny không có gì để bắt → cũng trả `[]`.
- **Vạch đứt:** `maxLineGap=100` giúp nối các đoạn gần nhau. Sliding window còn hiệu quả hơn với vạch đứt vì dựa vào histogram.
- **Cua gắt:** polynomial bậc 1 sẽ không fit tốt; bật `poly_deg=2` hoặc `use_perspective=True + sliding_window`. Ghi chú rõ hạn chế này trong output confidence.
- **Bóng đổ mạnh:** HLS S-channel ít bị ảnh hưởng hơn BGR thô. Nhưng bóng mạnh vẫn có thể tạo cạnh giả trong Canny.
- **Slope quá gắt (|slope| > slope_max=5.0):** Loại bỏ — đây thường là nhiễu dọc cột.

---

### Task 3.2 — Hoàn thiện `ModelLaneDetector` stub

**Mục đích:** Cắm model học máy (sau Phase 8) vào cùng interface. Stub hiện tại cần được bổ sung để: (a) load model đúng cách, (b) hướng dẫn rõ ràng cho người implement sau.

**File:** `src/drivevision/perception/lane.py` (tiếp theo)

**Pseudocode:**

```python
class ModelLaneDetector(LaneDetector):
    """Phát hiện làn đường bằng model học máy.

    Weights được train/fine-tune trên Kaggle (Phase 8) và lưu tại
    ``models/weights/lane.pt``.

    Hỗ trợ hai loại model:
    - ``model_type="yolov8-seg"``: YOLOv8 segmentation, output mask per class.
    - ``model_type="yolop"``: YOLOP multitask, output lane_mask + drivable_mask.
    - ``model_type="ufld"``: Ultra-Fast-Lane-Detection v2, output row anchors.

    Chưa implement — raise NotImplementedError với hướng dẫn rõ ràng.
    Implement trong Phase 8 sau khi có weights.
    """

    def __init__(
        self,
        weights: str = "models/weights/lane.pt",
        model_type: str = "yolov8-seg",
        device: str = "cpu",
        conf: float = 0.5,
    ) -> None:
        self.weights    = weights
        self.model_type = model_type
        self.device     = device
        self.conf       = conf
        self._model     = None
        log.info(
            "ModelLaneDetector created (weights=%s, type=%s). "
            "Call _load() hoặc implement detect() sau Phase 8.",
            weights, model_type,
        )

    def _load(self) -> None:
        """Load model từ weights file.

        Implement theo từng loại model:

        YOLOv8-seg:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)

        YOLOP (torch.hub):
            self._model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=False)
            state = torch.load(self.weights, map_location=self.device)
            self._model.load_state_dict(state)
            self._model.eval()

        UFLD v2:
            Tải qua repo chính thức ultra-fast-lane-detection-v2,
            copy model định nghĩa, load state_dict.
        """
        raise NotImplementedError(
            f"ModelLaneDetector._load() chưa được implement.\n"
            f"Weights cần tại: {self.weights}\n"
            f"Dataset gợi ý: TuSimple / CULane / BDD100K lane\n"
            f"Train tại: Kaggle notebook (Phase 8)\n"
            f"Xem phase_8.md để biết chi tiết fine-tuning."
        )

    def detect(self, frame: Frame) -> List[Lane]:  # pragma: no cover
        """Chạy model trên frame và trả về List[Lane].

        Bước implement (sau Phase 8):
        1. Gọi self._load() nếu self._model is None.
        2. Tiền xử lý: resize về kích thước model yêu cầu, normalize.
        3. Forward pass.
        4. Hậu xử lý tuỳ model_type:
           - yolov8-seg: lấy mask class lane, findContours, lấy contour dài nhất
             mỗi bên, chia trái/phải theo x trung bình.
           - yolop: threshold lane_mask, connected components, mỗi component
             → Lane với points từ cv2.approxPolyDP.
           - ufld: row anchors → reconstruct (x, y) per lane line.
        5. Tạo List[Lane] với LaneSide.LEFT / RIGHT / UNKNOWN.
        6. Trả về list (có thể rỗng nếu không detect được).
        """
        raise NotImplementedError(
            "ModelLaneDetector.detect() chưa implement. "
            "Dùng ClassicalLaneDetector trong khi chờ Phase 8."
        )
```

---

### Task 3.3 — Cập nhật `Annotator` vẽ làn đường

**Mục đích:** Hiển thị kết quả lane detection trực quan — polyline màu + tô vùng giữa hai làn.

**File:** `src/drivevision/viz/annotator.py`

**Pseudocode:**

```python
# src/drivevision/viz/annotator.py

from __future__ import annotations

import cv2
import numpy as np

from ..types import Lane, LaneSide, PipelineResult, RiskLevel

# Màu sắc (BGR)
_LANE_COLOR = {
    LaneSide.LEFT:    (0, 255, 255),   # vàng cyan cho làn trái
    LaneSide.RIGHT:   (0, 255, 0),     # xanh lá cho làn phải
    LaneSide.UNKNOWN: (200, 200, 200), # xám cho không xác định
}
_EGO_AREA_COLOR  = (0, 255, 0)   # xanh lá (vùng ego-lane an toàn)
_LANE_THICKNESS  = 3
_LANE_ALPHA      = 0.35           # độ trong suốt khi tô vùng

class Annotator:
    def draw(self, result: PipelineResult) -> np.ndarray:
        frame = result.frame.image.copy()
        scene = result.scene

        # --- 1. Vẽ tracks (bounding boxes + track id) ---
        for track in scene.tracks:
            bbox = track.detection.bbox
            x1, y1, x2, y2 = bbox.as_int()
            color = (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{track.track_id} {track.detection.class_name}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # --- 2. Vẽ làn đường ---
        frame = self._draw_lanes(frame, scene.lanes)

        # --- 3. HUD: risk + decision ---
        frame = self._draw_hud(frame, result)

        return frame

    def _draw_lanes(
        self,
        img: np.ndarray,
        lanes: list[Lane],
    ) -> np.ndarray:
        """Vẽ polyline cho từng làn và tô vùng ego-lane giữa trái/phải."""
        if not lanes:
            return img

        overlay = img.copy()

        # Vẽ polyline từng làn
        for lane in lanes:
            if len(lane.points) < 2:
                continue
            color = _LANE_COLOR.get(lane.side, _LANE_COLOR[LaneSide.UNKNOWN])
            pts = np.array(lane.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=False,
                          color=color, thickness=_LANE_THICKNESS,
                          lineType=cv2.LINE_AA)

        # Tô vùng giữa làn trái và phải
        left_lane  = next((l for l in lanes if l.side == LaneSide.LEFT),  None)
        right_lane = next((l for l in lanes if l.side == LaneSide.RIGHT), None)

        if left_lane and right_lane and len(left_lane.points) >= 2 and len(right_lane.points) >= 2:
            # Nối các điểm: bên trái đi từ trên xuống, bên phải từ dưới lên
            left_pts  = np.array(left_lane.points,          dtype=np.int32)
            right_pts = np.array(right_lane.points[::-1],   dtype=np.int32)
            polygon   = np.vstack([left_pts, right_pts]).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [polygon], color=_EGO_AREA_COLOR)

        # Blend với alpha
        cv2.addWeighted(overlay, _LANE_ALPHA, img, 1 - _LANE_ALPHA, 0, img)

        # Vẽ lại polyline đặc lên trên (không bị mờ)
        for lane in lanes:
            if len(lane.points) < 2:
                continue
            color = _LANE_COLOR.get(lane.side, _LANE_COLOR[LaneSide.UNKNOWN])
            pts = np.array(lane.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False,
                          color=color, thickness=_LANE_THICKNESS,
                          lineType=cv2.LINE_AA)

        return img

    def _draw_hud(
        self, img: np.ndarray, result: PipelineResult
    ) -> np.ndarray:
        """Vẽ HUD góc trên trái: risk level + decision."""
        lines = []
        if result.risk:
            lines.append(f"Risk: {result.risk.level.value.upper()}")
        if result.decision:
            lines.append(f"Action: {result.decision.action.value}")
        lines.append(f"Lanes: {len(result.scene.lanes)}")

        risk_colors = {
            "safe":    (0, 200, 0),
            "caution": (0, 200, 255),
            "warning": (0, 100, 255),
            "danger":  (0, 0, 255),
        }
        level_str = result.risk.level.value if result.risk else "safe"
        hud_color = risk_colors.get(level_str, (255, 255, 255))

        for i, line in enumerate(lines):
            y = 28 + i * 22
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, hud_color, 2, cv2.LINE_AA)
        return img
```

---

### Task 3.4 — Viết test `tests/test_lane.py`

**Mục đích:** Đảm bảo `ClassicalLaneDetector` không crash trên các trường hợp đặc biệt và phân loại trái/phải đúng trên ảnh synthetic.

**File:** `tests/test_lane.py`

**Pseudocode:**

```python
# tests/test_lane.py

import numpy as np
import pytest

from drivevision.perception.lane import ClassicalLaneDetector
from drivevision.types import Frame, LaneSide


def _make_frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(index=index, timestamp=float(index) / 30.0, image=image)


def _blank_frame(h: int = 480, w: int = 640) -> Frame:
    """Ảnh đen hoàn toàn — không có vạch nào."""
    return _make_frame(np.zeros((h, w, 3), dtype=np.uint8))


def _synthetic_lane_frame(h: int = 480, w: int = 640) -> Frame:
    """Ảnh xám với 2 đường trắng hội tụ về trung tâm (mô phỏng làn đường)."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 80  # nền xám tối

    # Làn trái: từ (50, h-1) → (w//2 - 20, h//2)
    # Làn phải: từ (w-50, h-1) → (w//2 + 20, h//2)
    import cv2
    cv2.line(img, (50, h - 1),         (w // 2 - 20, h // 2), (255, 255, 255), 12)
    cv2.line(img, (w - 50, h - 1),     (w // 2 + 20, h // 2), (255, 255, 255), 12)
    return _make_frame(img)


class TestClassicalLaneDetector:

    def test_blank_frame_returns_empty(self):
        """Ảnh trống → không crash, trả về list rỗng."""
        detector = ClassicalLaneDetector()
        lanes = detector.detect(_blank_frame())
        assert isinstance(lanes, list)
        # Có thể trả về [] hoặc lanes từ history (trống ở frame 0)
        # Quan trọng: không raise exception

    def test_synthetic_detects_two_sides(self):
        """Ảnh có 2 vạch trắng rõ → phải tìm được ít nhất 1 làn."""
        detector = ClassicalLaneDetector()
        frame = _synthetic_lane_frame()
        lanes = detector.detect(frame)
        # Chấp nhận 1-2 làn
        assert len(lanes) >= 1

    def test_lane_has_valid_points(self):
        """Mỗi Lane phải có ít nhất 2 điểm."""
        detector = ClassicalLaneDetector()
        lanes = detector.detect(_synthetic_lane_frame())
        for lane in lanes:
            assert len(lane.points) >= 2

    def test_side_classification(self):
        """Làn phát hiện được phải có side là LEFT hoặc RIGHT (không UNKNOWN)."""
        detector = ClassicalLaneDetector()
        lanes = detector.detect(_synthetic_lane_frame())
        sides = {lane.side for lane in lanes}
        assert LaneSide.LEFT in sides or LaneSide.RIGHT in sides

    def test_confidence_in_range(self):
        detector = ClassicalLaneDetector()
        lanes = detector.detect(_synthetic_lane_frame())
        for lane in lanes:
            assert 0.0 <= lane.confidence <= 1.0

    def test_points_within_image_bounds(self):
        """Tất cả điểm phải nằm trong hoặc gần giới hạn ảnh."""
        h, w = 480, 640
        detector = ClassicalLaneDetector()
        lanes = detector.detect(_synthetic_lane_frame(h, w))
        for lane in lanes:
            for x, y in lane.points:
                assert -50 <= x <= w + 50, f"x={x} out of bounds"
                assert   0 <= y <= h,      f"y={y} out of bounds"

    def test_multiple_frames_smoothing(self):
        """Chạy 10 frame liên tiếp không crash; confidence ổn định."""
        detector = ClassicalLaneDetector(smooth_buffer=5)
        frame = _synthetic_lane_frame()
        for i in range(10):
            f = Frame(index=i, timestamp=i / 30.0, image=frame.image)
            lanes = detector.detect(f)
            assert isinstance(lanes, list)

    def test_dark_frame_no_crash(self):
        """Ảnh rất tối (ban đêm) → không crash."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 5  # cực tối
        detector = ClassicalLaneDetector()
        lanes = detector.detect(_make_frame(img))
        assert isinstance(lanes, list)

    def test_reset_clears_history(self):
        detector = ClassicalLaneDetector(smooth_buffer=5)
        for i in range(5):
            f = Frame(index=i, timestamp=i / 30.0, image=_synthetic_lane_frame().image)
            detector.detect(f)
        detector.reset()
        assert len(detector._history) == 0

    def test_perspective_mode_no_crash(self):
        """use_perspective=True không crash trên ảnh synthetic."""
        detector = ClassicalLaneDetector(use_perspective=True)
        lanes = detector.detect(_synthetic_lane_frame())
        assert isinstance(lanes, list)


class TestModelLaneDetectorStub:
    def test_raises_not_implemented(self):
        from drivevision.perception.lane import ModelLaneDetector
        detector = ModelLaneDetector()
        with pytest.raises(NotImplementedError):
            detector.detect(_blank_frame())
```

---

### Task 3.5 — Cập nhật cấu hình

**Mục đích:** Thêm sub-keys vào `perception.lane` để `ClassicalLaneDetector` đọc được từ config.

Xem §8 bên dưới.

---

### Task 3.6 — Cập nhật `builder.py` để truyền tham số

**Mục đích:** Truyền các tham số mới (`use_perspective`, `smooth_alpha`, `poly_deg`) từ config vào `ClassicalLaneDetector`.

**File:** `src/drivevision/pipeline/builder.py`

Thay đổi nhỏ tại block `perception.lane.enabled` (dòng 67–71 hiện tại):

```python
# Trước (dòng 67-71 hiện tại):
if get_path(cfg, "perception.lane.enabled"):
    from ..perception.lane import ClassicalLaneDetector, ModelLaneDetector
    method = get_path(cfg, "perception.lane.method", "classical")
    lane_detector = ModelLaneDetector() if method == "model" else ClassicalLaneDetector()

# Sau (thay thế):
if get_path(cfg, "perception.lane.enabled"):
    from ..perception.lane import ClassicalLaneDetector, ModelLaneDetector
    method = get_path(cfg, "perception.lane.method", "classical")
    if method == "model":
        lane_detector = ModelLaneDetector(
            weights=get_path(cfg, "perception.lane.weights", "models/weights/lane.pt"),
            model_type=get_path(cfg, "perception.lane.model_type", "yolov8-seg"),
            device=get_path(cfg, "perception.lane.device", "cpu"),
        )
    else:
        lane_detector = ClassicalLaneDetector(
            use_perspective=get_path(cfg, "perception.lane.use_perspective", False),
            smooth_alpha=get_path(cfg, "perception.lane.smooth_alpha", 0.3),
            smooth_buffer=get_path(cfg, "perception.lane.smooth_buffer", 7),
            poly_deg=get_path(cfg, "perception.lane.poly_deg", 1),
        )
```

---

## 8. Thay đổi cấu hình

**File:** `configs/default.yaml`

Thay khối `perception.lane` hiện tại:

```yaml
# Trước:
  lane:
    enabled: false
    method: classical    # classical | model

# Sau:
  lane:
    enabled: true          # bật khi bắt đầu Phase 3
    method: classical      # classical | model

    # --- Tham số Classical ---
    use_perspective: false  # true = bird's-eye view + sliding window
    smooth_alpha: 0.30      # EMA alpha (0=không update, 1=không smooth)
    smooth_buffer: 7        # số frame lịch sử giữ trong deque
    poly_deg: 1             # bậc đa thức (1=thẳng, 2=cong)

    # --- Tham số Model (dùng khi method: model, Phase 8) ---
    weights: models/weights/lane.pt
    model_type: yolov8-seg  # yolov8-seg | yolop | ufld
    device: cpu             # cpu | cuda (Kaggle dùng cuda khi train)
```

---

## 9. Kiểm thử

### 9.1 Unit test (tự động)

```bash
# Chạy tất cả test lane
cd /home/quan/DriveVision
python -m pytest tests/test_lane.py -v

# Chạy với coverage
python -m pytest tests/test_lane.py --cov=src/drivevision/perception/lane --cov-report=term-missing
```

Target coverage: ≥ 80% cho `ClassicalLaneDetector`.

### 9.2 Integration test trên ảnh/video thật

```bash
# Test trên ảnh tĩnh BDD100K
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --source data/samples/bdd100k_sample.jpg

# Test trên video clip ngắn (30 frame)
drivevision --config configs/default.yaml --source data/samples/sample.mp4
```

### 9.3 Kiểm tra thủ công kết quả visualisation

1. Bật `output.display: true` trong `configs/default.yaml` (nếu có môi trường GUI).
2. Hoặc lưu video: `output.save_path: output/annotated_lane.mp4`.
3. Kiểm tra: làn trái màu cyan, làn phải màu xanh lá, vùng ego-lane tô nhạt.

### 9.4 Test edge cases thủ công

| Kịch bản | Input | Kết quả mong đợi |
|---|---|---|
| Ảnh toàn đen | `np.zeros((480,640,3))` | `[]`, không exception |
| Ảnh toàn trắng | `np.ones * 255` | `[]` hoặc lanes nhiễu, không exception |
| 1 làn bị che | Vẽ hình chữ nhật đen che làn trái | Chỉ tìm được làn phải |
| Vạch đứt | Vẽ vạch dash 30px on / 30px off | Vẫn tìm được với maxLineGap=100 |
| Cua gắt 45° | Ảnh có đường nghiêng 45° | Phát hiện được, confidence thấp hơn |
| Video ban đêm | BDD100K nighttime clip | Không crash; có thể ít làn hơn |

---

## 10. Tiêu chí hoàn thành

- [ ] `ClassicalLaneDetector.detect()` có HLS color filter và gọi được không crash
- [ ] `ClassicalLaneDetector` có polynomial fit (polyfit degree 1)
- [ ] `ClassicalLaneDetector` có temporal smoothing (EMA hoặc deque average)
- [ ] `ClassicalLaneDetector` có `use_perspective` mode (bird's-eye + sliding window)
- [ ] `ClassicalLaneDetector.reset()` hoạt động đúng
- [ ] Tất cả edge case (ảnh đen, không vạch, ban đêm) trả về `[]` không crash
- [ ] `ModelLaneDetector` có stub sạch với `NotImplementedError` + message hướng dẫn
- [ ] `Annotator.draw()` vẽ polyline làn và tô vùng ego-lane
- [ ] `Annotator.draw()` có HUD hiển thị số làn + risk + decision
- [ ] `tests/test_lane.py` chạy pass 100% (≥ 9 test case)
- [ ] Coverage `perception/lane.py` ≥ 80%
- [ ] `configs/default.yaml` có đủ sub-keys mới
- [ ] `pipeline/builder.py` truyền tham số mới vào `ClassicalLaneDetector`
- [ ] `pipeline.process()` trả `PipelineResult` với `scene.lanes` được điền
- [ ] Video output có visualisation làn đường nhìn thấy rõ
- [ ] Không thay đổi interface `Lane`, `LaneSide`, `Frame`, `SceneState`, `LaneDetector` ABC

---

## 11. Rủi ro & cách xử lý

| Rủi ro | Khả năng | Tác động | Cách xử lý |
|---|---|---|---|
| HLS mask quá nhạy → nhiều false positive | Cao (tuỳ video) | Vẽ nhầm làn | Điều chỉnh ngưỡng HLS, thêm min_pix filter |
| polyfit rank warning khi không đủ điểm | Trung bình | Warning spam | Wrap trong try/except; require ≥ 4 điểm trước khi fit |
| Slope classification nhầm trái/phải khi cua | Trung bình | Làn bị gán sai side | Kết hợp slope VÀ x position; thêm UNKNOWN khi không rõ |
| Temporal smoothing lag khi scene thay đổi nhanh | Thấp | Làn "ảo" hiển thị thêm vài frame | Giảm `smooth_buffer`, tăng `smooth_alpha`; reset() khi detect scene cut |
| Perspective transform src/dst không phù hợp camera | Cao (camera cụ thể) | Bird's-eye bị méo | Document rõ là cần calibrate; mặc định `use_perspective: false` |
| Sliding window sai nếu nhiều nhiễu | Trung bình | Nhiều lane rác | Thêm min_pix_per_window filter; require ≥ 3 window có pixel |
| ModelLaneDetector weights chưa tồn tại | Chắc chắn ở Phase 3 | ImportError nếu method=model | Builder catch NotImplementedError, fallback về classical + log warning |
| Video clip quá ngắn để test smoothing | Thấp | Test không đủ frame | Dùng `source.loop: true` để lặp lại |

---

## 12. Hiệu năng & tài nguyên

### Classical mode

| Metric | Ước tính | Ghi chú |
|---|---|---|
| Thời gian xử lý / frame | 5–20 ms | CPU only, ảnh 640×480 |
| RAM | < 50 MB | Deque buffer nhỏ |
| GPU | Không cần | Chạy trên bất kỳ máy nào |
| Tốc độ tương đương | 50–200 FPS nếu chạy riêng | Pipeline đầy đủ sẽ chậm hơn do YOLO |

### Model mode (Phase 8, Kaggle)

| Model | Inference time (T4) | RAM | Ghi chú |
|---|---|---|---|
| YOLOv8n-seg | ~15 ms/frame | ~2 GB VRAM | Nhẹ nhất |
| YOLOP | ~25 ms/frame | ~3 GB VRAM | 3-in-1 output |
| HybridNets | ~20 ms/frame | ~2.5 GB VRAM | Cân bằng tốt |
| UFLD v2 | ~10 ms/frame | ~1.5 GB VRAM | Nhanh nhất nhưng lane only |

### Mục tiêu Phase 3 (classical only)

- Toàn pipeline (YOLO + tracker + lane classical) < 100 ms/frame trên CPU → ~ 10 FPS — đủ cho demo.
- Không đặt mục tiêu real-time; ưu tiên accuracy và demo chất lượng.

---

## 13. Sản phẩm bàn giao

| Sản phẩm | File / Path | Mô tả |
|---|---|---|
| `ClassicalLaneDetector` hoàn chỉnh | `src/drivevision/perception/lane.py` | HLS filter + polyfit + EMA smoothing + perspective mode |
| `ModelLaneDetector` stub sạch | `src/drivevision/perception/lane.py` | Hướng dẫn rõ ràng cho Phase 8 |
| `Annotator` có vẽ lane | `src/drivevision/viz/annotator.py` | Polyline + filled polygon + HUD |
| Config cập nhật | `configs/default.yaml` | Sub-keys đầy đủ cho lane |
| Builder cập nhật | `src/drivevision/pipeline/builder.py` | Truyền tham số lane vào constructor |
| Test suite | `tests/test_lane.py` | ≥ 9 test case, coverage ≥ 80% |
| Video demo | `output/annotated_lane.mp4` (nếu có clip input) | Demo visualisation |

---

## 14. Điểm nhấn cho Portfolio

### 14.1 Kỹ thuật nổi bật

- **Hai hướng triển khai** cùng interface: portfolio tốt phải cho thấy cả phương pháp cổ điển lẫn ML, không chỉ dùng black-box model.
- **Temporal smoothing**: biểu hiện hiểu biết về ổn định hệ thống real-world — không phải chỉ xử lý từng frame riêng lẻ.
- **Perspective transform + sliding window**: kỹ thuật kinh điển từ Udacity Self-Driving Car Nanodegree — nhà tuyển dụng auto-detection quen biết, tạo ấn tượng tốt.
- **HLS color space**: cho thấy hiểu biết về computer vision ngoài grayscale đơn thuần — xử lý được điều kiện ánh sáng đa dạng.
- **Multitask model path (Phase 8)**: YOLOP/HybridNets cho thấy hiểu biết về trade-off giữa specialized vs. multitask models trong production.

### 14.2 Demo visual ấn tượng

- Video với làn đường được tô màu xanh lá + đường polyline vàng + HUD risk level.
- So sánh side-by-side: classical vs. model (sau Phase 8).
- GIF ngắn (5–10 giây) trên README: đủ để gây ấn tượng trong 3 giây khi recruiter lướt GitHub.

### 14.3 Điểm differentiation so với "lane detection tutorial"

Hầu hết tutorial dừng ở Hough cơ bản. DriveVision có thêm:
- Temporal smoothing (không giật).
- Polynomial fit thay vì raw segments.
- Tích hợp vào pipeline modular (không phải script đơn lẻ).
- Config-driven (bật/tắt bằng YAML, không sửa code).
- Test coverage.
- Lộ trình rõ ràng đến model-based.

---

## 15. Tham khảo

| Tên | Link | Ghi chú |
|---|---|---|
| Udacity Advanced Lane Finding Project | github.com/udacity/CarND-Advanced-Lane-Lines | Nguồn gốc sliding window + perspective transform |
| Ultra-Fast-Lane-Detection v2 | github.com/cfzd/Ultra-Fast-Lane-Detection-v2 | Kiến trúc row-anchor, nhẹ và nhanh |
| LaneNet (lanenet-lane-detection) | github.com/MaybeShewill-CV/lanenet-lane-detection | Instance segmentation từng làn |
| YOLOP | github.com/hustvl/YOLOP | Multitask: det + lane + drivable area, BDD100K |
| HybridNets | github.com/datvuthanh/HybridNets | Multitask nhẹ hơn, phù hợp Kaggle |
| TuSimple Dataset | github.com/TuSimple/tusimple-benchmark | Dataset lane detection tiêu chuẩn |
| CULane Dataset | github.com/XingangPan/SCNN | Dataset với cua gắt, che khuất |
| BDD100K Lane Annotations | bdd-data.eecs.berkeley.edu | 100K frames đa dạng thời tiết/giờ giấc |
| OpenCV HLS docs | docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html | Công thức chuyển đổi màu |
| numpy.polyfit | numpy.org/doc/stable/reference/generated/numpy.polyfit.html | API polyfit chính xác |

---

## 16. Checklist tổng kết

```
Phase 3 — Lane Detection
════════════════════════════════════════════════════════════

TRIỂN KHAI
─────────────────────────────────────────────────────────────
[ ] Task 3.1  src/drivevision/perception/lane.py
              ├─ [ ] _hls_color_mask() — white + yellow
              ├─ [ ] _roi_trapezoid() — tham số hoá đủ
              ├─ [ ] _separate_lines() — slope + x position
              ├─ [ ] _fit_lane_points() — polyfit degree 1/2
              ├─ [ ] _get_perspective_matrices() — M, M_inv
              ├─ [ ] _sliding_window_fit() — n_windows=9
              ├─ [ ] ClassicalLaneDetector.__init__() — params
              ├─ [ ] ClassicalLaneDetector.detect() — wrapper try/except
              ├─ [ ] ClassicalLaneDetector._detect_hough() — full pipeline
              ├─ [ ] ClassicalLaneDetector._detect_perspective() — bev mode
              ├─ [ ] ClassicalLaneDetector._ema_coeffs() — EMA smoothing
              ├─ [ ] ClassicalLaneDetector._smooth_from_history() — deque avg
              ├─ [ ] ClassicalLaneDetector._build_from_coeffs() — fallback
              └─ [ ] ClassicalLaneDetector.reset()

[ ] Task 3.2  ModelLaneDetector stub
              ├─ [ ] __init__() — weights, model_type, device, conf
              ├─ [ ] _load() — hướng dẫn 3 loại model
              └─ [ ] detect() — NotImplementedError + message

[ ] Task 3.3  src/drivevision/viz/annotator.py
              ├─ [ ] _draw_lanes() — polylines cho từng lane
              ├─ [ ] _draw_lanes() — filled polygon ego-area
              ├─ [ ] _draw_lanes() — alpha blend
              ├─ [ ] _draw_hud() — risk + decision + lane count
              └─ [ ] draw() — tích hợp tất cả

[ ] Task 3.4  tests/test_lane.py
              ├─ [ ] test_blank_frame_returns_empty
              ├─ [ ] test_synthetic_detects_two_sides
              ├─ [ ] test_lane_has_valid_points
              ├─ [ ] test_side_classification
              ├─ [ ] test_confidence_in_range
              ├─ [ ] test_points_within_image_bounds
              ├─ [ ] test_multiple_frames_smoothing
              ├─ [ ] test_dark_frame_no_crash
              ├─ [ ] test_reset_clears_history
              ├─ [ ] test_perspective_mode_no_crash
              └─ [ ] test_raises_not_implemented (ModelLaneDetector)

[ ] Task 3.5  configs/default.yaml — sub-keys lane đầy đủ

[ ] Task 3.6  src/drivevision/pipeline/builder.py
              └─ [ ] truyền tham số vào ClassicalLaneDetector constructor

KIỂM THỬ
─────────────────────────────────────────────────────────────
[ ] pytest tests/test_lane.py → 100% pass
[ ] coverage ≥ 80% cho perception/lane.py
[ ] Chạy pipeline trên video mẫu, không crash
[ ] Video output có visualisation làn đường

KHÔNG PHÁ VỠ
─────────────────────────────────────────────────────────────
[ ] types.py không bị sửa (Lane, LaneSide, Frame, SceneState)
[ ] perception/base.py không bị sửa (LaneDetector ABC)
[ ] pipeline/pipeline.py không bị sửa
[ ] Các phase khác (P1, P2, P4+) không bị ảnh hưởng

GIAO NỘPCHO PHASE TIẾP THEO
─────────────────────────────────────────────────────────────
[ ] SceneState.lanes luôn là List[Lane] (không None, không crash)
[ ] Mỗi Lane có points ≥ 2, side ∈ {LEFT, RIGHT, UNKNOWN}, confidence ∈ [0,1]
[ ] Phase 5 có thể gán ego-lane dựa trên Lane.side + Lane.points
[ ] ModelLaneDetector.detect() có thể được implement trong Phase 8
    mà không cần thay đổi interface
```
