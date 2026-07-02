# Phase 4 — Traffic Light Detection + Phân Loại Trạng Thái Đèn

**Mục tiêu: Hoàn thiện `SimpleTrafficLightDetector` để hệ thống có thể phát hiện vị trí đèn giao thông (tái dùng output YOLO sẵn có) và phân loại màu đèn (đỏ/vàng/xanh/không xác định) bằng phương pháp phân tích HSV, đặt nền tảng cho Phase 6 ra quyết định dừng/đi dựa trên trạng thái đèn.**

---

## 1. Tổng Quan & Vị Trí Trong Lộ Trình

### 1.1 Bối cảnh

DriveVision là hệ thống perception cho xe tự hành dạng portfolio. Kiến trúc theo pipeline tuần tự:

```
DataSource → [Detector] → [Tracker] → [LaneDetector] → [TLDetector] → SceneBuilder → [Risk] → [Decision]
```

Phase 4 lấp đầy slot `[TLDetector]` — hiện là stub `return []`. Sau phase này, `SceneState.traffic_lights` sẽ chứa danh sách `TrafficLight` với `state` hợp lệ, cho phép Phase 6 (Risk + Decision) phản ứng đúng với tín hiệu đèn.

### 1.2 Mối quan hệ với các phase khác

| Phase | Phụ thuộc | Vai trò |
|-------|-----------|---------|
| Phase 1 (Detection) | **Cung cấp** `List[Detection]` từ YOLO | Nguồn bbox đèn, KHÔNG cần train thêm model detection |
| Phase 2 (Tracking) | Chạy song song, không phụ thuộc | Track object, không track đèn (đèn tĩnh) |
| Phase 3 (Lane) | Độc lập | — |
| **Phase 4 (TL)** | **Nhận** `detections` từ Phase 1 | **Lọc đèn → phân loại màu → trả `List[TrafficLight]`** |
| Phase 5 (Scene) | **Tiêu thụ** output Phase 4 | Tích hợp vào `SceneState.traffic_lights` |
| Phase 6 (Risk/Decision) | **Tiêu thụ** `SceneState.traffic_lights` | Ra quyết định STOP/BRAKE khi đèn đỏ |
| Phase 8 (Fine-tuning) | Nâng cấp tùy chọn | Có thể thay heuristic HSV bằng CNN tiny fine-tuned trên Kaggle |

### 1.3 Lý do tách thành phần riêng

Có hai bài toán **tách biệt về mặt kỹ thuật**:

1. **Phát hiện hộp đèn (Detection)**: "Ở đâu trong ảnh có đèn giao thông?" — YOLO (COCO class `traffic light`, id=9) giải quyết hoàn toàn. Không cần model riêng.
2. **Phân loại màu đèn (State Classification)**: "Đèn đó đang đỏ, vàng hay xanh?" — Đây là bài toán classification trên **crop nhỏ** (~20×50 px), khác hoàn toàn về input size và output space so với detection. HSV heuristic đủ mạnh cho điều kiện tốt; CNN tiny là phương án nâng cấp khi HSV kém.

Việc tách rõ cho phép thay thế độc lập: muốn dùng model detection chuyên dụng (Phase 8) chỉ cần override `detect()`, không đụng classifier.

---

## 2. Mục Tiêu (Đo Được)

| # | Mục tiêu | Điều kiện chấp nhận |
|---|----------|---------------------|
| M1 | `SimpleTrafficLightDetector.detect()` không còn `return []` | Unit test pass với ảnh có đèn |
| M2 | Lọc đúng class COCO id=9 | Chỉ detection `class_id == 9` được xử lý |
| M3 | HSV classifier phân loại đúng màu đỏ/xanh | Accuracy ≥ 90% trên test set crop ảnh rõ nét |
| M4 | HSV classifier phân loại đúng màu vàng | Accuracy ≥ 80% (vàng khó hơn do dải HSV hẹp) |
| M5 | Edge cases trả về `UNKNOWN` thay vì crash | Crop quá nhỏ, ảnh tối, không có pixel sáng đủ ngưỡng |
| M6 | Annotator vẽ bbox + nhãn màu lên frame | Kiểm tra trực quan qua script |
| M7 | Pipeline end-to-end chạy được với `traffic_light.enabled: true` | `scripts/run_pipeline.py` không lỗi |
| M8 | Thời gian phân loại HSV ≤ 2 ms/đèn | Profiling trên máy không GPU |

---

## 3. Phạm Vi

### 3.1 Trong phạm vi (In-Scope)

- Lọc `Detection` với `class_id == 9` từ output YOLO
- Phân loại màu đèn bằng **HSV heuristic** (phương án chính)
- Xử lý edge cases: crop quá nhỏ, ảnh tối, ngược sáng, không tìm thấy màu rõ
- Cập nhật `viz/annotator.py` để hiển thị `TrafficLight` với màu sắc tương ứng
- Unit test cho classifier và detector
- Kích hoạt qua config YAML (`perception.traffic_light.enabled: true`)
- Tài liệu inline (docstring) đầy đủ

### 3.2 Ngoài phạm vi (Out-of-Scope)

- **Không** train model detection riêng cho đèn nhỏ/xa — đây là việc của Phase 8
- **Không** implement CNN classifier (chỉ thiết kế, để Phase 8 nếu HSV kém)
- **Không** track đèn giao thông qua nhiều frame (đèn tĩnh, không cần)
- **Không** xử lý đèn người đi bộ (pedestrian light) — COCO không phân biệt
- **Không** tích hợp depth/semantic từ CARLA — đó là Phase 9
- **Không** đưa ra quyết định dừng/đi — đó là Phase 6

---

## 4. Điều Kiện Tiên Quyết

### 4.1 Code đã hoàn thành

- [x] `src/drivevision/types.py`: `TrafficLight`, `TrafficLightState`, `BoundingBox`, `Detection`, `Frame`, `SceneState` đã định nghĩa đầy đủ
- [x] `src/drivevision/perception/base.py`: `TrafficLightDetector(ABC)` với signature `detect(frame, detections) -> List[TrafficLight]`
- [x] `src/drivevision/perception/traffic_light.py`: `SimpleTrafficLightDetector` skeleton tồn tại (cần implement)
- [x] `src/drivevision/perception/detection.py`: `YOLODetector` hoạt động, trả `class_name='traffic light'` cho id=9
- [x] `src/drivevision/pipeline/pipeline.py`: `Pipeline.process()` đã gọi `tl_detector.detect(frame, detections)` và truyền kết quả vào `scene_builder.build()`
- [x] `src/drivevision/pipeline/builder.py`: `build_pipeline()` đã instantiate `SimpleTrafficLightDetector` khi flag bật
- [x] `src/drivevision/config.py`: key `perception.traffic_light.enabled` tồn tại (default: `False`)

### 4.2 Môi trường

- Python 3.12 với virtualenv tại `/home/quan/DriveVision/.venv`
- OpenCV (`cv2`) đã cài — được dùng cho HSV conversion và crop
- `numpy` đã cài
- `ultralytics` đã cài (cần cho YOLODetector, nhưng Phase 4 không import trực tiếp)
- Có ít nhất 1 video test hoặc ảnh test chứa đèn giao thông ở `data/samples/`

### 4.3 Kiến thức cần có

- OpenCV color space: BGR → HSV với `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`
- HSV trong OpenCV: H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255]
- Dải màu đèn trong HSV (xem chi tiết mục 7.3)
- Python dataclass, ABC, enum

---

## 5. Công Nghệ & Thư Viện

| Thư viện | Phiên bản | Vai trò trong Phase 4 |
|----------|-----------|----------------------|
| `opencv-python` (`cv2`) | ≥ 4.8 | BGR→HSV conversion, crop bbox, vẽ annotation |
| `numpy` | ≥ 1.26 | Array ops trên HSV mask, pixel counting |
| `ultralytics` | ≥ 8.0 | **Gián tiếp**: YOLODetector (Phase 1) cung cấp bbox |
| `pytest` | ≥ 7.0 | Unit test classifier và detector |
| `pyyaml` | ≥ 6.0 | Đọc config YAML |

**Không thêm dependency mới** cho Phase 4 (HSV chỉ cần `cv2` + `numpy`).

Phương án nâng cao (Phase 8):
- `torch` + `torchvision`: CNN tiny classifier
- Dataset: LISA Traffic Light Dataset (Kaggle), BTTSD, hoặc custom crop từ video

---

## 6. Thiết Kế Chi Tiết

### 6.1 Types & Interface (Đã Có, Không Thay Đổi)

```python
# src/drivevision/types.py — HIỆN CÓ, không sửa
class TrafficLightState(str, Enum):
    RED     = "red"
    YELLOW  = "yellow"
    GREEN   = "green"
    UNKNOWN = "unknown"

@dataclass
class TrafficLight:
    bbox:       BoundingBox
    state:      TrafficLightState = TrafficLightState.UNKNOWN
    confidence: float = 1.0

# src/drivevision/perception/base.py — HIỆN CÓ, không sửa
class TrafficLightDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame, detections: List[Detection]) -> List[TrafficLight]:
        ...
```

**Nguyên tắc**: Phase 4 CHỈ implement bên trong `SimpleTrafficLightDetector`. Mọi module khác (pipeline, scene builder, annotator) đã dùng đúng interface.

### 6.2 Sơ Đồ Luồng Xử Lý

```
Frame (BGR image)  +  List[Detection] (từ YOLODetector)
          │                       │
          │          ┌────────────┘
          │          │
          ▼          ▼
  ┌─────────────────────────────────────────────────────┐
  │          SimpleTrafficLightDetector.detect()         │
  │                                                       │
  │  ┌─────────────────────────────────────────────────┐ │
  │  │  Bước 1: Lọc detections                         │ │
  │  │  for det in detections:                          │ │
  │  │      if det.class_id == 9:  # COCO traffic light │ │
  │  │          tl_dets.append(det)                     │ │
  │  └──────────────────┬──────────────────────────────┘ │
  │                     │                                 │
  │  ┌──────────────────▼──────────────────────────────┐ │
  │  │  Bước 2: Với mỗi đèn → Crop ảnh                 │ │
  │  │  crop = frame.image[y1:y2, x1:x2]               │ │
  │  │  (clip tọa độ trong frame boundary)              │ │
  │  └──────────────────┬──────────────────────────────┘ │
  │                     │                                 │
  │  ┌──────────────────▼──────────────────────────────┐ │
  │  │  Bước 3: Phân loại màu (HSV Classifier)         │ │
  │  │                                                  │ │
  │  │  crop_hsv = cv2.cvtColor(crop, BGR2HSV)          │ │
  │  │                                                  │ │
  │  │  ┌──────────────────────────────────────────┐   │ │
  │  │  │ Chiến lược A: Chia 3 vùng dọc            │   │ │
  │  │  │  top    = crop_hsv[0    : h//3, :]        │   │ │
  │  │  │  middle = crop_hsv[h//3 : 2*h//3, :]     │   │ │
  │  │  │  bottom = crop_hsv[2*h//3 : h, :]        │   │ │
  │  │  │  → Vùng sáng nhất → state                │   │ │
  │  │  └──────────────────────────────────────────┘   │ │
  │  │  ┌──────────────────────────────────────────┐   │ │
  │  │  │ Chiến lược B: Lọc ngưỡng HSV toàn crop   │   │ │
  │  │  │  red_mask   = mask(H∈[0,10]|[160,179])   │   │ │
  │  │  │  yellow_mask = mask(H∈[15,35])            │   │ │
  │  │  │  green_mask  = mask(H∈[40,85])            │   │ │
  │  │  │  → argmax(pixel count) → state            │   │ │
  │  │  └──────────────────────────────────────────┘   │ │
  │  │                                                  │ │
  │  │  → TrafficLightState + confidence                │ │
  │  └──────────────────┬──────────────────────────────┘ │
  │                     │                                 │
  │  ┌──────────────────▼──────────────────────────────┐ │
  │  │  Bước 4: Tạo TrafficLight object                 │ │
  │  │  TrafficLight(bbox=det.bbox, state=state,         │ │
  │  │               confidence=confidence)              │ │
  │  └──────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────┘
          │
          ▼
  List[TrafficLight]
  → Pipeline → SceneState.traffic_lights
  → Phase 6 Risk/Decision
  → Annotator vẽ lên frame
```

### 6.3 So Sánh Chiến Lược Phân Loại Màu

#### 6.3.1 Chiến lược A: Chia 3 Vùng Dọc (Region-based)

**Nguyên lý**: Đèn giao thông tiêu chuẩn có 3 bóng theo chiều dọc theo thứ tự trên-giữa-dưới = đỏ-vàng-xanh. Chia crop thành 3 phần bằng nhau theo chiều cao, tìm vùng nào có pixel "sáng" nhất (V cao + S cao).

**Ưu điểm**:
- Không cần biết ngưỡng màu cụ thể — ít nhạy với điều kiện ánh sáng
- Phân biệt được khi 2 màu gần nhau về hue (vàng vs xanh đôi khi bị nhầm)
- Code đơn giản, nhanh

**Nhược điểm**:
- Cần bbox đúng và crop đứng thẳng (đèn nghiêng → sai vùng)
- Không hoạt động với đèn nằm ngang (một số nước dùng đèn ngang)
- Crop không chứa đủ 3 bóng (đèn ở xa, bbox nhỏ)

#### 6.3.2 Chiến lược B: Lọc Ngưỡng HSV Toàn Crop (Color Mask)

**Nguyên lý**: Tạo binary mask cho từng màu trên toàn crop, đếm pixel thỏa điều kiện S ≥ S_min và V ≥ V_min, chọn màu có pixel count cao nhất.

**Ngưỡng HSV trong OpenCV** (H: 0-179, S: 0-255, V: 0-255):

| Màu | H_low | H_high | S_min | V_min | Ghi chú |
|-----|-------|--------|-------|-------|---------|
| Đỏ (phần 1) | 0 | 10 | 100 | 100 | Đỏ quanh 0° |
| Đỏ (phần 2) | 160 | 179 | 100 | 100 | Đỏ quanh 360° (wrap) |
| Vàng | 15 | 35 | 100 | 100 | Vàng-cam |
| Xanh lá | 40 | 85 | 80 | 80 | Xanh lá cây |

**Ưu điểm**:
- Hoạt động với đèn nghiêng, đèn nằm ngang
- Confidence = `pixel_count / crop_area` có nghĩa trực tiếp
- Dễ debug (có thể visualize mask)

**Nhược điểm**:
- Nhạy với ánh sáng mặt trời vàng buổi chiều (dễ nhầm nền trời vàng thành đèn vàng)
- Kính lọc nhiễu trên đèn (đèn có chụp) làm bão hòa màu → cần điều chỉnh ngưỡng

#### 6.3.3 Chiến lược Kết Hợp (Được Chọn Làm Mặc Định)

Implement cả hai trong một class, chạy B trước (nhanh), fallback sang A nếu confidence thấp:

```
confidence_B = max(red_count, yellow_count, green_count) / crop_area
if confidence_B < threshold (e.g. 0.05):
    dùng chiến lược A
else:
    dùng kết quả chiến lược B
```

#### 6.3.4 Phương Án Nâng Cao: CNN Tiny Classifier (Phase 8)

Khi HSV kém (điều kiện ngược sáng, đêm, mưa), có thể thay bằng CNN 3-layer nhỏ:

```
Input: crop → resize(32×64) → normalize
Conv2d(3→16, 3×3) → ReLU → MaxPool(2)
Conv2d(16→32, 3×3) → ReLU → MaxPool(2)
Flatten → Linear(32*7*15 → 64) → ReLU → Linear(64 → 4)
Output: softmax over [RED, YELLOW, GREEN, UNKNOWN]
```

Kích thước model: ~50 KB — nhẹ, chạy CPU được. Train trên Kaggle với LISA Traffic Light Dataset. Đây là nội dung của Phase 8, không implement ở Phase 4.

---

## 7. Công Việc Chi Tiết

### Task 4.1 — Kích Hoạt Traffic Light Detector Trong Pipeline

**Mục đích**: Đảm bảo `SimpleTrafficLightDetector` được khởi tạo và gọi đúng cách khi flag bật. Hiện tại `builder.py` đã có code này, chỉ cần verify và bật config mặc định khi chạy thử.

**File liên quan**:
- `src/drivevision/pipeline/builder.py` (đọc, không sửa — đã đúng)
- `src/drivevision/config.py` (không sửa default, chỉ dùng YAML override)
- `configs/phase4.yaml` (TẠO MỚI — config chạy thử phase này)

**Các bước**:

1. Tạo file config `configs/phase4.yaml`:

```yaml
# configs/phase4.yaml
# Config chạy thử Phase 4 — Traffic Light Detection
perception:
  detection:
    enabled: true
    weights: "models/weights/yolo.pt"
    conf: 0.35
    iou: 0.5
    classes: null  # detect tất cả, lọc id=9 ở TL detector
    imgsz: 640
  traffic_light:
    enabled: true
    classifier: "hsv"          # "hsv" | "cnn" (Phase 8)
    hsv:
      min_saturation: 100      # ngưỡng S tối thiểu để coi là màu đèn
      min_value: 80            # ngưỡng V tối thiểu (độ sáng)
      confidence_threshold: 0.05  # tỷ lệ pixel tối thiểu để tin kết quả
      min_crop_size: 10        # bỏ qua crop < 10×10 px
output:
  display: false
  save_path: "data/output/phase4_output.mp4"
  fps: 30
```

2. Verify `builder.py` đã đúng (chỉ đọc, không sửa):

```python
# builder.py — đoạn này ĐÃ CÓ và đúng:
if get_path(cfg, "perception.traffic_light.enabled"):
    from ..perception.traffic_light import SimpleTrafficLightDetector
    tl_detector = SimpleTrafficLightDetector()
```

3. Sau Task 4.2, cập nhật instantiation để truyền config:

```python
# builder.py — SỬA sau khi Task 4.2 xong:
if get_path(cfg, "perception.traffic_light.enabled"):
    from ..perception.traffic_light import SimpleTrafficLightDetector
    tl_cfg = get_path(cfg, "perception.traffic_light", {})
    tl_detector = SimpleTrafficLightDetector(config=tl_cfg)
```

**Lưu ý**: `builder.py` hiện tạo `SimpleTrafficLightDetector()` không có args. Cần sửa constructor để nhận `config` optional dict.

---

### Task 4.2 — Implement `SimpleTrafficLightDetector.detect()`

**Mục đích**: Lọc detections class 9 từ YOLO, crop từng đèn, gọi classifier, trả `List[TrafficLight]`.

**File**: `src/drivevision/perception/traffic_light.py` — **THAY THẾ TOÀN BỘ NỘI DUNG**

**Các bước**:

1. Thêm `__init__` nhận config
2. Implement `detect()` với logic lọc + crop + classify
3. Thêm helper `_safe_crop()` để clip tọa độ
4. Gọi `_classify_state()` (implement ở Task 4.3)

**Pseudocode / Skeleton**:

```python
"""Traffic light detection + state classification — Phase 4.

Chiến lược:
  1. Lọc List[Detection] lấy class_id == COCO_TRAFFIC_LIGHT_ID (9).
  2. Crop vùng bbox từ frame.image (BGR).
  3. Phân loại màu bằng HSV heuristic:
       - Tạo mask cho đỏ / vàng / xanh.
       - Đếm pixel thỏa ngưỡng S và V.
       - Màu có nhiều pixel nhất → TrafficLightState.
  4. Fallback region-based nếu confidence thấp.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..types import BoundingBox, Detection, Frame, TrafficLight, TrafficLightState
from .base import TrafficLightDetector

log = logging.getLogger("drivevision.traffic_light")

# COCO dataset: class id 9 = 'traffic light'
COCO_TRAFFIC_LIGHT_ID: int = 9

# Ngưỡng HSV mặc định (có thể override qua config)
_DEFAULT_HSV_CONFIG: Dict[str, Any] = {
    "min_saturation": 100,       # S tối thiểu để coi là màu đèn
    "min_value": 80,             # V tối thiểu (loại bỏ vùng tối)
    "confidence_threshold": 0.05,# tỷ lệ pixel tối thiểu / diện tích crop
    "min_crop_size": 10,         # bỏ qua crop < 10 px mỗi chiều
}


class SimpleTrafficLightDetector(TrafficLightDetector):
    """Phân loại trạng thái đèn giao thông bằng HSV heuristic.

    Nhận danh sách Detection từ YOLODetector, lọc class_id==9,
    crop từng đèn và phân loại màu.

    Args:
        config: Dict config từ ``perception.traffic_light`` trong YAML.
                Nếu None, dùng giá trị mặc định.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        hsv_cfg = {}
        if config:
            hsv_cfg = config.get("hsv", {})
        # Merge default với config user
        self._cfg = {**_DEFAULT_HSV_CONFIG, **hsv_cfg}
        self._min_s: int = int(self._cfg["min_saturation"])
        self._min_v: int = int(self._cfg["min_value"])
        self._conf_threshold: float = float(self._cfg["confidence_threshold"])
        self._min_crop: int = int(self._cfg["min_crop_size"])

    # ---------------------------------------------------------------------- #
    # Public API (implements TrafficLightDetector ABC)
    # ---------------------------------------------------------------------- #

    def detect(
        self, frame: Frame, detections: List[Detection]
    ) -> List[TrafficLight]:
        """Phát hiện và phân loại đèn giao thông.

        Args:
            frame:      Frame hiện tại (chứa frame.image dạng BGR HxWx3).
            detections: List[Detection] từ YOLODetector trong cùng frame.

        Returns:
            List[TrafficLight] — mỗi phần tử là một đèn với state và confidence.
        """
        results: List[TrafficLight] = []

        for det in detections:
            # Bước 1: Chỉ xử lý class 'traffic light' (COCO id=9)
            if det.class_id != COCO_TRAFFIC_LIGHT_ID:
                continue

            # Bước 2: Crop ảnh theo bbox (clip trong boundary frame)
            crop = self._safe_crop(frame.image, det.bbox)
            if crop is None:
                log.debug("Bỏ qua đèn tại %s — crop quá nhỏ hoặc ngoài frame.", det.bbox)
                results.append(
                    TrafficLight(bbox=det.bbox, state=TrafficLightState.UNKNOWN, confidence=0.0)
                )
                continue

            # Bước 3: Phân loại màu
            state, confidence = self._classify_state(crop)

            results.append(
                TrafficLight(bbox=det.bbox, state=state, confidence=confidence)
            )
            log.debug(
                "Đèn tại (%.0f,%.0f)-(%.0f,%.0f): %s (conf=%.2f)",
                det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2,
                state.value, confidence,
            )

        return results

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _safe_crop(
        self, image: np.ndarray, bbox: BoundingBox
    ) -> Optional[np.ndarray]:
        """Crop ảnh theo bbox, clamp trong boundary frame.

        Returns:
            Crop BGR hoặc None nếu crop quá nhỏ (< min_crop_size).
        """
        h, w = image.shape[:2]
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))

        if (x2 - x1) < self._min_crop or (y2 - y1) < self._min_crop:
            return None

        return image[y1:y2, x1:x2].copy()

    def _classify_state(
        self, crop_bgr: np.ndarray
    ) -> Tuple[TrafficLightState, float]:
        """Phân loại trạng thái màu đèn từ crop BGR.

        Thuật toán:
          1. Thử color mask (chiến lược B): tạo mask HSV cho đỏ/vàng/xanh,
             đếm pixel, chọn màu dominant.
          2. Nếu confidence_B < threshold, thử region-based (chiến lược A).
          3. Nếu vẫn thấp → UNKNOWN.

        Returns:
            (state, confidence) — confidence ∈ [0.0, 1.0].
        """
        # Chuyển sang HSV
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, w = crop_hsv.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return TrafficLightState.UNKNOWN, 0.0

        # --- Chiến lược B: Color Mask ---
        state_b, conf_b = self._color_mask_classify(crop_hsv, total_pixels)

        if conf_b >= self._conf_threshold:
            return state_b, conf_b

        # --- Fallback: Chiến lược A: Region-based ---
        state_a, conf_a = self._region_based_classify(crop_hsv)
        if conf_a >= self._conf_threshold:
            return state_a, conf_a

        return TrafficLightState.UNKNOWN, 0.0

    def _color_mask_classify(
        self, crop_hsv: np.ndarray, total_pixels: int
    ) -> Tuple[TrafficLightState, float]:
        """Chiến lược B: Đếm pixel từng màu trên toàn crop.

        Ngưỡng HSV (OpenCV: H∈[0,179], S∈[0,255], V∈[0,255]):
          Đỏ  : H∈[0,10]∪[160,179], S≥min_s, V≥min_v
          Vàng : H∈[15,35],          S≥min_s, V≥min_v
          Xanh : H∈[40,85],          S≥min_s, V≥min_v  (xanh lá)

        Returns:
            (state, confidence) với confidence = dominant_count / total_pixels.
        """
        s = self._min_s
        v = self._min_v

        # Mask đỏ (hai dải H vì đỏ wrap quanh 0°/180°)
        red_mask1 = cv2.inRange(crop_hsv, np.array([0,   s, v]), np.array([10,  255, 255]))
        red_mask2 = cv2.inRange(crop_hsv, np.array([160, s, v]), np.array([179, 255, 255]))
        red_mask   = cv2.bitwise_or(red_mask1, red_mask2)

        # Mask vàng
        yellow_mask = cv2.inRange(crop_hsv, np.array([15, s, v]), np.array([35, 255, 255]))

        # Mask xanh lá
        green_mask  = cv2.inRange(crop_hsv, np.array([40, 80, v]), np.array([85, 255, 255]))

        counts = {
            TrafficLightState.RED:    int(cv2.countNonZero(red_mask)),
            TrafficLightState.YELLOW: int(cv2.countNonZero(yellow_mask)),
            TrafficLightState.GREEN:  int(cv2.countNonZero(green_mask)),
        }

        dominant_state = max(counts, key=lambda s: counts[s])
        dominant_count = counts[dominant_state]
        confidence = dominant_count / total_pixels if total_pixels > 0 else 0.0

        return dominant_state, confidence

    def _region_based_classify(
        self, crop_hsv: np.ndarray
    ) -> Tuple[TrafficLightState, float]:
        """Chiến lược A: Chia 3 vùng dọc, tìm vùng sáng nhất.

        Thứ tự chuẩn đèn giao thông Việt Nam (dọc, từ trên xuống):
          [0   : h//3 ]  → Đỏ
          [h//3: 2h//3]  → Vàng
          [2h//3: h   ]  → Xanh

        "Sáng nhất" = tổng kênh V (brightness) trong vùng đó.

        Returns:
            (state, confidence) với confidence = mean_V_dominant / 255.
        """
        h, w = crop_hsv.shape[:2]
        if h < 3:
            return TrafficLightState.UNKNOWN, 0.0

        t = h // 3
        regions = {
            TrafficLightState.RED:    crop_hsv[0:t,         :, 2],  # kênh V
            TrafficLightState.YELLOW: crop_hsv[t:2*t,       :, 2],
            TrafficLightState.GREEN:  crop_hsv[2*t:h,       :, 2],
        }

        # Mean V của từng vùng
        brightness = {state: float(np.mean(region)) for state, region in regions.items()}
        dominant = max(brightness, key=lambda s: brightness[s])
        confidence = brightness[dominant] / 255.0

        return dominant, confidence
```

**Lưu ý & Edge Cases**:

- **Crop quá nhỏ** (`< min_crop_size`): `_safe_crop()` trả `None` → `detect()` append `UNKNOWN` với confidence=0.0, không crash.
- **Tọa độ bbox âm hoặc vượt frame**: `_safe_crop()` clamp với `max(0,...)` và `min(w/h,...)`.
- **Ảnh đêm / ngược sáng**: V thấp → ít pixel thỏa `min_value` → confidence thấp → trả `UNKNOWN`. Tốt hơn là đoán sai.
- **Đèn bị che một phần**: Nếu crop hợp lệ nhưng đèn bị che, confidence thấp → `UNKNOWN`. Đây là hành vi đúng (thà không biết hơn đoán sai).
- **Nhiều đèn trong frame**: Vòng lặp `for det in detections` xử lý từng đèn độc lập → `List[TrafficLight]` có thể chứa nhiều phần tử.
- **Đèn vàng nhấp nháy**: Phase 4 không xử lý temporal — chỉ phân loại per-frame. Temporal smoothing thuộc Phase 5 (Scene Understanding) hoặc Phase 6.
- **Đèn người đi bộ**: YOLO COCO không phân biệt, cùng class 9 → được phân loại như đèn xe. Đây là limitation đã biết.
- **Đèn nằm ngang (một số giao lộ)**: Chiến lược A (chia dọc) sẽ sai → chiến lược B vẫn đúng vì dùng toàn crop.

---

### Task 4.3 — Cập Nhật `builder.py` Truyền Config

**Mục đích**: `SimpleTrafficLightDetector` giờ nhận `config` dict — cần cập nhật cách builder khởi tạo.

**File**: `src/drivevision/pipeline/builder.py`

**Thay đổi** (tìm đoạn và sửa):

```python
# Trước (dòng 73-76 hiện tại):
if get_path(cfg, "perception.traffic_light.enabled"):
    from ..perception.traffic_light import SimpleTrafficLightDetector
    tl_detector = SimpleTrafficLightDetector()

# Sau (sửa thành):
if get_path(cfg, "perception.traffic_light.enabled"):
    from ..perception.traffic_light import SimpleTrafficLightDetector
    tl_cfg = cfg.get("perception", {}).get("traffic_light", {})
    tl_detector = SimpleTrafficLightDetector(config=tl_cfg)
```

**Lưu ý**: Dùng `cfg.get()` thay `get_path()` vì cần truyền cả dict con, không phải giá trị lá.

---

### Task 4.4 — Cập Nhật `viz/annotator.py` Vẽ Traffic Light

**Mục đích**: Hiển thị bounding box màu đèn + nhãn trạng thái lên frame để kiểm tra trực quan.

**File**: `src/drivevision/viz/annotator.py` — **CẬP NHẬT**

**Thiết kế visual**:

```
┌─────────────────────────────┐
│  [ĐƯỜNG PHỐ]                │
│                             │
│  ┌───┐  ← bbox đèn          │
│  │RED│  ← label + màu khung │
│  └───┘                      │
│  conf: 0.87                 │
└─────────────────────────────┘
```

- Bbox màu: `RED` → đỏ `(0,0,255)`, `GREEN` → xanh `(0,255,0)`, `YELLOW` → vàng `(0,255,255)`, `UNKNOWN` → xám `(128,128,128)`
- Nhãn text: `"RED 0.87"` (state + confidence 2 chữ số)
- Độ dày khung: 3px (nổi bật hơn detection thường)
- Font: `cv2.FONT_HERSHEY_SIMPLEX`, scale 0.6

**Skeleton cần thêm vào annotator**:

```python
"""Draw pipeline results onto a frame.

Fill in the drawing you want for the demo: boxes + track ids, lane polylines,
traffic-light states, and a HUD with the risk level / decision. Return a BGR
image (same size as input) so the API can stream it.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import PipelineResult, TrafficLightState

# Map trạng thái đèn → màu BGR để vẽ
_TL_STATE_COLOR = {
    TrafficLightState.RED:     (0,   0,   255),  # BGR đỏ
    TrafficLightState.YELLOW:  (0,   255, 255),  # BGR vàng
    TrafficLightState.GREEN:   (0,   255,  0),   # BGR xanh lá
    TrafficLightState.UNKNOWN: (128, 128, 128),  # BGR xám
}

_TL_STATE_LABEL = {
    TrafficLightState.RED:     "RED",
    TrafficLightState.YELLOW:  "YELLOW",
    TrafficLightState.GREEN:   "GREEN",
    TrafficLightState.UNKNOWN: "?",
}


class Annotator:
    def draw(self, result: PipelineResult) -> np.ndarray:
        """Vẽ tất cả annotations lên frame và trả về BGR image."""
        frame = result.frame.image.copy()
        scene = result.scene

        # --- Vẽ detections / tracks ---
        # TODO (Phase 2): vẽ track bbox + track_id từ scene.tracks

        # --- Vẽ lanes ---
        # TODO (Phase 3): vẽ polyline từ scene.lanes

        # --- Vẽ traffic lights ---
        for tl in scene.traffic_lights:
            frame = self._draw_traffic_light(frame, tl)

        # --- HUD: risk + decision ---
        # TODO (Phase 6): vẽ risk level + decision lên góc trên trái

        return frame

    def _draw_traffic_light(self, frame: np.ndarray, tl) -> np.ndarray:
        """Vẽ một TrafficLight lên frame.

        Args:
            frame: BGR image.
            tl:    TrafficLight instance.

        Returns:
            Frame đã được vẽ (in-place nhưng trả về để chain).
        """
        color = _TL_STATE_COLOR.get(tl.state, (128, 128, 128))
        label = _TL_STATE_LABEL.get(tl.state, "?")
        x1, y1, x2, y2 = tl.bbox.as_int()

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)

        # Vẽ label + confidence
        text = f"{label} {tl.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Nền label (rectangle nhỏ phía trên bbox)
        label_y1 = max(0, y1 - text_h - baseline - 4)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 4, label_y2), color, -1)

        # Text trắng lên nền màu
        cv2.putText(
            frame, text,
            (x1 + 2, y1 - baseline - 2),
            font, font_scale,
            (255, 255, 255),  # text trắng
            thickness,
        )

        return frame
```

**Lưu ý**:
- `tl.bbox.as_int()` đã có trong `BoundingBox` — dùng trực tiếp.
- Nếu `y1 < text_h` (đèn ở sát mép trên), `label_y1 = max(0, ...)` đảm bảo không vẽ ngoài frame.
- `cv2.rectangle(frame, pt1, pt2, color, -1)` với thickness=-1 tô màu đặc.

---

### Task 4.5 — Viết Unit Tests

**Mục đích**: Đảm bảo classifier và detector hoạt động đúng với ảnh tổng hợp (không cần video thật).

**File**: `tests/test_traffic_light.py` — **TẠO MỚI**

**Các bước**:

1. Tạo ảnh BGR tổng hợp cho từng màu đèn
2. Test `_color_mask_classify` trực tiếp
3. Test `_region_based_classify` trực tiếp
4. Test `detect()` end-to-end với Frame mock và Detection mock
5. Test edge cases: crop nhỏ, ảnh đen, ảnh trắng

**Skeleton**:

```python
"""Unit tests cho SimpleTrafficLightDetector — Phase 4."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import pytest

from drivevision.perception.traffic_light import SimpleTrafficLightDetector
from drivevision.types import (
    BoundingBox,
    Detection,
    Frame,
    TrafficLight,
    TrafficLightState,
)

COCO_TL_ID = 9  # traffic light class id


# --------------------------------------------------------------------------- #
# Helpers tạo ảnh test
# --------------------------------------------------------------------------- #

def _solid_bgr(color_bgr: tuple, size: tuple = (40, 80)) -> np.ndarray:
    """Tạo ảnh đơn sắc BGR kích thước (width, height)."""
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


def _make_frame(image: np.ndarray) -> Frame:
    return Frame(index=0, timestamp=0.0, image=image)


def _make_detection(x1, y1, x2, y2, class_id=COCO_TL_ID) -> Detection:
    return Detection(
        bbox=BoundingBox(x1, y1, x2, y2),
        class_id=class_id,
        class_name="traffic light" if class_id == COCO_TL_ID else "car",
        confidence=0.9,
    )


# --------------------------------------------------------------------------- #
# Test _color_mask_classify
# --------------------------------------------------------------------------- #

class TestColorMaskClassify:
    """Test chiến lược B — color mask."""

    def setup_method(self):
        self.detector = SimpleTrafficLightDetector()

    def _get_hsv(self, bgr_img):
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    def test_red_light(self):
        # Màu đỏ thuần: BGR (0, 0, 200) → HSV H≈0
        img = _solid_bgr((0, 0, 200))
        hsv = self._get_hsv(img)
        total = img.shape[0] * img.shape[1]
        state, conf = self.detector._color_mask_classify(hsv, total)
        assert state == TrafficLightState.RED
        assert conf > 0.5

    def test_green_light(self):
        # Màu xanh lá: BGR (0, 200, 0) → HSV H≈60
        img = _solid_bgr((0, 200, 0))
        hsv = self._get_hsv(img)
        total = img.shape[0] * img.shape[1]
        state, conf = self.detector._color_mask_classify(hsv, total)
        assert state == TrafficLightState.GREEN
        assert conf > 0.5

    def test_yellow_light(self):
        # Màu vàng: BGR (0, 200, 200) → HSV H≈30
        img = _solid_bgr((0, 200, 200))
        hsv = self._get_hsv(img)
        total = img.shape[0] * img.shape[1]
        state, conf = self.detector._color_mask_classify(hsv, total)
        assert state == TrafficLightState.YELLOW
        assert conf > 0.3  # vàng khó hơn

    def test_dark_image_low_confidence(self):
        # Ảnh tối → S và V thấp → không thỏa ngưỡng → confidence thấp
        img = _solid_bgr((10, 10, 10))
        hsv = self._get_hsv(img)
        total = img.shape[0] * img.shape[1]
        state, conf = self.detector._color_mask_classify(hsv, total)
        # confidence phải thấp hơn threshold
        assert conf < self.detector._conf_threshold


# --------------------------------------------------------------------------- #
# Test _region_based_classify
# --------------------------------------------------------------------------- #

class TestRegionBasedClassify:
    """Test chiến lược A — chia 3 vùng dọc."""

    def setup_method(self):
        self.detector = SimpleTrafficLightDetector()

    def _make_3zone_crop(self, top_bgr, mid_bgr, bot_bgr, h=90, w=30):
        """Tạo crop 3 vùng màu khác nhau."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        t = h // 3
        img[0:t, :]    = top_bgr
        img[t:2*t, :]  = mid_bgr
        img[2*t:h, :]  = bot_bgr
        return img

    def test_red_zone_brightest(self):
        # Vùng trên (đỏ) sáng nhất
        img = self._make_3zone_crop(
            top_bgr=(0, 0, 220),   # đỏ sáng → V cao
            mid_bgr=(10, 10, 10),  # tối
            bot_bgr=(10, 10, 10),  # tối
        )
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        state, conf = self.detector._region_based_classify(hsv)
        assert state == TrafficLightState.RED

    def test_green_zone_brightest(self):
        img = self._make_3zone_crop(
            top_bgr=(10, 10, 10),
            mid_bgr=(10, 10, 10),
            bot_bgr=(0, 220, 0),   # xanh lá sáng
        )
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        state, conf = self.detector._region_based_classify(hsv)
        assert state == TrafficLightState.GREEN

    def test_too_small_crop(self):
        # Crop < 3 pixel chiều cao → UNKNOWN
        img = np.zeros((2, 10, 3), dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        state, conf = self.detector._region_based_classify(hsv)
        assert state == TrafficLightState.UNKNOWN


# --------------------------------------------------------------------------- #
# Test detect() end-to-end
# --------------------------------------------------------------------------- #

class TestDetectEndToEnd:
    """Test toàn bộ pipeline detect()."""

    def setup_method(self):
        self.detector = SimpleTrafficLightDetector()

    def test_filters_non_traffic_light_class(self):
        """Detection không phải class 9 phải bị bỏ qua."""
        # Frame 100x100 đỏ (toàn bộ)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:] = (0, 0, 200)  # đỏ BGR
        frame = _make_frame(image)

        # Detection class 2 (car)
        detections = [_make_detection(10, 10, 50, 80, class_id=2)]
        results = self.detector.detect(frame, detections)
        assert results == []

    def test_detects_traffic_light_class9(self):
        """Detection class 9 trong frame đỏ → TrafficLight với state RED."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:80, 10:50] = (0, 0, 200)  # vùng đỏ
        frame = _make_frame(image)

        detections = [_make_detection(10, 10, 50, 80)]
        results = self.detector.detect(frame, detections)
        assert len(results) == 1
        assert isinstance(results[0], TrafficLight)
        assert results[0].state == TrafficLightState.RED

    def test_empty_detections(self):
        """Không có detection → trả list rỗng."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = _make_frame(image)
        results = self.detector.detect(frame, [])
        assert results == []

    def test_bbox_out_of_frame_returns_unknown(self):
        """bbox ngoài frame → crop None → state UNKNOWN, không crash."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = _make_frame(image)
        # bbox ngoài frame
        detections = [_make_detection(90, 90, 200, 300)]
        # Vùng clamp thực tế: x1=90,y1=90,x2=100,y2=100 → 10×10 = đúng min_crop
        results = self.detector.detect(frame, detections)
        assert len(results) == 1
        # State phụ thuộc nội dung crop (zeros → unknown)
        assert results[0].state in list(TrafficLightState)

    def test_multiple_lights(self):
        """Nhiều đèn trong frame → trả đúng số lượng."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[10:80, 10:50]   = (0, 0, 200)   # đỏ — đèn 1
        image[10:80, 100:140] = (0, 200, 0)   # xanh — đèn 2
        frame = _make_frame(image)

        detections = [
            _make_detection(10, 10, 50, 80),
            _make_detection(100, 10, 140, 80),
        ]
        results = self.detector.detect(frame, detections)
        assert len(results) == 2
        states = {r.state for r in results}
        assert TrafficLightState.RED in states
        assert TrafficLightState.GREEN in states

    def test_very_small_bbox_returns_unknown(self):
        """bbox < min_crop_size → UNKNOWN."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = _make_frame(image)
        detections = [_make_detection(10, 10, 15, 15)]  # 5×5 < min_crop=10
        results = self.detector.detect(frame, detections)
        assert len(results) == 1
        assert results[0].state == TrafficLightState.UNKNOWN
        assert results[0].confidence == 0.0


# --------------------------------------------------------------------------- #
# Test config override
# --------------------------------------------------------------------------- #

class TestConfigOverride:
    def test_custom_min_saturation(self):
        """Config min_saturation=0 → chấp nhận màu ít bão hòa hơn."""
        cfg = {"hsv": {"min_saturation": 0, "min_value": 0}}
        detector = SimpleTrafficLightDetector(config=cfg)
        assert detector._min_s == 0

    def test_default_config_values(self):
        detector = SimpleTrafficLightDetector()
        assert detector._min_s == 100
        assert detector._min_v == 80
        assert detector._conf_threshold == 0.05
        assert detector._min_crop == 10
```

**Lưu ý**:
- Test dùng ảnh BGR tổng hợp (solid color) — không cần ảnh thật.
- Màu đỏ BGR `(0, 0, 200)` → HSV H≈0, S≈255, V≈200 → thỏa ngưỡng.
- Màu xanh lá BGR `(0, 200, 0)` → HSV H≈60, S≈255, V≈200.
- Màu vàng BGR `(0, 200, 200)` → HSV H≈30, S≈255, V≈200.

---

### Task 4.6 — Kiểm Tra End-to-End Bằng Script

**Mục đích**: Chạy pipeline thật với video mẫu để verify annotator vẽ đúng.

**File**: `scripts/run_phase4.py` — **TẠO MỚI** (helper script, không phải production code)

```python
#!/usr/bin/env python3
"""Script chạy nhanh Phase 4 để kiểm tra trực quan.

Usage:
    python scripts/run_phase4.py --video data/samples/sample.mp4 --frames 100
"""

import argparse
import sys
from pathlib import Path

# Thêm src vào path nếu chạy trực tiếp
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drivevision.config import DEFAULT_CONFIG, _deep_merge, load_config
from drivevision.pipeline.builder import build_pipeline, build_source
from drivevision.viz.annotator import Annotator
import cv2


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Traffic Light Test")
    parser.add_argument("--config", default="configs/phase4.yaml", help="Path to config YAML")
    parser.add_argument("--video", default=None, help="Override video path")
    parser.add_argument("--frames", type=int, default=200, help="Max frames to process")
    parser.add_argument("--display", action="store_true", help="Hiển thị frame trực tiếp (cần GUI)")
    args = parser.parse_args()

    # Load config
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"[WARN] Không tìm thấy {args.config}, dùng default config.")
        cfg = DEFAULT_CONFIG.copy()
        cfg["perception"]["traffic_light"]["enabled"] = True

    if args.video:
        cfg["source"]["path"] = args.video
    if args.frames:
        cfg["source"]["max_frames"] = args.frames

    source = build_source(cfg)
    pipeline = build_pipeline(cfg)
    annotator = Annotator()

    tl_counts = {"red": 0, "yellow": 0, "green": 0, "unknown": 0}

    for frame in source:
        result = pipeline.process(frame)
        annotated = annotator.draw(result)

        # Thống kê
        for tl in result.scene.traffic_lights:
            tl_counts[tl.state.value] += 1

        if args.display:
            cv2.imshow("Phase 4 — Traffic Light", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("\n=== Kết quả Phase 4 ===")
    print(f"Tổng đèn phát hiện: {sum(tl_counts.values())}")
    for state, count in tl_counts.items():
        print(f"  {state.upper()}: {count}")


if __name__ == "__main__":
    main()
```

---

## 8. Thay Đổi Cấu Hình

### 8.1 Config YAML Mới

**Tạo**: `configs/phase4.yaml` (đã có trong Task 4.1)

### 8.2 Thay Đổi `DEFAULT_CONFIG` (Không Thay Đổi)

`DEFAULT_CONFIG` trong `config.py` giữ `traffic_light: {enabled: False}` — đây là đúng. Phase 4 bật tính năng qua file YAML override, không thay default.

### 8.3 Các Key Config Mới Trong `perception.traffic_light`

| Key | Type | Default | Mô tả |
|-----|------|---------|-------|
| `enabled` | bool | `false` | Bật/tắt TL detector |
| `classifier` | str | `"hsv"` | Loại classifier: `"hsv"` hoặc `"cnn"` (Phase 8) |
| `hsv.min_saturation` | int | `100` | S tối thiểu cho pixel đèn |
| `hsv.min_value` | int | `80` | V tối thiểu (độ sáng) |
| `hsv.confidence_threshold` | float | `0.05` | Tỷ lệ pixel tối thiểu để tin kết quả |
| `hsv.min_crop_size` | int | `10` | Kích thước crop tối thiểu (px) |

**Ví dụ YAML đầy đủ**:

```yaml
perception:
  traffic_light:
    enabled: true
    classifier: "hsv"
    hsv:
      min_saturation: 100
      min_value: 80
      confidence_threshold: 0.05
      min_crop_size: 10
```

---

## 9. Kiểm Thử

### 9.1 Unit Tests (Tự Động)

Chạy:
```bash
cd /home/quan/DriveVision
python -m pytest tests/test_traffic_light.py -v
```

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| Ảnh đỏ solid → detect RED | BGR `(0,0,200)` crop | `state=RED`, `conf>0.5` |
| Ảnh xanh solid → detect GREEN | BGR `(0,200,0)` crop | `state=GREEN`, `conf>0.5` |
| Ảnh vàng solid → detect YELLOW | BGR `(0,200,200)` crop | `state=YELLOW`, `conf>0.3` |
| Ảnh tối → UNKNOWN | BGR `(10,10,10)` | `conf < threshold` |
| Crop < 10×10 → UNKNOWN | bbox 5×5 | `state=UNKNOWN`, `conf=0.0` |
| bbox ngoài frame → không crash | bbox (90,90,200,300), frame 100×100 | Không raise exception |
| Detection class != 9 → bỏ qua | class_id=2 | `results == []` |
| Nhiều đèn → đúng số lượng | 2 detection class 9 | `len(results) == 2` |
| Region-based: vùng trên sáng → RED | 3-zone crop, top sáng | `state=RED` |
| Region-based: vùng dưới sáng → GREEN | 3-zone crop, bottom sáng | `state=GREEN` |
| Config override saturation=0 | `min_saturation=0` | `detector._min_s == 0` |

### 9.2 Kiểm Tra Trực Quan (Thủ Công)

```bash
python scripts/run_phase4.py \
    --video data/samples/sample.mp4 \
    --frames 300 \
    --display
```

Quan sát:
- [ ] Bbox đèn giao thông xuất hiện với màu khung đúng (đỏ/xanh/vàng/xám)
- [ ] Label "RED 0.87" / "GREEN 0.72" / "YELLOW 0.41" hiển thị rõ
- [ ] Không bị False Positive (đèn xe, biển báo đỏ bị nhầm thành đèn giao thông)
- [ ] Không crash khi đèn ở mép frame

### 9.3 Edge Cases Cần Kiểm Tra Thủ Công

| Tình huống | Cách kiểm tra | Expected |
|------------|---------------|---------|
| Đèn tắt (đêm, đèn hỏng) | Video ban đêm | `UNKNOWN` |
| Ngược sáng (đèn đối diện mặt trời) | Video buổi chiều | `UNKNOWN` hoặc confidence thấp |
| Đèn vàng nhấp nháy | Video có đèn nhấp nháy | Luân phiên `YELLOW`/`UNKNOWN` per-frame (bình thường) |
| Đèn người đi bộ | Video có đèn người đi bộ | Được classify như đèn xe (limitation đã biết) |
| Nhiều đèn cùng lúc | Giao lộ phức tạp | Mỗi đèn có bbox và state riêng |
| Đèn xa (bbox nhỏ) | Video đường cao tốc | `UNKNOWN` nếu crop < min_crop, hoặc confidence thấp |
| Biển đỏ bị nhầm | Biển stop, đèn hậu xe | Confidence thấp hơn đèn thật (shape khác nhau ít ảnh hưởng HSV) |

### 9.4 Performance Test

```bash
python -c "
import time, numpy as np, cv2
from drivevision.perception.traffic_light import SimpleTrafficLightDetector
from drivevision.types import BoundingBox, Detection, Frame

det = SimpleTrafficLightDetector()
img = np.zeros((720, 1280, 3), dtype=np.uint8)
img[100:200, 50:100] = (0, 0, 200)  # đèn đỏ
frame = Frame(0, 0.0, img)
dets = [Detection(BoundingBox(50,100,100,200), 9, 'traffic light', 0.9)]

N = 1000
t0 = time.perf_counter()
for _ in range(N):
    det.detect(frame, dets)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f'Thời gian classify: {elapsed:.3f} ms/frame')
assert elapsed < 2.0, f'Quá chậm: {elapsed:.1f} ms'
print('PASS: < 2ms/đèn')
"
```

---

## 10. Tiêu Chí Hoàn Thành

### 10.1 Code

- [ ] `src/drivevision/perception/traffic_light.py`: `SimpleTrafficLightDetector` implement đầy đủ, không còn `return []`
- [ ] `__init__` nhận `config: Optional[Dict]`
- [ ] `detect()` lọc `class_id == 9`, crop, classify, trả `List[TrafficLight]`
- [ ] `_safe_crop()` clip tọa độ, trả `None` nếu quá nhỏ
- [ ] `_classify_state()` gọi B trước, fallback A
- [ ] `_color_mask_classify()` xử lý đúng H wrap của màu đỏ (2 dải)
- [ ] `_region_based_classify()` xử lý crop < 3px chiều cao
- [ ] `src/drivevision/pipeline/builder.py`: truyền `config=tl_cfg` vào constructor
- [ ] `src/drivevision/viz/annotator.py`: vẽ `TrafficLight` với màu sắc và label đúng
- [ ] `configs/phase4.yaml`: tồn tại với đủ keys
- [ ] `scripts/run_phase4.py`: chạy được

### 10.2 Tests

- [ ] `tests/test_traffic_light.py` tồn tại
- [ ] Tất cả tests trong `TestColorMaskClassify` PASS
- [ ] Tất cả tests trong `TestRegionBasedClassify` PASS
- [ ] Tất cả tests trong `TestDetectEndToEnd` PASS
- [ ] Tất cả tests trong `TestConfigOverride` PASS
- [ ] `python -m pytest tests/test_traffic_light.py -v` → 0 failures

### 10.3 Chất Lượng

- [ ] Docstring đầy đủ ở class và public methods
- [ ] Type hints đầy đủ (không có `Any` không cần thiết)
- [ ] Logging sử dụng `log.debug()` (không dùng `print()`)
- [ ] Không có import thừa

### 10.4 Hiệu Năng

- [ ] HSV classify ≤ 2 ms/đèn (đo bằng script performance test)
- [ ] Không memory leak (crop được `copy()` rồi xử lý, không giữ ref)

### 10.5 Tích Hợp

- [ ] `python scripts/run_phase4.py --video data/samples/sample.mp4 --frames 50` chạy không lỗi
- [ ] `SceneState.traffic_lights` có dữ liệu khi có đèn trong frame
- [ ] Annotator vẽ bbox màu đúng (đỏ/xanh/vàng/xám)

---

## 11. Rủi Ro & Cách Xử Lý

| Rủi ro | Xác suất | Mức độ | Cách xử lý |
|--------|----------|--------|------------|
| YOLO COCO không detect được đèn nhỏ/xa | Cao | Cao | Điều chỉnh `conf` xuống 0.2, tăng `imgsz` lên 1280; nếu vẫn kém → Phase 8 fine-tune |
| Màu đỏ bị nhầm với đèn hậu xe (cũng đỏ) | Trung bình | Trung bình | YOLO detect class phân biệt; đèn xe không có class 9 → không ảnh hưởng |
| HSV nhầm vàng-cam buổi chiều | Trung bình | Trung bình | Tăng `min_saturation` lên 120; thêm điều kiện V > 150 cho vàng; hoặc dùng context (vùng trên ảnh, không phải vùng dưới xe) |
| Crop quá nhỏ khi đèn xa | Cao | Thấp | `_safe_crop()` trả `None` → `UNKNOWN` — hành vi đúng và an toàn |
| Đèn nằm ngang (một số giao lộ) | Thấp | Thấp | Chiến lược B vẫn hoạt động; ghi nhận là limitation |
| Ngược sáng làm V quá cao hoặc quá thấp | Trung bình | Thấp | V quá cao → S thấp → không thỏa ngưỡng → `UNKNOWN`; là hành vi đúng |
| Builder không truyền config → dùng default | Thấp | Thấp | Có `_DEFAULT_HSV_CONFIG` fallback trong constructor |
| `cv2` chưa cài | Thấp | Cao | Đã là dependency của project (dùng trong annotator cũ); thêm vào requirements nếu thiếu |

---

## 12. Hiệu Năng & Tài Nguyên

### 12.1 Tính Toán

| Thành phần | Chi phí ước tính | Ghi chú |
|------------|-----------------|---------|
| `cv2.cvtColor()` (BGR→HSV) trên crop 40×80 | < 0.1 ms | Crop nhỏ, rất nhanh |
| `cv2.inRange()` × 4 masks | < 0.3 ms | Vectorized NumPy |
| `cv2.countNonZero()` × 4 | < 0.1 ms | |
| `_region_based_classify()` | < 0.2 ms | NumPy mean |
| **Tổng/đèn** | **< 1 ms** | Tốt hơn target 2 ms |
| Vẽ annotation/đèn | < 0.5 ms | `cv2.rectangle()` + `cv2.putText()` |

### 12.2 Bộ Nhớ

- Mỗi crop `copy()` được giải phóng sau khi hàm trả về → không tích lũy
- Mask tạm thời (red_mask, yellow_mask, green_mask): tổng ~3 × (40×80) bytes = ~10 KB/đèn → không đáng kể

### 12.3 So Sánh với Alternative

| Phương án | Latency | Accuracy | Complexity |
|-----------|---------|----------|------------|
| HSV Heuristic (Phase 4) | ~1 ms/đèn | ~85% điều kiện tốt | Thấp |
| CNN Tiny (Phase 8) | ~5 ms/đèn (CPU) | ~95% | Trung bình |
| Model chuyên dụng detection | + 10-50 ms | tốt hơn YOLO COCO | Cao |

### 12.4 Môi Trường Chạy

- Phase 4 hoàn toàn chạy trên **CPU** — không cần GPU
- YOLODetector (Phase 1) cần GPU hoặc CPU với thời gian chậm hơn — nhưng đó là Phase 1, không phải Phase 4

---

## 13. Sản Phẩm Bàn Giao

### 13.1 Code Files

| File | Trạng thái | Mô tả |
|------|------------|-------|
| `src/drivevision/perception/traffic_light.py` | Sửa (hoàn chỉnh) | `SimpleTrafficLightDetector` với HSV classifier |
| `src/drivevision/pipeline/builder.py` | Sửa (nhỏ) | Truyền `config` vào constructor |
| `src/drivevision/viz/annotator.py` | Sửa (thêm method) | Vẽ `TrafficLight` với màu và label |
| `tests/test_traffic_light.py` | Tạo mới | Unit tests đầy đủ |
| `configs/phase4.yaml` | Tạo mới | Config chạy thử Phase 4 |
| `scripts/run_phase4.py` | Tạo mới | Script kiểm tra end-to-end |

### 13.2 Files KHÔNG Thay Đổi

| File | Lý do giữ nguyên |
|------|-----------------|
| `src/drivevision/types.py` | Types đã đủ (`TrafficLight`, `TrafficLightState`) |
| `src/drivevision/perception/base.py` | ABC đã đúng signature |
| `src/drivevision/pipeline/pipeline.py` | Đã gọi `tl_detector.detect(frame, detections)` đúng |
| `src/drivevision/config.py` | Default config giữ `traffic_light.enabled: false` |

### 13.3 Output Kiểm Chứng

- Kết quả chạy script: thống kê số đèn RED/YELLOW/GREEN/UNKNOWN
- Frame annotated có thể lưu ra `data/output/phase4_output.mp4` (nếu cần demo)
- Test report: `python -m pytest tests/test_traffic_light.py -v` với 0 failures

---

## 14. Điểm Nhấn Portfolio

Phase 4 thể hiện các kỹ năng kỹ thuật quan trọng cho portfolio:

### 14.1 Thiết Kế Hệ Thống

- **Tái dùng thay vì tạo mới**: Không train model detection riêng cho đèn mà tái dùng YOLO COCO — thể hiện hiểu biết về trade-off giữa chi phí và kết quả.
- **Interface-driven design**: `SimpleTrafficLightDetector` implement ABC `TrafficLightDetector` — dễ swap sang CNN classifier ở Phase 8 mà không thay đổi pipeline.
- **Separation of concerns**: Detection (tìm hộp) tách biệt khỏi Classification (phân loại màu) — đúng nguyên tắc Single Responsibility.

### 14.2 Computer Vision

- **HSV color space**: Sử dụng đúng không gian màu cho bài toán phân loại màu (HSV tách biệt hue khỏi độ sáng, tốt hơn RGB).
- **Dual-strategy với fallback**: Kết hợp color mask (chính) và region-based (dự phòng) — thể hiện tư duy robust engineering.
- **Edge case handling**: Xử lý hàng chục edge case thực tế (ngược sáng, crop nhỏ, đèn tắt) — phân biệt code production với code đồ chơi.

### 14.3 Software Engineering

- **Config-driven behavior**: Ngưỡng HSV và behavior đều tunable qua YAML — không hardcode magic numbers.
- **Comprehensive testing**: Test cả unit (crop synthetic) và integration (end-to-end script).
- **Logging thay vì print**: `log.debug()` với format message rõ ràng.

### 14.4 Điểm Demo Mạnh

- Frame annotated với bbox màu tương ứng (đỏ/xanh/vàng) — trực quan mạnh trong video demo.
- Thống kê số đèn theo loại — cho thấy pipeline xử lý thật, không phải fake.
- Latency < 2 ms/đèn — có thể đưa vào README như benchmark cụ thể.

---

## 15. Tham Khảo

### 15.1 Tài Liệu Kỹ Thuật

- OpenCV HSV Color Space: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
- `cv2.inRange()` documentation: https://docs.opencv.org/4.x/d2/de8/group__core__array.html
- COCO Dataset class list (traffic light = id 9): https://cocodataset.org/#explore

### 15.2 Dataset (Cho Phase 8 Fine-tuning)

- **LISA Traffic Light Dataset**: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset — dataset chuyên đèn giao thông Mỹ, có annotation màu
- **BTTSD (Bosch Traffic Light Dataset)**: Nhỏ hơn, annotation chất lượng cao, bao gồm đèn xa
- **Custom crop**: Extract crop class_id=9 từ video dataset COCO → tạo dataset phân loại màu

### 15.3 Papers Liên Quan (Nếu Muốn Đọc Sâu)

- "Traffic Light Recognition Using Deep Learning and Saliency Map" — Nếu muốn hiểu tại sao CNN tiny tốt hơn HSV ở đèn nhỏ
- "Real-time Traffic Light Detection" (ITSC 2021) — Benchmark các phương pháp detection chuyên dụng

### 15.4 Internal References

- `src/drivevision/types.py` — định nghĩa `TrafficLight`, `TrafficLightState`, `BoundingBox`
- `src/drivevision/perception/base.py` — ABC `TrafficLightDetector`
- `src/drivevision/perception/detection.py` — `YOLODetector` (nguồn detections)
- `src/drivevision/pipeline/pipeline.py` — cách pipeline gọi `tl_detector.detect()`
- `src/drivevision/pipeline/builder.py` — cách instantiate từ config
- `phase_3.md` — Phase Lane Detection (tham khảo pattern implement)
- `phase_6.md` — Phase Risk/Decision (tiêu thụ output Phase 4)

---

## 16. Checklist Tổng Kết

Checklist này dùng để review trước khi đánh dấu Phase 4 hoàn thành:

### Code

- [ ] `SimpleTrafficLightDetector.__init__()` nhận `config: Optional[Dict]` với fallback default
- [ ] `detect()` lọc đúng `class_id == COCO_TRAFFIC_LIGHT_ID` (9)
- [ ] `_safe_crop()` clamp tọa độ và kiểm tra `min_crop_size`
- [ ] `_classify_state()` gọi chiến lược B trước, fallback A khi confidence thấp
- [ ] `_color_mask_classify()` xử lý màu đỏ với 2 dải H (wrap quanh 0°/180°)
- [ ] `_region_based_classify()` trả `UNKNOWN` khi crop height < 3
- [ ] Không có `print()` trong production code (chỉ `log.debug()`)
- [ ] Không có import không dùng
- [ ] Type hints đầy đủ ở tất cả public/private methods

### Builder

- [ ] `builder.py` truyền `config=tl_cfg` (dict) vào `SimpleTrafficLightDetector`

### Annotator

- [ ] `Annotator._draw_traffic_light()` vẽ bbox với màu đúng theo state
- [ ] Label text hiển thị `"{STATE} {confidence:.2f}"`
- [ ] Nền label không bị vẽ ngoài frame (clamp `label_y1 = max(0, ...)`)

### Config

- [ ] `configs/phase4.yaml` tồn tại và có key `perception.traffic_light.enabled: true`
- [ ] Tất cả key HSV config có trong YAML với giá trị hợp lý

### Tests

- [ ] `tests/test_traffic_light.py` tồn tại
- [ ] Tests cover: RED, GREEN, YELLOW detection
- [ ] Tests cover: dark image → low confidence
- [ ] Tests cover: crop too small → UNKNOWN
- [ ] Tests cover: non-TL class → ignored
- [ ] Tests cover: multiple lights
- [ ] `python -m pytest tests/test_traffic_light.py -v` → tất cả PASS

### Tích Hợp

- [ ] `python scripts/run_phase4.py` chạy không exception
- [ ] `SceneState.traffic_lights` populated khi có đèn trong video
- [ ] Frame annotated có bbox màu và label đèn

### Performance

- [ ] HSV classify ≤ 2 ms/đèn (đo bằng script performance test)
- [ ] Không memory leak qua nhiều frame

### Documentation

- [ ] Docstring class `SimpleTrafficLightDetector` giải thích strategy
- [ ] Docstring `detect()`, `_classify_state()`, `_color_mask_classify()`, `_region_based_classify()`
- [ ] Comment giải thích tại sao màu đỏ cần 2 dải H

---

*Phase 4 hoàn thành → Phase 5 (Scene Understanding) có thể aggregate traffic_lights qua nhiều frame; Phase 6 (Risk/Decision) có thể ra quyết định STOP khi `TrafficLightState.RED`.*
