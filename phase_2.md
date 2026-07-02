# Phase 2 — Multi-Object Tracking (MOT)

**Mục tiêu: gán và duy trì ID ổn định cho từng vật thể qua các frame liên tiếp, tạo nền tảng dữ liệu quỹ đạo và vận tốc pixel để Phase 6 ước tính TTC (Time-to-Collision) và đánh giá rủi ro.**

---

## 1. Tổng quan & vị trí trong lộ trình

### Vị trí trong lộ trình tổng thể

```
P0 Nền tảng ──► P1 Detection + Viz ──► [P2 Tracking] ──► P3 Lane
     ✓                  ✓                   ← đây            ○
                                              │
                                              ▼
                        P6 Risk + Decision ◄── P5 Scene ◄── P4 Traffic Light
```

Phase 2 đứng ngay sau P1 (Detection đã hoạt động) và là điều kiện bắt buộc để P6 có được
dữ liệu vận tốc cần thiết cho tính toán TTC. Các phase P3, P4, P5 có thể phát triển song
song nhưng P6 **phụ thuộc trực tiếp** vào output của P2.

### Tracking là gì trong ngữ cảnh này?

Multi-Object Tracking (MOT) **không phải** là một mô hình học sâu riêng biệt theo nghĩa
thông thường — đây là **thuật toán kết hợp (association algorithm)** chạy trên đầu ra của
detector. Detector (YOLOv8) cho biết "có vật thể ở đâu trong frame này"; tracker hỏi
"vật thể ở frame này là *cùng một vật thể* với vật thể nào ở frame trước?"

Quá trình gồm ba bước lặp lại mỗi frame:
1. **Dự đoán (Predict)** — ước tính vị trí mới của track hiện có (đơn giản: giữ nguyên
   bbox cuối; nâng cao: Kalman filter).
2. **Khớp (Match / Associate)** — đo độ tương đồng giữa track cũ và detection mới (IoU,
   khoảng cách, appearance), sau đó tìm phép gán tối ưu.
3. **Cập nhật (Update)** — track được khớp nhận detection mới; track không được khớp tăng
   `time_since_update`; detection không được khớp sinh track mới.

### Hai lớp triển khai trong P2

| Lớp | Thuật toán | Dependency | Khi nào dùng |
|-----|-----------|------------|--------------|
| `SimpleTracker` (nâng cấp) | IoU + Hungarian | `scipy` (stdlib-lite) | mặc định, dễ debug |
| `ByteTracker` (mới) | ByteTrack via ultralytics | `ultralytics` | demo cao cấp |

Cả hai cùng implement interface `Tracker` trong `perception/base.py` — pipeline **không
cần biết** đang dùng loại nào.

---

## 2. Mục tiêu (đo được)

| # | Mục tiêu | Cách đo | Ngưỡng chấp nhận |
|---|---------|---------|-----------------|
| 2.1 | ID ổn định qua ít nhất 30 frame liên tiếp không có occlusion | Unit test chuỗi frame giả lập | 0 ID switch |
| 2.2 | Hungarian matching không gán hai track cho cùng một detection | Unit test assertion | Không xảy ra |
| 2.3 | `min_hits` loại bỏ track giả (ghost) khởi tạo từ detection nhiễu | Visual inspection + test | Track ID không xuất hiện trong output trước khi đủ `min_hits` |
| 2.4 | Vận tốc pixel `Track.velocity` hội tụ trong ≤ 2 frame sau khi track đủ `min_hits` | Kiểm tra giá trị khác `(0.0, 0.0)` | history ≥ 2 |
| 2.5 | Quỹ đạo (trajectory) vẽ lên frame trong Annotator | Visual inspection demo | Đường polyline màu tương ứng track_id hiện trên video |
| 2.6 | `ByteTracker` chạy không crash, trả đúng kiểu `List[Track]` | Smoke test | Pass |
| 2.7 | Tổng thời gian xử lý tracking ≤ 5 ms/frame trên CPU (SimpleTracker) với ≤ 50 tracks | Benchmark đơn giản `time.perf_counter` | ≤ 5 ms |

---

## 3. Phạm vi

### Trong phạm vi (In-scope)

- Nâng cấp `SimpleTracker`: thay greedy bằng **Hungarian matching**
  (`scipy.optimize.linear_sum_assignment`), thêm `min_hits`, giữ max lịch sử center.
- Thêm `ByteTracker` wrapper dùng `ultralytics` `model.track(persist=True)`.
- Vẽ **track ID** và **trajectory (quỹ đạo)** lên frame trong `viz/annotator.py`.
- Bổ sung tham số `min_hits` vào `configs/default.yaml`.
- Viết `tests/test_tracking.py` kiểm tra ID consistency và matching.
- Giải thích khái niệm trong comment code: association, ID switch, occlusion.

### Ngoài phạm vi (Out-of-scope)

- Kalman filter đầy đủ (lưu ý về sau, không implement trong P2).
- Re-identification bằng appearance embedding (DeepSORT style) — để Phase 8/10.
- Metric MOTA / IDF1 trên benchmark chuẩn (MOTChallenge) — để Phase 8/10.
- Multi-camera tracking.
- Tracking vật thể 3D (cần depth từ CARLA — Phase 9).
- Fine-tuning bất kỳ model nào (P2 không dùng Kaggle).

---

## 4. Điều kiện tiên quyết

| Điều kiện | Trạng thái | Ghi chú |
|-----------|-----------|---------|
| P0 — Skeleton dựng xong | ✓ Xong | types.py, base.py, pipeline.py, builder.py |
| P1 — YOLODetector trả `List[Detection]` ổn định | ✓ Xong | detection.py hoạt động |
| `Track`, `BoundingBox.iou()`, `Track.velocity` định nghĩa trong types.py | ✓ Xong | Không sửa types.py trong P2 |
| `Tracker` ABC (`perception/base.py`) | ✓ Xong | interface bất biến |
| `scipy` có sẵn trong môi trường | Cần verify | `pip install scipy` nếu thiếu |
| `ultralytics` (tùy chọn, cho ByteTracker) | Có sẵn từ P1 | import guard giống P1 |

Kiểm tra nhanh trước khi bắt đầu:

```bash
python -c "from scipy.optimize import linear_sum_assignment; print('scipy OK')"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
PYTHONPATH=src pytest -q tests/test_pipeline.py   # P1 tests phải còn xanh
```

---

## 5. Công nghệ & thư viện

### Thư viện cốt lõi

| Thư viện | Phiên bản tối thiểu | Mục đích |
|----------|-------------------|---------|
| `scipy` | ≥ 1.11 | `linear_sum_assignment` cho Hungarian algorithm |
| `numpy` | ≥ 1.26 | Ma trận cost, phép tính vector |
| `opencv-python` | ≥ 4.9 | Vẽ polyline, putText, rectangle trên frame |
| `ultralytics` | ≥ 8.2 | ByteTrack qua `model.track(persist=True)` (optional) |

### Không cần thêm dependency mới nào bắt buộc

`scipy` là dependency duy nhất thêm mới bắt buộc. Nếu không có `scipy`, có thể fallback
về greedy matching (đã có) bằng cơ chế try/except trong `SimpleTracker.__init__`.

### Cập nhật requirements.txt / pyproject.toml

```toml
# pyproject.toml — thêm vào [project.dependencies]
"scipy>=1.11",
```

---

## 6. Thiết kế chi tiết

### 6.1 Interface bất biến (không đổi)

```python
# src/drivevision/perception/base.py — KHÔNG SỬA
class Tracker(ABC):
    @abstractmethod
    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        """Nhận frame + detections, trả List[Track] đang hiển thị trong frame hiện tại."""
        ...
```

Pipeline gọi `tracker.update(frame, detections)` — output luôn là danh sách Track
**đang visible** (tức là `time_since_update == 0`), không bao giờ trả track đã mất.

### 6.2 Sơ đồ luồng xử lý mỗi frame (ASCII)

```
                      ┌─────────────────────────────────────────────┐
                      │         Tracker.update(frame, dets)          │
                      └─────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │  BƯỚC 1: Predict                               │
                    │  Với mỗi track đang sống:                      │
                    │    predicted_bbox = track.detection.bbox       │
                    │    (SimpleTracker: giữ nguyên bbox cuối cùng)  │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │  BƯỚC 2: Build cost matrix                     │
                    │  cost[i][j] = 1 - IoU(track_i, det_j)         │
                    │  Kích thước: (n_tracks) x (n_dets)            │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │  BƯỚC 3: Hungarian matching                    │
                    │  row_ind, col_ind = linear_sum_assignment(cost)│
                    │  Lọc: chỉ giữ cặp có IoU ≥ iou_threshold      │
                    └───────────┬───────────────────────┬───────────┘
                                │                       │
               ┌────────────────▼──────┐    ┌───────────▼──────────────┐
               │  Matched pairs        │    │  Unmatched               │
               │  (track_i ↔ det_j)   │    │  tracks → time_since_    │
               │                       │    │  update += 1             │
               │  track.detection = det│    │                          │
               │  track.age += 1       │    │  detections → spawn      │
               │  track.tsu = 0        │    │  new Track (tsu=0,age=0) │
               │  track.history.append │    └──────────────────────────┘
               │    (det.bbox.center)  │
               └────────────────┬──────┘
                                │
                    ┌───────────▼───────────────────────────────────┐
                    │  BƯỚC 4: Prune stale tracks                    │
                    │  Xoá track nếu time_since_update > max_age    │
                    └───────────┬───────────────────────────────────┘
                                │
                    ┌───────────▼───────────────────────────────────┐
                    │  BƯỚC 5: Filter confirmed tracks               │
                    │  Chỉ trả track có age >= min_hits AND tsu == 0 │
                    └───────────────────────────────────────────────┘
```

### 6.3 Ma trận cost và Hungarian algorithm

**Vấn đề với Greedy (hiện tại):**
Trong greedy matching, track đầu tiên trong danh sách "giành" detection tốt nhất, dù
detection đó có thể phù hợp hơn với track khác. Điều này dẫn đến ID switch không cần thiết
khi hai vật thể đi gần nhau.

**Giải pháp Hungarian:**
Bài toán gán tối ưu — tìm phép gán 1-1 giữa n tracks và m detections sao cho tổng
chi phí (cost) nhỏ nhất. `scipy.optimize.linear_sum_assignment` giải trong O(n³).

```
Cost matrix (IoU-based):
          det_0   det_1   det_2
track_0 [  0.1     0.9     0.8  ]   cost = 1 - IoU
track_1 [  0.95    0.05    0.7  ]
track_2 [  0.85    0.8     0.1  ]

Greedy kết quả:  track_0→det_0,  track_1→det_1,  track_2→det_2  (tổng cost = 0.1+0.05+0.1 = 0.25)
Hungarian kết quả: giống nhau ở ví dụ này, nhưng khác khi có xung đột.

Kịch bản xung đột:
          det_0   det_1
track_0 [  0.1     0.2  ]
track_1 [  0.15    0.9  ]

Greedy: track_0 lấy det_0 (0.1), track_1 còn lại det_1 (0.9). Tổng = 1.0
Hungarian: track_0→det_0, track_1→det_1. Tổng = 1.0 (giống)

Nhưng:
          det_0   det_1
track_0 [  0.1     0.2  ]
track_1 [  0.12    0.9  ]

Greedy (thứ tự track_0 trước): track_0→det_0, track_1→det_1. Tổng = 1.0
Greedy (thứ tự track_1 trước): track_1→det_0, track_0→det_1. Tổng = 1.02
Hungarian: track_0→det_0, track_1→det_1. Tổng = 1.0 (luôn tối ưu bất kể thứ tự)
```

**Ngưỡng IoU sau khi gán:**
Sau khi Hungarian trả về các cặp, loại bỏ cặp nào có `IoU < iou_threshold`. Đây là
bước quan trọng vì Hungarian *bắt buộc* gán (nếu cost matrix không vuông, nó gán theo
chiều nhỏ hơn), nên cần lọc lại các match chất lượng thấp.

### 6.4 Tham số `min_hits` — tránh track giả (ghost tracks)

**Vấn đề:** Detector đôi khi sinh false positive — detection nhìn thấy trong 1-2 frame
rồi biến mất. Nếu ngay lập tức phát sinh Track với ID, dashboard sẽ hiển thị các ID
nhảy bất thường.

**Giải pháp `min_hits`:** Một Track chỉ được đưa vào output khi `age >= min_hits`.
Trong thời gian "chờ xét", track vẫn được cập nhật bình thường nhưng không xuất hiện
trong `List[Track]` trả về.

```
min_hits = 3

Frame 1: Detection A xuất hiện → Track(id=5, age=0)  → KHÔNG trả ra
Frame 2: Detection A xuất hiện → Track(id=5, age=1)  → KHÔNG trả ra
Frame 3: Detection A xuất hiện → Track(id=5, age=2)  → KHÔNG trả ra
Frame 4: Detection A xuất hiện → Track(id=5, age=3)  → TRẢ RA (age >= min_hits)
```

**Đánh đổi:** `min_hits` cao → ít ghost nhưng track thật xuất hiện muộn hơn (trễ 
`min_hits` frame). Giá trị mặc định `min_hits=3` hợp lý ở 30fps (≈ 100ms trễ).

### 6.5 Khái niệm: ID Switch và Occlusion

**ID Switch:** Xảy ra khi hai vật thể giao nhau và tracker gán nhầm ID sau khi tách ra.
```
Frame 10: car_A(id=1) ...... car_B(id=2)
Frame 15: car_A(id=1) ≈ car_B(id=2)  ← gần nhau, IoU cao lẫn nhau
Frame 20: car_A(id=2) ...... car_B(id=1)  ← ID bị hoán đổi!
```
Hungarian giảm thiểu nhưng không loại hoàn toàn ID switch khi IoU hai phương án gần bằng nhau.

**Occlusion (che khuất):** Vật thể bị vật khác che → detector không sinh detection.
Track sẽ có `time_since_update` tăng dần. Nếu `time_since_update <= max_age`, track
"sống sót" và có thể re-link khi vật thể xuất hiện lại. Đây là re-identification đơn giản
nhất (chỉ dựa vào vị trí không gian, không cần appearance).

**Motion model:** SimpleTracker dùng "constant position" — bbox dự đoán = bbox cuối cùng.
Kalman filter (Phase 8/10 nếu cần) dùng constant velocity để dự đoán bbox tốt hơn khi
vật thể bị che khuất một phần.

### 6.6 ByteTrack — phương án nâng cao

ByteTrack (Zhang et al., 2022) cải tiến trên SORT bằng cách **tái sử dụng cả detection
confidence thấp** trong bước matching thứ hai, giúp giảm ID switch khi vật thể bị che
khuất một phần. Ultralytics tích hợp sẵn ByteTrack.

Cách sử dụng qua ultralytics:
```python
model = YOLO("yolov8n.pt")
results = model.track(source=frame_bgr, persist=True, tracker="bytetrack.yaml")
# persist=True: ultralytics giữ tracker state giữa các lần gọi
# → trả về results[0].boxes.id  (tensor các track_id)
```

Wrapper `ByteTracker` trong `perception/tracking.py` chuyển đổi output ultralytics sang
`List[Track]` theo đúng interface, không để logic ByteTrack "rò rỉ" ra pipeline.

**So sánh SimpleTracker vs ByteTracker:**

| Tiêu chí | SimpleTracker (nâng cấp) | ByteTracker |
|---------|--------------------------|-------------|
| Dependency | scipy | ultralytics |
| ID switch rate | Trung bình | Thấp hơn |
| Tốc độ (CPU) | ~1-2 ms | ~5-15 ms (bao gồm inference) |
| Debug | Dễ (code tự viết) | Khó (black box) |
| Kiểm soát | Hoàn toàn | Hạn chế |
| Phù hợp | Học thuật, demo cơ bản | Demo cao cấp |

Chiến lược: mặc định dùng SimpleTracker, bật ByteTracker qua config khi muốn showcase.

### 6.7 Velocity pixel và liên kết với Phase 6

`Track.velocity` (đã có trong `types.py`) trả `(vx, vy)` là delta pixel giữa hai center
liên tiếp cuối cùng trong `history`. Phase 6 dùng velocity như sau:

```
Nếu vật thể đang di chuyển với vx > 0 (dịch về phải, tức là ra xa nếu camera hướng trước),
hoặc bbox tăng kích thước → đang lại gần.

TTC ≈ (distance_to_object) / (closing_speed)

Trong không gian pixel:
  - "Distance" ~ 1 / sqrt(bbox.area)   (vật thể càng gần, bbox càng lớn)
  - "Closing speed" ~ (bbox_area[t] - bbox_area[t-1]) / dt

Phase 6 sẽ đọc track.history để tính bbox_area qua các frame.
→ P2 phải đảm bảo history lưu ĐỦ dài (hiện tại cắt tại 30 entries là hợp lý).
```

---

## 7. Công việc chi tiết

### Task 2.1 — Nâng cấp SimpleTracker: Hungarian matching + min_hits

**Mục đích:** Thay thuật toán greedy bằng Hungarian để matching tối ưu toàn cục; thêm
`min_hits` để lọc ghost track.

**File:** `src/drivevision/perception/tracking.py`

**Các bước:**
1. Thêm `min_hits: int = 3` vào `__init__`.
2. Import `linear_sum_assignment` từ `scipy.optimize` (lazy import, try/except).
3. Refactor `update()`:
   a. Xây dựng cost matrix numpy shape `(n_tracks, n_dets)`.
   b. Gọi `linear_sum_assignment(cost_matrix)`.
   c. Lọc matched pairs theo `iou_threshold`.
   d. Xác định unmatched tracks và unmatched detections.
   e. Spawn tracks mới cho unmatched detections.
   f. Drop track nếu `time_since_update > max_age`.
   g. Return chỉ track có `time_since_update == 0 AND age >= min_hits`.

**Pseudocode / Skeleton:**

```python
"""src/drivevision/perception/tracking.py"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..types import Detection, Frame, Track
from .base import Tracker

log = logging.getLogger("drivevision.tracking")

# Lazy import — scipy không bắt buộc
try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    log.warning(
        "scipy không tìm thấy — SimpleTracker fallback về greedy matching. "
        "Cài scipy để dùng Hungarian: pip install scipy"
    )


def _build_iou_cost_matrix(
    tracks: List[Track], detections: List[Detection]
) -> np.ndarray:
    """Tạo cost matrix kích thước (n_tracks, n_dets).

    cost[i][j] = 1 - IoU(track_i.bbox, det_j.bbox)
    Giá trị nằm trong [0, 1]: 0 = khớp hoàn hảo, 1 = không chồng lấp.
    """
    n_t = len(tracks)
    n_d = len(detections)
    cost = np.ones((n_t, n_d), dtype=np.float32)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = track.detection.bbox.iou(det.bbox)
            cost[i, j] = 1.0 - iou
    return cost


def _hungarian_match(
    cost: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Chạy Hungarian, lọc theo ngưỡng IoU.

    Trả về:
        matched   : danh sách (track_idx, det_idx)
        unmatched_tracks  : danh sách track_idx không được khớp
        unmatched_dets    : danh sách det_idx không được khớp
    """
    n_t, n_d = cost.shape
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    row_ind, col_ind = _lsa(cost)

    matched = []
    matched_track_set = set()
    matched_det_set = set()

    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost[r, c]
        if iou >= iou_threshold:
            matched.append((r, c))
            matched_track_set.add(r)
            matched_det_set.add(c)

    unmatched_tracks = [i for i in range(n_t) if i not in matched_track_set]
    unmatched_dets   = [j for j in range(n_d) if j not in matched_det_set]
    return matched, unmatched_tracks, unmatched_dets


def _greedy_match(
    tracks: List[Track], detections: List[Detection], iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Fallback greedy matching khi không có scipy."""
    unmatched_dets = list(range(len(detections)))
    matched = []
    matched_track_set = set()

    for i, track in enumerate(tracks):
        best_iou, best_j = iou_threshold, -1
        for j in unmatched_dets:
            score = track.detection.bbox.iou(detections[j].bbox)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_j >= 0:
            matched.append((i, best_j))
            matched_track_set.add(i)
            unmatched_dets.remove(best_j)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_set]
    return matched, unmatched_tracks, unmatched_dets


class SimpleTracker(Tracker):
    """IoU-based tracker với Hungarian matching.

    Tham số:
        max_age        : số frame track "sống sót" mà không có detection nào khớp.
                         Track bị xoá khi time_since_update > max_age.
        iou_threshold  : IoU tối thiểu để coi một cặp (track, detection) là khớp.
        min_hits       : số frame liên tiếp track phải được thấy trước khi xuất hiện
                         trong output. Giảm ghost tracks từ false positive.
        history_len    : số center tối đa lưu trong track.history (dùng cho trajectory).
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
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API (implements Tracker ABC)
    # ------------------------------------------------------------------

    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        """Cập nhật tracker với detections của frame hiện tại.

        Quy trình:
        1. Xây cost matrix IoU giữa tracks hiện có và detections mới.
        2. Chạy Hungarian (hoặc greedy nếu không có scipy).
        3. Cập nhật matched tracks, tăng tsu cho unmatched, spawn track mới.
        4. Xoá track quá hạn (tsu > max_age).
        5. Trả về chỉ track đang visible (tsu == 0) VÀ đã "xác nhận" (age >= min_hits).
        """
        # Bước 1 & 2: Matching
        if _HAS_SCIPY:
            cost = _build_iou_cost_matrix(self._tracks, detections)
            matched, unmatched_t, unmatched_d = _hungarian_match(cost, self.iou_threshold)
        else:
            matched, unmatched_t, unmatched_d = _greedy_match(
                self._tracks, detections, self.iou_threshold
            )

        # Bước 3a: Cập nhật matched tracks
        for (t_idx, d_idx) in matched:
            track = self._tracks[t_idx]
            det = detections[d_idx]
            track.detection = det
            track.age += 1
            track.time_since_update = 0
            track.history.append(det.bbox.center)
            if len(track.history) > self.history_len:
                track.history = track.history[-self.history_len:]

        # Bước 3b: Tăng time_since_update cho unmatched tracks
        for t_idx in unmatched_t:
            self._tracks[t_idx].time_since_update += 1

        # Bước 3c: Spawn tracks mới cho unmatched detections
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

        # Bước 4: Xoá track quá hạn
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # Bước 5: Trả về track đang visible VÀ đã xác nhận
        return [
            t for t in self._tracks
            if t.time_since_update == 0 and t.age >= self.min_hits
        ]

    # ------------------------------------------------------------------
    # Introspection helpers (hữu ích cho debug & test)
    # ------------------------------------------------------------------

    @property
    def all_tracks(self) -> List[Track]:
        """Tất cả track đang sống (kể cả chưa xác nhận và đang mất)."""
        return list(self._tracks)

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Reset tracker về trạng thái ban đầu (hữu ích cho test)."""
        self._tracks.clear()
        self._next_id = 0
```

**Lưu ý & edge cases:**
- Khi `detections = []`: `unmatched_t = list(range(n_tracks))`, `unmatched_d = []`.
  Tất cả track tăng `tsu`. Đây là trường hợp bình thường (frame bị nhiễu, YOLO confidence thấp).
- Khi `self._tracks = []` (frame đầu tiên): Cost matrix shape `(0, n_dets)`, tất cả det
  được spawn thành track mới. `_hungarian_match` phải xử lý đúng trường hợp này.
- `history_len=30` ở 30fps = 1 giây lịch sử — đủ cho trajectory visualization nhưng không
  chiếm nhiều bộ nhớ.
- Nếu hai vật thể hoàn toàn chồng lấp (IoU = 1.0 cho cả hai cặp): Hungarian vẫn tìm
  được phép gán tối ưu; greedy sẽ gán nhầm.

---

### Task 2.2 — ByteTracker wrapper

**Mục đích:** Cung cấp implementation mạnh hơn dưới cùng interface `Tracker`, dùng khi
muốn demo tracking chất lượng cao mà không cần tự implement Kalman filter.

**File:** `src/drivevision/perception/tracking.py` (thêm class mới vào cuối file)

**Cách ByteTrack hoạt động qua ultralytics:**
```
model.track(source, persist=True) trên mỗi frame:
→ ultralytics gọi ByteTrack nội bộ với Kalman filter
→ trả Results object chứa:
    - boxes.xyxy   : tensor [N, 4]  tọa độ bbox
    - boxes.id     : tensor [N]     track ID (None nếu không có track)
    - boxes.conf   : tensor [N]     confidence
    - boxes.cls    : tensor [N]     class id
```

**Pseudocode / Skeleton:**

```python
class ByteTracker(Tracker):
    """Wrapper quanh ultralytics ByteTrack.

    Giao diện giống hệt SimpleTracker — pipeline không phân biệt.
    Khác biệt: ByteTracker tự chạy detection bên trong (dùng YOLO model),
    nên tham số `detections` từ pipeline bị bỏ qua — tracker dùng
    detections nội bộ của ultralytics.

    Lưu ý quan trọng:
        Vì ByteTracker chạy detection nội bộ, nó KHÔNG dùng `detections`
        argument. Builder phải tắt detector riêng khi dùng ByteTracker
        (hoặc chấp nhận duplicate inference). Ghi rõ trong config.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        classes: list[int] | None = None,
        history_len: int = 30,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ByteTracker cần ultralytics: pip install ultralytics"
            ) from exc

        self._model = YOLO(weights)
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._classes = classes
        self.history_len = history_len
        # history lưu ngoài ultralytics để tính velocity
        self._histories: dict[int, list[tuple[float, float]]] = {}

    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        """Chạy ByteTrack qua ultralytics model.track().

        `detections` argument bị bỏ qua — ultralytics tự detect.
        """
        import numpy as np  # đã có trong scope nhưng explicit hơn

        results = self._model.track(
            source=frame.image,
            persist=True,           # giữ tracker state giữa các lần gọi
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            classes=self._classes,
            verbose=False,
        )

        tracks: List[Track] = []
        if not results or results[0].boxes is None:
            return tracks

        boxes = results[0].boxes
        if boxes.id is None:
            # Không có track ID → tracker chưa khởi tạo đủ
            return tracks

        ids     = boxes.id.cpu().numpy().astype(int)
        xyxy    = boxes.xyxy.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()
        clss    = boxes.cls.cpu().numpy().astype(int)

        # Lấy class names từ model
        names = results[0].names  # dict {class_id: class_name}

        for track_id, box, conf, cls_id in zip(ids, xyxy, confs, clss):
            bbox = BoundingBox(
                x1=float(box[0]), y1=float(box[1]),
                x2=float(box[2]), y2=float(box[3]),
            )
            det = Detection(
                bbox=bbox,
                class_id=int(cls_id),
                class_name=names.get(int(cls_id), str(cls_id)),
                confidence=float(conf),
            )

            # Cập nhật history thủ công
            center = bbox.center
            hist = self._histories.get(track_id, [])
            hist.append(center)
            if len(hist) > self.history_len:
                hist = hist[-self.history_len:]
            self._histories[track_id] = hist

            tracks.append(
                Track(
                    track_id=int(track_id),
                    detection=det,
                    age=len(hist),
                    time_since_update=0,
                    history=list(hist),
                )
            )

        # Dọn history của track đã mất (không xuất hiện trong frame này)
        active_ids = set(ids.tolist())
        self._histories = {k: v for k, v in self._histories.items() if k in active_ids}

        return tracks
```

Cần thêm import `BoundingBox` từ `..types` vào đầu file.

**Lưu ý & edge cases:**
- `persist=True` là bắt buộc — không có nó, ultralytics reset tracker mỗi frame.
- Lần đầu gọi `model.track`, ultralytics khởi tạo ByteTrack → có thể chậm ~200ms;
  các frame sau nhanh hơn nhiều.
- Nếu `boxes.id is None` (xảy ra khi frame đầu tiên hoặc không track nào được confirm),
  trả list rỗng thay vì crash.
- `self._histories` cần được dọn định kỳ nếu video rất dài (nhiều triệu track ID khác nhau);
  trong thực tế demo không thành vấn đề.

**Cập nhật builder.py** để hỗ trợ `ByteTracker`:

```python
# src/drivevision/pipeline/builder.py — trong hàm build_pipeline()
if get_path(cfg, "perception.tracking.enabled"):
    backend = get_path(cfg, "perception.tracking.backend", "simple")
    if backend == "bytetrack":
        try:
            from ..perception.tracking import ByteTracker
            tracker = ByteTracker(
                weights=get_path(cfg, "perception.detection.weights", "yolov8n.pt"),
                conf=get_path(cfg, "perception.detection.conf", 0.35),
                iou=get_path(cfg, "perception.detection.iou", 0.5),
                imgsz=get_path(cfg, "perception.detection.imgsz", 640),
                classes=get_path(cfg, "perception.detection.classes"),
            )
            # Khi dùng ByteTracker, tắt detector riêng để tránh duplicate inference
            detector = None
            log.info("Dùng ByteTracker (ultralytics). Detector riêng đã tắt.")
        except ImportError as exc:
            log.warning("ByteTracker không khả dụng (%s). Fallback SimpleTracker.", exc)
            backend = "simple"
    if backend == "simple":
        from ..perception.tracking import SimpleTracker
        tracker = SimpleTracker(
            max_age=get_path(cfg, "perception.tracking.max_age", 30),
            iou_threshold=get_path(cfg, "perception.tracking.iou_threshold", 0.3),
            min_hits=get_path(cfg, "perception.tracking.min_hits", 3),
        )
```

---

### Task 2.3 — Vẽ Track ID và Trajectory trong Annotator

**Mục đích:** Hiển thị ID ổn định và quỹ đạo di chuyển của từng vật thể lên frame — đây
là phần demo trực quan nhất của P2.

**File:** `src/drivevision/viz/annotator.py`

**Thiết kế trực quan:**
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│    ┌──────────────┐                                 │
│    │  car #3      │  ← bbox + class_name + track_id │
│    │  0.92        │  ← confidence                   │
│    └──────────────┘                                 │
│            ·                                        │
│           ·                                         │  ← trajectory (polyline)
│          ·                                          │
│         ●  ← current center                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Màu sắc: mỗi `track_id` có màu riêng, ổn định qua các frame:
```python
COLOR_PALETTE = [
    (255,  56,  56),  # đỏ tươi
    ( 56, 255,  56),  # xanh lá
    ( 56,  56, 255),  # xanh dương
    (255, 157,  56),  # cam
    (255,  56, 157),  # hồng
    (157,  56, 255),  # tím
    ( 56, 255, 157),  # ngọc
    (255, 255,  56),  # vàng
]

def _track_color(track_id: int) -> tuple[int, int, int]:
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]
```

**Pseudocode / Skeleton đầy đủ:**

```python
"""src/drivevision/viz/annotator.py"""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from ..types import PipelineResult, Track

# Bảng màu BGR — một màu ứng với mỗi track_id (mod palette)
_PALETTE: list[Tuple[int, int, int]] = [
    (  56, 56, 255),  # đỏ
    (  56, 255, 56),  # xanh lá
    (255,  56, 56),   # xanh dương
    (  56, 157, 255), # cam
    (157,  56, 255),  # tím
    (255, 255,  56),  # vàng
    (  56, 255, 157), # ngọc
    (255,  56, 157),  # hồng
]


def _color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


class Annotator:
    """Vẽ kết quả pipeline lên một frame BGR.

    Không giữ state — stateless transformer.
    """

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        font_thickness: int = 1,
        trajectory_min_len: int = 2,  # ít nhất 2 điểm mới vẽ đường
        trajectory_alpha: float = 0.6,  # độ trong suốt trajectory (unused hiện tại)
    ) -> None:
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.trajectory_min_len = trajectory_min_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, result: PipelineResult) -> np.ndarray:
        """Trả về frame BGR đã được annotate."""
        canvas = result.frame.image.copy()
        self._draw_tracks(canvas, result.scene.tracks)
        self._draw_hud(canvas, result)
        return canvas

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_tracks(self, canvas: np.ndarray, tracks: list[Track]) -> None:
        """Vẽ bbox, track ID, confidence, và trajectory cho từng track."""
        for track in tracks:
            color = _color(track.track_id)
            self._draw_bbox(canvas, track, color)
            self._draw_label(canvas, track, color)
            self._draw_trajectory(canvas, track, color)

    def _draw_bbox(
        self, canvas: np.ndarray, track: Track, color: Tuple[int, int, int]
    ) -> None:
        x1, y1, x2, y2 = track.detection.bbox.as_int()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness)

    def _draw_label(
        self, canvas: np.ndarray, track: Track, color: Tuple[int, int, int]
    ) -> None:
        det = track.detection
        label = f"#{track.track_id} {det.class_name} {det.confidence:.2f}"
        x1, y1, _, _ = det.bbox.as_int()

        # Nền cho text
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        bg_y1 = max(y1 - th - baseline - 4, 0)
        bg_y2 = y1
        bg_x2 = x1 + tw + 4
        cv2.rectangle(canvas, (x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Text trắng trên nền màu
        cv2.putText(
            canvas, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_trajectory(
        self, canvas: np.ndarray, track: Track, color: Tuple[int, int, int]
    ) -> None:
        """Vẽ polyline nối các center trong lịch sử quỹ đạo.

        Màu đậm dần về phía điểm hiện tại (alpha blending đơn giản bằng
        cách thay đổi độ dày nét vẽ).
        """
        hist = track.history
        if len(hist) < self.trajectory_min_len:
            return

        pts = np.array(hist, dtype=np.int32)

        # Vẽ các đoạn thẳng với độ dày tăng dần (biểu thị thời gian)
        n = len(pts)
        for k in range(1, n):
            # Độ mờ theo khoảng cách về quá khứ: cũ hơn → mỏng hơn
            alpha = k / n  # 0 (cũ nhất) → 1 (mới nhất)
            thickness = max(1, int(3 * alpha))
            pt1 = tuple(pts[k - 1].tolist())
            pt2 = tuple(pts[k].tolist())
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Đánh dấu center hiện tại bằng chấm tròn
        cx, cy = pts[-1].tolist()
        cv2.circle(canvas, (cx, cy), 4, color, -1)

    def _draw_hud(self, canvas: np.ndarray, result: PipelineResult) -> None:
        """Hiển thị HUD ở góc trên trái: frame index, số track, risk level."""
        h, w = canvas.shape[:2]
        lines = [
            f"Frame #{result.frame.index}",
            f"Tracks: {len(result.scene.tracks)}",
        ]
        if result.risk:
            lines.append(f"Risk: {result.risk.level.value.upper()}")
        if result.decision:
            lines.append(f"Action: {result.decision.action.value}")

        y0 = 20
        for i, line in enumerate(lines):
            y = y0 + i * 22
            # Viền đen để dễ đọc trên nền bất kỳ
            cv2.putText(
                canvas, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                canvas, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )
```

**Lưu ý & edge cases:**
- `track.history` chứa `Tuple[float, float]` — cần ép kiểu sang `int` khi vẽ cv2.
- Nếu `history` rỗng hoặc chỉ 1 điểm, bỏ qua vẽ trajectory (không crash).
- HUD dùng putText hai lần (đen rồi trắng) — kỹ thuật "outline text" giúp đọc được
  trên cả nền sáng và tối.
- Không import `Frame` trong annotator — chỉ dùng qua `result.frame` để tránh circular.

---

### Task 2.4 — Cập nhật cấu hình

**Mục đích:** Bổ sung `min_hits` và `backend` vào config; giữ backward compatibility
(file config cũ không có hai key này vẫn chạy được nhờ default trong builder).

**File:** `configs/default.yaml`

Thêm vào section `perception.tracking`:

```yaml
perception:
  # ... (giữ nguyên detection section)
  tracking:
    enabled: true
    backend: simple          # simple | bytetrack
    max_age: 30              # frames track sống sót không có detection
    iou_threshold: 0.3       # IoU tối thiểu để matching
    min_hits: 3              # frames liên tiếp cần thấy trước khi confirm track
```

---

### Task 2.5 — Viết test ID consistency

**Mục đích:** Xác minh rằng SimpleTracker gán ID ổn định qua nhiều frame, không bị ID switch
trong điều kiện lý tưởng (không occlusion, không nhiễu).

**File:** `tests/test_tracking.py` (file mới)

**Các bước:**
1. Tạo chuỗi Detection giả lập chuyển động tuyến tính.
2. Chạy tracker qua N frame.
3. Assert ID không thay đổi.
4. Test trường hợp track mất rồi xuất hiện lại.
5. Test `min_hits` filter.
6. Test edge case: không có detection.

**Pseudocode / Skeleton:**

```python
"""tests/test_tracking.py

Kiểm tra tính nhất quán ID của SimpleTracker.
Không cần GPU, không cần ultralytics — chỉ dùng numpy + scipy.
"""

import numpy as np
import pytest

from drivevision.perception.tracking import SimpleTracker
from drivevision.types import BoundingBox, Detection, Frame


# ------------------------------------------------------------------ helpers ---

def _frame(idx: int = 0) -> Frame:
    return Frame(index=idx, timestamp=float(idx) / 30.0,
                 image=np.zeros((480, 640, 3), dtype=np.uint8))


def _det(x1: float, y1: float, x2: float, y2: float,
         cls: str = "car", conf: float = 0.9) -> Detection:
    return Detection(
        bbox=BoundingBox(x1, y1, x2, y2),
        class_id=2,
        class_name=cls,
        confidence=conf,
    )


def _move(det: Detection, dx: float = 5.0, dy: float = 0.0) -> Detection:
    """Dịch chuyển detection theo delta."""
    b = det.bbox
    return _det(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy,
                det.class_name, det.confidence)


# ------------------------------------------------------------------ tests ----

class TestSimpleTrackerIDConsistency:
    """ID phải ổn định qua các frame liên tiếp."""

    def test_single_object_stable_id(self):
        """Một vật thể di chuyển thẳng → cùng ID suốt."""
        tracker = SimpleTracker(max_age=5, iou_threshold=0.3, min_hits=1)
        det = _det(100, 100, 200, 200)
        seen_ids = set()

        for i in range(30):
            tracks = tracker.update(_frame(i), [det])
            assert len(tracks) == 1, f"Frame {i}: expected 1 track, got {len(tracks)}"
            seen_ids.add(tracks[0].track_id)
            det = _move(det, dx=2.0)  # di chuyển nhỏ, IoU vẫn cao

        assert len(seen_ids) == 1, f"ID switch xảy ra! Các ID gặp: {seen_ids}"

    def test_two_objects_no_swap(self):
        """Hai vật thể cách nhau đủ xa → ID không hoán đổi."""
        tracker = SimpleTracker(max_age=5, iou_threshold=0.3, min_hits=1)
        det_a = _det( 50, 100, 150, 200)  # bên trái
        det_b = _det(400, 100, 500, 200)  # bên phải

        # Ghi nhận ID ban đầu
        tracks = tracker.update(_frame(0), [det_a, det_b])
        assert len(tracks) == 2
        id_left  = next(t.track_id for t in tracks if t.detection.bbox.x1 < 200)
        id_right = next(t.track_id for t in tracks if t.detection.bbox.x1 > 300)

        for i in range(1, 20):
            det_a = _move(det_a, dx=3.0)
            det_b = _move(det_b, dx=-3.0)  # hai vật thể đến gần nhau
            tracks = tracker.update(_frame(i), [det_a, det_b])

            if len(tracks) == 2:
                cur_left  = next(
                    (t.track_id for t in tracks if t.detection.bbox.x1 < 300), None
                )
                cur_right = next(
                    (t.track_id for t in tracks if t.detection.bbox.x1 >= 300), None
                )
                if cur_left is not None:
                    assert cur_left == id_left, f"Frame {i}: ID trái bị swap!"
                if cur_right is not None:
                    assert cur_right == id_right, f"Frame {i}: ID phải bị swap!"

    def test_min_hits_filters_ghost(self):
        """Track chưa đủ min_hits không được phép xuất hiện trong output."""
        tracker = SimpleTracker(max_age=10, iou_threshold=0.3, min_hits=3)
        det = _det(100, 100, 200, 200)

        tracks0 = tracker.update(_frame(0), [det])
        assert len(tracks0) == 0, "Frame 0: track chưa đủ min_hits=3, không được trả ra"

        tracks1 = tracker.update(_frame(1), [det])
        assert len(tracks1) == 0, "Frame 1: age=1, vẫn chưa đủ"

        tracks2 = tracker.update(_frame(2), [det])
        assert len(tracks2) == 0, "Frame 2: age=2, vẫn chưa đủ"

        tracks3 = tracker.update(_frame(3), [det])
        assert len(tracks3) == 1, "Frame 3: age=3 >= min_hits=3, phải xuất hiện"

    def test_track_survives_occlusion(self):
        """Track sống sót qua max_age frame không có detection."""
        tracker = SimpleTracker(max_age=5, iou_threshold=0.3, min_hits=1)
        det = _det(100, 100, 200, 200)

        # Khởi tạo track
        tracker.update(_frame(0), [det])

        # Occlusion: không có detection trong 4 frame (< max_age=5)
        for i in range(1, 5):
            tracks = tracker.update(_frame(i), [])
            assert len(tracks) == 0  # không visible nhưng track vẫn sống trong _tracks

        # Track vẫn sống (age trong _tracks)
        assert tracker.track_count == 1, "Track phải vẫn còn sống sau 4 frame occlusion"

        # Xuất hiện lại
        tracks = tracker.update(_frame(5), [det])
        assert len(tracks) == 1, "Track phải re-link khi detection xuất hiện lại"

    def test_track_dies_after_max_age(self):
        """Track bị xoá sau max_age frame không được khớp."""
        tracker = SimpleTracker(max_age=3, iou_threshold=0.3, min_hits=1)
        det = _det(100, 100, 200, 200)

        tracker.update(_frame(0), [det])

        # max_age + 1 frame không có detection
        for i in range(1, 5):
            tracker.update(_frame(i), [])

        assert tracker.track_count == 0, "Track phải bị xoá sau khi vượt max_age"

    def test_no_detection_no_crash(self):
        """Tracker không crash khi danh sách detection rỗng ngay từ đầu."""
        tracker = SimpleTracker(min_hits=1)
        for i in range(5):
            tracks = tracker.update(_frame(i), [])
            assert tracks == []

    def test_velocity_converges(self):
        """velocity trả (vx, vy) khác (0, 0) sau khi history đủ 2 điểm."""
        tracker = SimpleTracker(max_age=10, iou_threshold=0.3, min_hits=1)
        det = _det(100, 100, 200, 200)

        tracker.update(_frame(0), [det])
        det = _move(det, dx=10.0)
        tracks = tracker.update(_frame(1), [det])

        assert len(tracks) == 1
        vx, vy = tracks[0].velocity
        assert vx != 0.0 or vy != 0.0, "velocity phải khác (0,0) khi bbox đã di chuyển"

    def test_history_length_capped(self):
        """history không vượt quá history_len entries."""
        tracker = SimpleTracker(max_age=200, iou_threshold=0.3, min_hits=1,
                                history_len=10)
        det = _det(100, 100, 200, 200)

        for i in range(50):
            det = _move(det, dx=1.0)
            tracker.update(_frame(i), [det])

        for t in tracker.all_tracks:
            assert len(t.history) <= 10, \
                f"history dài {len(t.history)} > history_len=10"


class TestHungarianVsGreedy:
    """Đảm bảo Hungarian matching chọn phép gán tốt hơn hoặc bằng greedy."""

    def test_matching_assigns_correct_track(self):
        """Khi hai track gần nhau, Hungarian gán đúng track cho đúng detection."""
        tracker = SimpleTracker(max_age=5, iou_threshold=0.1, min_hits=1)

        # Hai track ban đầu
        det_a = _det(100, 100, 200, 200)
        det_b = _det(210, 100, 310, 200)
        tracker.update(_frame(0), [det_a, det_b])

        # Cả hai dịch phải một chút
        det_a2 = _det(105, 100, 205, 200)
        det_b2 = _det(215, 100, 315, 200)
        tracks = tracker.update(_frame(1), [det_a2, det_b2])

        assert len(tracks) == 2
        # ID phải giữ nguyên (không swap)
        ids = sorted(t.track_id for t in tracks)
        assert ids == [0, 1], f"IDs sau frame 2 phải là [0, 1], nhận: {ids}"
```

**Lưu ý & edge cases:**
- Tất cả test dùng frame giả (numpy zeros) — không cần video thật.
- `min_hits=1` trong phần lớn test để không cần đợi nhiều frame trước khi track xuất hiện.
- Test `test_two_objects_no_swap` sử dụng logic "phân biệt trái/phải theo x1" — không phụ
  thuộc vào thứ tự list detector trả về.
- Nếu scipy không có, `SimpleTracker` dùng greedy → `test_matching_assigns_correct_track`
  vẫn pass trong điều kiện đơn giản nhưng có thể fail khi IoU rất sít. Đây là trade-off
  được chấp nhận (test đủ đơn giản để cả greedy pass được).

---

### Task 2.6 — Smoke test ByteTracker

**Mục đích:** Xác nhận `ByteTracker` không crash và trả đúng kiểu, không test tracking
quality (black box ultralytics).

**File:** `tests/test_tracking.py` (thêm vào cuối)

```python
class TestByteTrackerSmoke:
    """Smoke test — chỉ kiểm tra interface, không kiểm tra chất lượng tracking."""

    @pytest.mark.skipif(
        not _has_ultralytics(),
        reason="ultralytics không có sẵn"
    )
    def test_bytetracker_returns_list_of_tracks(self):
        from drivevision.perception.tracking import ByteTracker
        from drivevision.types import Track

        tracker = ByteTracker(weights="yolov8n.pt", conf=0.5)
        frame = _frame(0)
        frame = Frame(
            index=0, timestamp=0.0,
            image=np.zeros((640, 640, 3), dtype=np.uint8)
        )
        result = tracker.update(frame, [])
        assert isinstance(result, list)
        for t in result:
            assert isinstance(t, Track)


def _has_ultralytics() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False
```

---

## 8. Thay đổi cấu hình

### configs/default.yaml — diff cần áp dụng

```yaml
perception:
  detection:
    enabled: true
    weights: models/weights/yolo.pt
    conf: 0.35
    iou: 0.5
    classes: null
    imgsz: 640
  tracking:
    enabled: true
    backend: simple          # THÊM MỚI: simple | bytetrack
    max_age: 30
    iou_threshold: 0.3
    min_hits: 3              # THÊM MỚI: tránh ghost tracks
    # history_len mặc định 30, không expose nếu không cần tuning
```

### pyproject.toml — thêm dependency

```toml
[project]
dependencies = [
    # ... existing ...
    "scipy>=1.11",           # THÊM MỚI: Hungarian matching
]
```

### Backward compatibility

Builder sử dụng `get_path(cfg, "perception.tracking.min_hits", 3)` với default value —
config cũ không có key `min_hits` vẫn hoạt động với giá trị mặc định 3.

---

## 9. Kiểm thử

### 9.1 Unit tests (tự động)

```bash
# Chạy toàn bộ test suite (P1 tests + P2 tests mới)
PYTHONPATH=src pytest -q tests/

# Chỉ chạy tracking tests
PYTHONPATH=src pytest -q tests/test_tracking.py -v

# Chạy với coverage
PYTHONPATH=src pytest --cov=drivevision.perception.tracking tests/test_tracking.py
```

Expected output:
```
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_single_object_stable_id PASSED
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_two_objects_no_swap PASSED
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_min_hits_filters_ghost PASSED
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_track_survives_occlusion PASSED
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_track_dies_after_max_age PASSED
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_no_detection_no_crash PASSED
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_velocity_converges PASSED
tests/test_tracking.py::TestSimpleTrackerIDConsistency::test_history_length_capped PASSED
tests/test_tracking.py::TestHungarianVsGreedy::test_matching_assigns_correct_track PASSED
9 passed in 0.XX s
```

### 9.2 Visual inspection (thủ công)

```bash
# Chạy pipeline trên video mẫu, lưu output
python scripts/run_pipeline.py \
  --source data/samples/sample.mp4 \
  --config configs/default.yaml \
  --save output/tracked.mp4

# Kiểm tra:
# 1. ID số trên bbox có ổn định không khi vật thể di chuyển?
# 2. Trajectory (đường polyline) có xuất hiện không?
# 3. ID có nhảy bất thường không?
```

### 9.3 Benchmark thời gian (thủ công)

```python
# scripts/benchmark_tracker.py — chạy riêng để đo hiệu năng
import time
import numpy as np
from drivevision.perception.tracking import SimpleTracker
from drivevision.types import BoundingBox, Detection, Frame

tracker = SimpleTracker(min_hits=1)
frame = Frame(index=0, timestamp=0.0, image=np.zeros((480, 640, 3), dtype=np.uint8))

# Giả lập 50 detection
dets = [
    Detection(
        bbox=BoundingBox(i*10, 10, i*10+80, 90),
        class_id=2, class_name="car", confidence=0.9
    )
    for i in range(50)
]

# Warm up
for _ in range(10):
    tracker.update(frame, dets)

# Benchmark
times = []
for _ in range(100):
    t0 = time.perf_counter()
    tracker.update(frame, dets)
    times.append(time.perf_counter() - t0)

print(f"Trung bình: {np.mean(times)*1000:.2f} ms")
print(f"P95:        {np.percentile(times, 95)*1000:.2f} ms")
```

### 9.4 Kiểm tra ByteTracker (nếu có ultralytics)

```bash
# Chạy với backend bytetrack
python scripts/run_pipeline.py \
  --source data/samples/sample.mp4 \
  --config configs/default.yaml \
  --override perception.tracking.backend=bytetrack \
  --save output/bytetracked.mp4

# So sánh visual với output simple tracker
```

---

## 10. Tiêu chí hoàn thành

### Checklist kỹ thuật

- [ ] `SimpleTracker.update()` dùng `scipy.optimize.linear_sum_assignment` (Hungarian)
- [ ] `SimpleTracker` có tham số `min_hits`, filter ghost track đúng cách
- [ ] `SimpleTracker` có tham số `history_len`, cap độ dài `track.history`
- [ ] `SimpleTracker` có method `reset()` và property `all_tracks`, `track_count`
- [ ] Fallback greedy hoạt động khi không có scipy (try/except + log warning)
- [ ] `ByteTracker` class được thêm vào `perception/tracking.py`
- [ ] `ByteTracker` implement đúng interface `Tracker` (return `List[Track]`)
- [ ] `ByteTracker` cập nhật `history` thủ công để `Track.velocity` hoạt động
- [ ] `builder.py` hỗ trợ `perception.tracking.backend: bytetrack`
- [ ] `builder.py` tắt detector riêng khi dùng ByteTracker
- [ ] `configs/default.yaml` có `backend`, `min_hits`
- [ ] `viz/annotator.py` vẽ bbox + label (class + track_id + conf)
- [ ] `viz/annotator.py` vẽ trajectory polyline với màu theo track_id
- [ ] `viz/annotator.py` vẽ HUD (frame index, track count, risk, action)
- [ ] `pyproject.toml` / `requirements.txt` có `scipy>=1.11`

### Checklist test

- [ ] `test_single_object_stable_id` pass
- [ ] `test_two_objects_no_swap` pass
- [ ] `test_min_hits_filters_ghost` pass
- [ ] `test_track_survives_occlusion` pass
- [ ] `test_track_dies_after_max_age` pass
- [ ] `test_no_detection_no_crash` pass
- [ ] `test_velocity_converges` pass
- [ ] `test_history_length_capped` pass
- [ ] `test_matching_assigns_correct_track` pass
- [ ] P1 tests (`test_pipeline.py`) vẫn pass (không regression)

### Checklist visual / demo

- [ ] Chạy `run_pipeline.py` trên video mẫu không crash
- [ ] Track ID hiển thị ổn định ≥ 5 giây trên vật thể di chuyển
- [ ] Trajectory polyline hiện rõ ràng trên frame annotated
- [ ] Màu sắc mỗi track khác nhau và nhất quán theo track_id

---

## 11. Rủi ro & cách xử lý

| Rủi ro | Khả năng | Tác động | Cách xử lý |
|--------|---------|---------|-----------|
| scipy không available trong môi trường | Thấp | Trung bình | Fallback greedy matching + log warning; thêm scipy vào requirements |
| ID switch khi nhiều vật thể chồng lấp | Cao | Thấp (P2 mục tiêu không phải eliminate hoàn toàn) | Chấp nhận; note rõ limitation; ByteTracker giảm thiểu |
| ByteTracker crash trên frame black (test) | Trung bình | Thấp | Kiểm tra `boxes.id is None`; wrap trong try/except; skipif trong pytest |
| `persist=True` ultralytics giữ state sai khi reset video | Trung bình | Trung bình | Document rõ: cần tạo ByteTracker instance mới mỗi video; hoặc gọi `model.predictor.trackers[0].reset()` |
| history grow unbounded trong long video | Thấp | Thấp | `history_len` đã giới hạn; đủ rồi |
| Annotator chậm khi nhiều track + trajectory dài | Thấp | Thấp | Giảm `history_len` hoặc bỏ qua frame cũ trong trajectory nếu cần |
| Circular import giữa annotator và types | Không | Cao | Annotator chỉ import từ `..types` — không import từ perception hay pipeline |
| min_hits quá cao → track hữu ích bị ẩn | Thấp | Thấp | Giá trị 3 frame cân bằng; tunable qua config |

---

## 12. Hiệu năng & tài nguyên

### Ước tính thời gian xử lý (CPU, không GPU)

| Component | Ước tính | Ghi chú |
|-----------|---------|---------|
| `_build_iou_cost_matrix` (50 tracks × 50 dets) | < 1 ms | Nested loop Python, không vectorize |
| `linear_sum_assignment` (50×50 matrix) | < 0.5 ms | Scipy C implementation |
| Cập nhật state (history, spawn, prune) | < 0.5 ms | List operations |
| **Tổng SimpleTracker.update()** | **< 2 ms** | Tốt hơn mục tiêu 5 ms |
| `ByteTracker.update()` (inference + tracking) | ~15-50 ms | Phụ thuộc GPU/CPU, model size |

### Tối ưu tiềm năng (nếu cần — Phase 8/10)

1. **Vector hóa cost matrix:** Thay nested loop bằng numpy broadcasting:
   ```python
   # Chưa implement trong P2, để đây như ghi chú tối ưu
   # IoU vectorized: cần expand bbox thành array rồi dùng broadcast
   ```
2. **Giảm `history_len`:** Từ 30 xuống 15 nếu memory là vấn đề.
3. **Skip tracking mỗi K frame:** Tracker không nhất thiết chạy mỗi frame — có thể
   interpolate cho frame bỏ qua.

### Bộ nhớ

| Cấu trúc | Kích thước | Ghi chú |
|----------|-----------|---------|
| Mỗi Track object | ~1 KB (với history 30 pts) | Rất nhỏ |
| 100 tracks đồng thời | ~100 KB | Không đáng kể |
| `self._histories` trong ByteTracker | ~100 KB | Cùng cỡ |

---

## 13. Sản phẩm bàn giao

### Code files thay đổi

| File | Loại thay đổi | Nội dung |
|------|-------------|---------|
| `src/drivevision/perception/tracking.py` | Sửa + thêm | `SimpleTracker` nâng cấp, `ByteTracker` mới |
| `src/drivevision/viz/annotator.py` | Sửa (fill stub) | Draw bbox, label, trajectory, HUD |
| `src/drivevision/pipeline/builder.py` | Sửa | Support `backend` config, `min_hits` |
| `configs/default.yaml` | Sửa | Thêm `backend`, `min_hits` |
| `pyproject.toml` | Sửa | Thêm `scipy` dependency |
| `tests/test_tracking.py` | Tạo mới | 9+ test cases |

### Files KHÔNG thay đổi

| File | Lý do |
|------|-------|
| `src/drivevision/types.py` | Interface bất biến; `Track`, `BoundingBox.iou()`, `velocity` đã đủ |
| `src/drivevision/perception/base.py` | `Tracker` ABC bất biến |
| `src/drivevision/pipeline/pipeline.py` | Đã gọi `tracker.update(frame, detections)` đúng cách |
| `tests/test_pipeline.py` | Không regression; P1 tests tiếp tục chạy |

### Demo artifacts

```
output/
  tracked.mp4          ← video annotated (SimpleTracker, trajectory visible)
  bytetracked.mp4      ← video annotated (ByteTracker, nếu có ultralytics)
```

---

## 14. Điểm nhấn cho Portfolio

### Điểm kỹ thuật có thể trình bày

1. **"Tôi hiểu tracking là association algorithm, không phải model"**
   — Giải thích rõ trong code comment và README: detector cho biết "cái gì ở đâu", tracker
   hỏi "đây có phải cùng vật thể không?"

2. **"Tôi implement Hungarian matching từ đầu và hiểu tại sao nó tốt hơn greedy"**
   — Có thể trình bày ví dụ cost matrix và giải thích thuật toán tối ưu toàn cục.

3. **"Clean architecture: cả SimpleTracker và ByteTracker cùng interface, swap không cần
   sửa pipeline"**
   — Đây là Open/Closed Principle trong thực tế.

4. **"Tôi biết khi nào dùng công cụ đơn giản (SimpleTracker) và khi nào cần công cụ mạnh
   hơn (ByteTracker)"**
   — Trade-off analysis có chủ đích, không chạy theo complexity không cần thiết.

5. **"History + velocity trong Track nuôi Phase 6 TTC estimation"**
   — Thiết kế dữ liệu có tư duy về downstream consumers.

6. **"Test ID consistency mà không cần video thật"**
   — Unit test với frame giả lập numpy demonstrating testability.

### Metric nâng cao cho portfolio (Phase 8/10)

Nếu muốn báo số cụ thể trong CV/portfolio, Phase 8/10 sẽ đánh giá trên BDD100K tracking:

- **MOTA** (Multiple Object Tracking Accuracy): chỉ số tổng hợp; MOTA = 1 - (FP + FN + ID_sw) / GT
- **IDF1** (ID F1 Score): đo khả năng maintain ID đúng qua thời gian; phù hợp hơn MOTA
  để đánh giá chất lượng tracking
- **HOTA** (Higher Order Tracking Accuracy): metric mới nhất (2020), cân bằng giữa detection
  và association quality

Công cụ đánh giá: `TrackEval` library (https://github.com/JonathonLuiten/TrackEval)

---

## 15. Tham khảo

### Papers

1. **SORT: Simple Online and Realtime Tracking** — Bewley et al., 2016.
   Nền tảng của SimpleTracker; Hungarian + Kalman filter.
   https://arxiv.org/abs/1602.00763

2. **ByteTrack: Multi-Object Tracking by Associating Every Detection Box** — Zhang et al., 2022.
   Tái sử dụng detection confidence thấp trong matching thứ hai.
   https://arxiv.org/abs/2110.06864

3. **HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking** — Luiten et al., 2021.
   Metric đánh giá hiện đại nhất cho MOT.
   https://arxiv.org/abs/2009.07736

### Code references

4. **ultralytics tracking docs** — model.track() API, persist=True, tracker config YAML.
   https://docs.ultralytics.com/modes/track/

5. **scipy.optimize.linear_sum_assignment** — Hungarian algorithm implementation.
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

6. **TrackEval** — Evaluation toolkit cho MOT benchmark.
   https://github.com/JonathonLuiten/TrackEval

7. **BDD100K Tracking benchmark** — Dataset với tracking annotation, dùng cho Phase 8.
   https://www.vis.xyz/bdd100k/

### Tài nguyên học

8. **AI Summer: Introduction to Object Tracking** — Giới thiệu IoU tracker, SORT, DeepSORT
   https://theaisummer.com/object-tracking/

9. **Computer Vision Zone: Object Tracking** — Hướng dẫn thực hành với code
   https://www.computervision.zone/

---

## 16. Checklist tổng kết

Checklist này dành cho review cuối trước khi merge / chuyển sang Phase 3.

### Code quality

- [ ] Tất cả function/class có docstring giải thích mục đích và tham số
- [ ] Comment giải thích khái niệm: association, ID switch, occlusion, min_hits
- [ ] Không có hardcoded magic number (dùng tham số có default)
- [ ] Import guard cho scipy và ultralytics (try/except + log warning)
- [ ] Không để `print()` debug trong production code (dùng `logging`)
- [ ] Type hints đầy đủ cho tất cả public method
- [ ] `tracking.py` không import từ `viz/` hay `api/` (không circular dependency)

### Correctness

- [ ] `_build_iou_cost_matrix` trả matrix đúng chiều `(n_tracks, n_dets)`
- [ ] `_hungarian_match` xử lý đúng khi n_tracks = 0 hoặc n_dets = 0
- [ ] Match bị lọc khi IoU < iou_threshold (không phải cost < threshold)
- [ ] Track chỉ xuất hiện trong output khi `age >= min_hits AND time_since_update == 0`
- [ ] `time_since_update` tăng đúng cho unmatched tracks, reset về 0 khi matched
- [ ] `track.history` không bao giờ vượt `history_len` entries
- [ ] `ByteTracker` xử lý `boxes.id is None` không crash

### Integration

- [ ] `PYTHONPATH=src pytest -q tests/` — toàn bộ test pass (kể cả P1 tests)
- [ ] `python scripts/run_pipeline.py --source <video>` chạy không crash
- [ ] Config `backend: bytetrack` hoạt động khi có ultralytics
- [ ] Config `backend: simple` (default) hoạt động không cần ultralytics
- [ ] Output video có track ID và trajectory visible

### Documentation

- [ ] Comment trong code giải thích tại sao Hungarian tốt hơn greedy
- [ ] Comment giải thích `min_hits` và tác dụng tránh ghost track
- [ ] `configs/default.yaml` có comment mô tả từng tham số mới
- [ ] Phase 6 connection được note rõ: "history.velocity → TTC estimation"

### Handoff sang Phase 3

- [ ] `SceneState.tracks` được điền đúng trong `pipeline.py` (đã có)
- [ ] `Track.velocity` trả giá trị hợp lý (không phải luôn `(0, 0)`)
- [ ] `Track.history` đủ dài (≥ 2 entries) cho Phase 6 dùng bbox area qua thời gian
- [ ] Không có breaking change nào với interface `Tracker`, `Frame`, `PipelineResult`
