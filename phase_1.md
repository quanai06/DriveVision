# Phase 1 — Object Detection (Pretrained) + Visualization + Output Pipeline

**Mục tiêu:** Cắm YOLOv8 pretrained vào pipeline để có demo chạy được đầu tiên — đọc video, phát hiện đa lớp đối tượng đường phố, vẽ annotation lên từng frame, xuất video có nhãn và đo FPS; đây là nền để Phase 2 (tracking) gắn thêm mà không thay đổi interface.**

---

## 1. Tổng quan & vị trí trong lộ trình

| Giai đoạn | Tên | Phụ thuộc vào | Cung cấp cho |
|---|---|---|---|
| P0 | Nền tảng (xong) | — | P1 |
| **P1** | **Object Detection + Viz + Output** | **P0** | **P2 (Tracking)** |
| P2 | Multi-Object Tracking | P1 | P3 |
| P3 | Lane Detection | P1 | P5 |
| P4 | Traffic Light + State | P1 | P5 |
| P5 | Scene Understanding | P2, P3, P4 | P6 |
| P6 | Risk + Decision | P5 | P7 |
| P7 | API + Dashboard | P6 | P8-P10 |

### Vai trò của Phase 1

Phase 1 là "cột mốc có thể demo được đầu tiên". Sau khi hoàn thành, bất kỳ ai chạy:

```bash
python scripts/run_pipeline.py --source data/samples/sample.mp4 --save output/demo.mp4
```

sẽ thấy video đầu ra có bounding box màu, nhãn tên lớp, thanh confidence, HUD FPS — tất cả chạy được mà không cần train gì thêm.

**Kiến trúc 2 mặt phẳng** ở Phase 1 chỉ dùng **RUNTIME PLANE** (local):

```
[VideoSource] --> [YOLODetector] --> [SceneBuilder] --> [Annotator] --> [VideoWriter / cv2.imshow]
```

RESEARCH PLANE (Kaggle) chưa liên quan ở phase này vì ta dùng pretrained weights tải tự động.

---

## 2. Mục tiêu (đo được)

| # | Mục tiêu | Cách đo | Ngưỡng chấp nhận |
|---|---|---|---|
| M1 | Pipeline chạy end-to-end trên video mẫu | `python scripts/run_pipeline.py --source <video>` không crash | Pass (0 exception) |
| M2 | Phát hiện ít nhất 6 class COCO liên quan giao thông | Đếm class xuất hiện trên video mẫu | car, person, truck, bus, motorcycle, bicycle |
| M3 | Xuất video annotated có bounding box + nhãn + confidence | Mở file output bằng trình phát | File tồn tại, xem được |
| M4 | FPS đo được và in ra stdout | Log cuối pipeline | Số dương > 0 |
| M5 | Xử lý graceful khi thiếu `ultralytics` | `pip uninstall ultralytics && python scripts/run_pipeline.py` | Cảnh báo log, không crash, pipeline tiếp tục với detector=None |
| M6 | Test suite pass | `PYTHONPATH=src pytest -q` | 0 failed |
| M7 | Chế độ `--display` hiển thị cửa sổ cv2 | Chạy tay trên máy có màn hình | Cửa sổ hiện, 'q' đóng được |

---

## 3. Phạm vi

### Trong phạm vi Phase 1

- Hoàn thiện `viz/annotator.py`: vẽ bounding box, nhãn, confidence, HUD (FPS + số detection)
- Hoàn thiện `cli.py`: tích hợp Annotator, VideoWriter, cv2.imshow, FPS counter, thêm CLI args
- Xác nhận `perception/detection.py` (đã có impl) hoạt động end-to-end
- Thêm CLI args: `--save`, `--display`, `--no-display`, `--max-frames`, `--conf`
- Cập nhật `configs/default.yaml`: chỉnh `output.*` keys
- Viết tests cho Annotator (unit) và pipeline smoke test với mock detector
- Hướng dẫn tải video mẫu BDD100K / KITTI để test
- Đo và log FPS (wall-clock frames/sec)
- **Một model YOLO phát hiện đồng thời nhiều class** — không cần model riêng cho mỗi loại vật thể

### Ngoài phạm vi Phase 1 (giữ nguyên stub)

- Tracking / track ID (Phase 2)
- Lane detection (Phase 3)
- Traffic light state (Phase 4)
- Risk / Decision logic (Phase 6)
- FastAPI / Dashboard (Phase 7)
- Fine-tuning trên BDD100K/KITTI (Phase 8)
- CARLA source (Phase 9)
- Depth / semantic channels

---

## 4. Điều kiện tiên quyết

### 4.1 Môi trường

- Python 3.12 (đúng với ràng buộc dự án)
- Git repo tại `/home/quan/DriveVision`
- Virtual environment kích hoạt (`.venv/`)
- Package `drivevision` đã cài editable (P0):
  ```bash
  pip install -e ".[dev]"
  ```

### 4.2 Kiểm tra P0 đã hoàn chỉnh

```bash
# Phải pass tất cả
PYTHONPATH=src pytest tests/ -q

# Phải import được
python -c "from drivevision.types import Frame, Detection, BoundingBox; print('OK')"
python -c "from drivevision.pipeline.pipeline import Pipeline; print('OK')"
python -c "from drivevision.config import load_config; print(load_config()['source']['type'])"
```

### 4.3 Video mẫu để test

Phase 1 cần ít nhất 1 video giao thông để kiểm thử end-to-end. Dùng một trong các cách sau:

**Cách A — BDD100K mini clip (khuyến nghị):**
```bash
# Tải 1 clip ngắn từ BDD100K public sample (không cần đăng ký)
mkdir -p /home/quan/DriveVision/data/samples
wget -O /home/quan/DriveVision/data/samples/sample.mp4 \
  "https://dl.yf.io/bdd-data/bdd100k/video/train/b1c66a42-6f7d74b4.mp4"
```

**Cách B — Tự tạo bằng ffmpeg từ ảnh tĩnh (offline):**
```bash
# Nếu không có mạng, dùng ffmpeg tạo video thử nghiệm 5s từ màu solid
ffmpeg -f lavfi -i "color=c=blue:size=1280x720:rate=30" \
  -t 5 /home/quan/DriveVision/data/samples/synthetic.mp4
```

**Cách C — Webcam (nếu có):**
```bash
# Đổi source.type = webcam trong config, hoặc:
python scripts/run_pipeline.py --source 0
```

**Cách D — Clip YouTube qua yt-dlp:**
```bash
pip install yt-dlp
yt-dlp -f "mp4[height<=720]" -o data/samples/sample.mp4 \
  "https://www.youtube.com/watch?v=<driving-video-id>"
```

---

## 5. Công nghệ & thư viện

| Thư viện | Version | Lý do chọn | Ghi chú cài đặt |
|---|---|---|---|
| `ultralytics` | `>=8.2` | YOLO v8/v10/v11 API thống nhất, tự tải pretrained, active maintained | `pip install "ultralytics>=8.2"` |
| `opencv-python` | `>=4.8` | Đọc/ghi video, vẽ annotation, cửa sổ hiển thị | Đã có trong base deps P0 |
| `numpy` | `>=1.24` | Mảng ảnh BGR | Đã có |
| `pyyaml` | `>=6.0` | Load config YAML | Đã có |
| `pytest` | `>=8.0` | Test suite | Đã có trong `[dev]` |

### Cài đặt perception extras

```bash
# Từ thư mục /home/quan/DriveVision
pip install -e ".[perception,dev]"
# hoặc thủ công:
pip install "ultralytics>=8.2"
```

### Lưu ý về ultralytics

- Lần đầu chạy `YOLO("yolov8n.pt")`, ultralytics **tự tải** weights (~6 MB) từ GitHub Releases về `~/.config/Ultralytics/` hoặc thư mục cache.
- Nếu không có mạng: tải trước bằng `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` ở nơi có internet, sau đó copy file `.pt` vào `models/weights/yolo.pt`.
- Trên Kaggle: ultralytics được cài sẵn; file `.pt` sinh ra từ fine-tuning sẽ được copy vào `models/weights/` để RUNTIME PLANE sử dụng.

### Lựa chọn model

| Model | Size | mAP COCO | Tốc độ (GPU) | Dùng khi |
|---|---|---|---|---|
| `yolov8n.pt` | 6 MB | 37.3 | Nhanh nhất | Default demo, Kaggle-free GPU |
| `yolov8s.pt` | 22 MB | 44.9 | Khá | Accuracy tốt hơn |
| `yolov8m.pt` | 50 MB | 50.2 | Vừa | Baseline fine-tune (Phase 8) |

**Quyết định Phase 1:** dùng `yolov8n.pt` làm default fallback.

---

## 6. Thiết kế chi tiết

### 6.1 Các types liên quan đến Phase 1

Từ `src/drivevision/types.py` (KHÔNG thay đổi, chỉ dùng):

```python
# Input vào pipeline
Frame(index, timestamp, image: np.ndarray[H,W,3 BGR], depth=None, semantic=None)
  .width -> int
  .height -> int

# Output của YOLODetector
Detection(
    bbox: BoundingBox(x1, y1, x2, y2),
    class_id: int,
    class_name: str,
    confidence: float,
)
BoundingBox.as_int() -> (int, int, int, int)  # dùng để vẽ cv2.rectangle
BoundingBox.center -> (float, float)
BoundingBox.width, .height

# Đầu ra pipeline cho Annotator
PipelineResult(
    frame: Frame,
    scene: SceneState(frame_index, timestamp, detections, tracks, lanes, traffic_lights),
    risk: None,      # Phase 6
    decision: None,  # Phase 6
)
```

### 6.2 Luồng dữ liệu tổng thể (ASCII)

```
                         RUNTIME PLANE — Phase 1
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  configs/default.yaml ──► load_config() ──► cfg: dict          │
│         │                                        │             │
│         ▼                                        ▼             │
│  build_source(cfg)                    build_pipeline(cfg)      │
│         │                                        │             │
│         ▼                                        ▼             │
│  VideoSource(path, loop, max_frames)    Pipeline(              │
│         │                                 detector=YOLODetector│
│         │  Frame(index, ts, image)        tracker=None,        │
│         ├──────────────────────────────►  ...other=None        │
│         │                              )                       │
│         │                                        │             │
│         │            PipelineResult              │             │
│         │◄───────────────────────────────────────┤             │
│         │                                        │             │
│         ▼                                        │             │
│  Annotator.draw(result) → annotated_frame (BGR)  │             │
│         │                                        │             │
│         ├──► cv2.VideoWriter.write(annotated)    │             │
│         │    (nếu output.save_path != null)       │             │
│         │                                        │             │
│         └──► cv2.imshow("DriveVision", annotated)│             │
│              (nếu output.display == true)        │             │
│                                                  │             │
│  FPS counter: time.perf_counter() per frame      │             │
│  Log: "Frame 30 | 5 dets | FPS: 12.4"           │             │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Luồng trong Pipeline.process() (Phase 1 chỉ dùng detection)

```
Pipeline.process(frame)
  ├── detector.detect(frame)     → List[Detection]    ← YOLODetector
  ├── tracker.update(...)        → []                 ← tracker=None, bỏ qua
  ├── lane_detector.detect(...)  → []                 ← lane_detector=None
  ├── traffic_light_detector...  → []                 ← tl_detector=None
  ├── scene_builder.build(...)   → SceneState
  ├── risk_assessor.assess(...)  → None               ← risk=None
  └── decision_support.decide(...) → None             ← decision=None
  return PipelineResult(frame, scene, risk=None, decision=None)
```

### 6.4 Annotator — thiết kế rendering

**Lớp vẽ (thứ tự từ dưới lên):**

```
Layer 0: frame.image.copy()           (ảnh gốc BGR)
Layer 1: Bounding boxes (cv2.rectangle)
Layer 2: Nhãn (class_name + confidence) trên nền mờ
Layer 3: HUD overlay (góc trên-trái): FPS, số detection
```

**Bảng màu theo class:**

Mỗi class_id ánh xạ sang màu BGR nhất quán. Dùng hash để không cần hardcode toàn bộ 80 class COCO:

```python
def _class_color(class_id: int) -> tuple[int, int, int]:
    # Tạo màu BGR bão hòa cao từ class_id, tái lặp qua 12 màu phân biệt
    palette = [
        (0, 0, 255),    # đỏ     — car
        (0, 255, 0),    # xanh lá — person
        (255, 128, 0),  # xanh dương nhạt — truck
        (0, 255, 255),  # vàng — bus
        (255, 0, 255),  # tím — bicycle
        (0, 128, 255),  # cam — motorcycle
        (128, 255, 0),  # xanh lam — traffic light
        (255, 0, 128),  # hồng — stop sign
        (0, 128, 128),  # nâu — ...
        (128, 0, 255),  # tím đậm
        (255, 255, 0),  # xanh cyan
        (128, 128, 0),  # xanh ô liu
    ]
    return palette[class_id % len(palette)]
```

**Định dạng nhãn:**

```
"car 0.87"
"person 0.92"
```

Vẽ background tối mờ phía sau chữ bằng `cv2.rectangle` rồi `cv2.putText`.

**HUD (góc trên trái):**

```
┌─────────────────────────────┐
│ DriveVision  FPS: 14.2      │
│ Frame: 128  Dets: 6         │
└─────────────────────────────┘
```

### 6.5 Quyết định thiết kế

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Annotator có lưu state FPS không? | Có, `Annotator` nhận FPS từ ngoài vào `draw(result, fps=0.0)` | FPS được tính ở cli.py, Annotator không cần biết thời gian |
| Màu bounding box | Phân biệt theo class_id | Nhìn đẹp hơn màu cố định |
| VideoWriter codec | `mp4v` (nén MP4) | Hỗ trợ rộng trên Linux/Mac, không cần cài thêm |
| Chế độ display mặc định | `False` | Chạy headless an toàn (môi trường không có màn hình) |
| Thứ tự tham số `draw()` | `draw(result, fps=0.0)` | Backward-compatible với Phase 2 (chỉ thêm track IDs mà không đổi chữ ký) |
| YOLODetector lazy import | Giữ nguyên pattern hiện có | Không crash khi import mà không có ultralytics |
| Một model cho nhiều class | Dùng `classes=None` (mặc định) → YOLO detect 80 class cùng lúc | Không cần multi-model, đơn giản hơn |

---

## 7. Công việc chi tiết

### Task 1.1 — Hoàn thiện `viz/annotator.py`

**Mục đích:** Biến stub thành module rendering đầy đủ chức năng: bounding box, nhãn, HUD, confidence bar.

**File:** `src/drivevision/viz/annotator.py`

**Các bước:**
1. Import `cv2` và các types cần thiết
2. Định nghĩa bảng màu 12 màu BGR
3. Thêm helper `_class_color(class_id)` → tuple BGR
4. Thêm helper `_draw_label(img, text, x, y, color)` vẽ text trên nền mờ
5. Thêm helper `_draw_hud(img, fps, n_dets, frame_index)` ở góc trên-trái
6. Implement `draw(result, fps=0.0)` → gọi các helpers

**Skeleton code đầy đủ:**

```python
"""Render pipeline results onto a BGR frame.

Thay đổi so với stub:
- draw() nhận thêm tham số fps: float (tính từ cli.py)
- Vẽ bounding box màu theo class, nhãn trên nền mờ, HUD FPS
- Trả về np.ndarray BGR cùng kích thước với frame.image
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import Detection, PipelineResult

# 12 màu BGR phân biệt cao
_PALETTE: list[tuple[int, int, int]] = [
    (0,   0,   255),   # 0  đỏ
    (0,   255,  0),    # 1  xanh lá
    (255, 128,  0),    # 2  xanh biển nhạt
    (0,   255, 255),   # 3  vàng
    (255,  0,  255),   # 4  tím
    (0,   128, 255),   # 5  cam
    (128, 255,  0),    # 6  xanh lam-vàng
    (255,  0,  128),   # 7  hồng
    (0,   128, 128),   # 8  nâu xanh
    (128,  0,  255),   # 9  tím đậm
    (255, 255,  0),    # 10 cyan
    (128, 128,  0),    # 11 olive
]

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 2
_BOX_THICKNESS = 2


def _class_color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _draw_label(
    img: np.ndarray,
    text: str,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    """Vẽ text nhãn có background tối phía trên bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    # nếu nhãn bị ra ngoài trên cùng, vẽ xuống phía dưới
    ty = y1 - 4 if y1 - th - 4 >= 0 else y1 + th + 4

    # background rectangle (nền tối bán trong suốt)
    pad = 3
    cv2.rectangle(
        img,
        (x1, ty - th - pad),
        (x1 + tw + pad * 2, ty + pad),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(img, text, (x1 + pad, ty), _FONT, _FONT_SCALE, color, _THICKNESS, cv2.LINE_AA)


def _draw_hud(
    img: np.ndarray,
    fps: float,
    n_dets: int,
    frame_index: int,
) -> None:
    """HUD góc trên-trái: FPS, số detection, frame index."""
    lines = [
        f"DriveVision  FPS: {fps:.1f}",
        f"Frame: {frame_index:>6d}  Dets: {n_dets}",
    ]
    x, y = 10, 28
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.65, 2)
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, line, (x, y), _FONT, 0.65, (0, 255, 180), 2, cv2.LINE_AA)
        y += th + 10


class Annotator:
    """Vẽ kết quả pipeline lên frame BGR và trả về ảnh đã annotate."""

    def draw(self, result: PipelineResult, fps: float = 0.0) -> np.ndarray:
        """
        Tham số
        -------
        result : PipelineResult
            Kết quả từ Pipeline.process(frame).
        fps : float
            FPS hiện tại (tính từ cli.py bằng time.perf_counter).

        Trả về
        ------
        np.ndarray
            Ảnh BGR cùng shape với result.frame.image, đã vẽ annotations.
        """
        img = result.frame.image.copy()

        # --- Vẽ detections ---
        for det in result.scene.detections:
            x1, y1, x2, y2 = det.bbox.as_int()
            color = _class_color(det.class_id)
            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)
            # Nhãn: "car 0.87"
            label = f"{det.class_name} {det.confidence:.2f}"
            _draw_label(img, label, x1, y1, color)

        # --- HUD ---
        _draw_hud(img, fps, len(result.scene.detections), result.frame.index)

        return img
```

**Lưu ý & edge cases:**
- Nếu `result.scene.detections` rỗng, vẫn vẽ HUD (FPS=0, Dets=0) — không crash
- Nếu ảnh rất nhỏ (< 100px), `_draw_hud` vẫn chạy nhưng text có thể bị cắt — không xử lý đặc biệt ở Phase 1
- `result.scene.tracks` tồn tại nhưng rỗng ở Phase 1 (tracker=None) — không vẽ track ID, Phase 2 sẽ bổ sung
- Không in lại ảnh gốc (`img = result.frame.image.copy()` bảo toàn frame gốc)
- `cv2.FONT_HERSHEY_SIMPLEX` là font built-in, không cần cài font ngoài

---

### Task 1.2 — Hoàn thiện `cli.py` — tích hợp Annotator + VideoWriter + FPS

**Mục đích:** `cli.py` là điểm vào chính (entry point). Phase 1 hoàn thiện vòng lặp xử lý frame: gọi Annotator, lưu video, hiển thị, đo FPS, xử lý tín hiệu ngắt.

**File:** `src/drivevision/cli.py`

**Các bước:**
1. Thêm CLI args: `--save`, `--display`, `--no-display`, `--max-frames`, `--conf`
2. Khởi tạo `Annotator`
3. Khởi tạo `cv2.VideoWriter` nếu `save_path` được cấu hình
4. Vòng lặp frame: gọi `pipeline.process()`, `annotator.draw()`, write/show
5. FPS counter: dùng `time.perf_counter()` + sliding window 30 frame
6. In tóm tắt cuối cùng
7. Đảm bảo `VideoWriter.release()` được gọi trong `finally`

**Skeleton code đầy đủ:**

```python
"""Command-line entry point — Phase 1 hoàn chỉnh.

Thay đổi so với stub:
- Thêm args: --save, --display/--no-display, --max-frames, --conf
- Tích hợp Annotator, VideoWriter, FPS counter
- Xử lý KeyboardInterrupt và đảm bảo giải phóng tài nguyên
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

from .config import load_config
from .pipeline.builder import build_pipeline, build_source
from .viz.annotator import Annotator


def _make_video_writer(
    path: str,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Tạo VideoWriter MP4 tại đường dẫn chỉ định."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Không thể tạo VideoWriter tại: {path}")
    return writer


class _FPSCounter:
    """Tính FPS bằng sliding window."""

    def __init__(self, window: int = 30) -> None:
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DriveVision — Autonomous Driving Perception Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--source", help="Video file path hoặc webcam index (ghi đè config)")
    parser.add_argument("--save", dest="save_path", help="Đường dẫn lưu video annotated (ghi đè config)")
    parser.add_argument("--display", action="store_true", default=None, help="Hiển thị cửa sổ cv2")
    parser.add_argument("--no-display", dest="display", action="store_false", help="Tắt cửa sổ cv2")
    parser.add_argument("--max-frames", type=int, help="Dừng sau N frames (ghi đè config)")
    parser.add_argument("--conf", type=float, help="Confidence threshold cho YOLO (ghi đè config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("drivevision")

    # --- Load config và áp dụng overrides từ CLI ---
    cfg = load_config(args.config)

    if args.source is not None:
        try:
            cfg["source"]["path"] = int(args.source)  # webcam index
        except ValueError:
            cfg["source"]["path"] = args.source

    if args.save_path is not None:
        cfg["output"]["save_path"] = args.save_path

    if args.display is not None:
        cfg["output"]["display"] = args.display

    if args.max_frames is not None:
        cfg["source"]["max_frames"] = args.max_frames

    if args.conf is not None:
        cfg["perception"]["detection"]["conf"] = args.conf

    # --- Khởi tạo components ---
    source   = build_source(cfg)
    pipeline = build_pipeline(cfg)
    annotator = Annotator()
    fps_counter = _FPSCounter(window=30)

    save_path: Optional[str] = cfg["output"].get("save_path")
    do_display: bool         = cfg["output"].get("display", False)
    output_fps: float        = float(cfg["output"].get("fps", 30))

    writer: Optional[cv2.VideoWriter] = None
    writer_initialized = False

    log.info("Pipeline sẵn sàng. Detector: %s | Save: %s | Display: %s",
             "YOLOv8" if pipeline.detector else "None (ultralytics thiếu)",
             save_path or "không lưu",
             do_display)

    n_frames = 0
    n_dets_total = 0
    t_start = time.perf_counter()

    try:
        with source as src:
            for frame in src.frames():
                # --- Inference ---
                result = pipeline.process(frame)
                fps_counter.tick()
                current_fps = fps_counter.fps

                # --- Annotate ---
                annotated = annotator.draw(result, fps=current_fps)

                # --- Khởi tạo VideoWriter lần đầu (biết kích thước sau khi có frame) ---
                if save_path and not writer_initialized:
                    h, w = annotated.shape[:2]
                    writer = _make_video_writer(save_path, w, h, output_fps)
                    writer_initialized = True
                    log.info("VideoWriter khởi tạo: %s (%dx%d @ %.0f fps)", save_path, w, h, output_fps)

                # --- Lưu frame ---
                if writer is not None:
                    writer.write(annotated)

                # --- Hiển thị ---
                if do_display:
                    cv2.imshow("DriveVision", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' hoặc ESC
                        log.info("Người dùng nhấn '%s', dừng.", chr(key) if key != 27 else "ESC")
                        break

                # --- Log định kỳ ---
                n_frames += 1
                n_dets = len(result.scene.detections)
                n_dets_total += n_dets
                if n_frames % 30 == 0:
                    log.info(
                        "Frame %5d | Dets: %2d | FPS: %5.1f",
                        frame.index, n_dets, current_fps,
                    )

    except KeyboardInterrupt:
        log.info("Dừng bởi Ctrl+C.")
    finally:
        # --- Giải phóng tài nguyên ---
        if writer is not None:
            writer.release()
            log.info("VideoWriter đã giải phóng: %s", save_path)
        if do_display:
            cv2.destroyAllWindows()

    # --- Tóm tắt ---
    t_elapsed = time.perf_counter() - t_start
    avg_fps = n_frames / t_elapsed if t_elapsed > 0 else 0.0
    log.info(
        "Hoàn thành: %d frames | %d detections tổng | FPS trung bình: %.1f | Thời gian: %.1fs",
        n_frames, n_dets_total, avg_fps, t_elapsed,
    )


if __name__ == "__main__":
    main()
```

**Lưu ý & edge cases:**
- `VideoWriter` khởi tạo LAZY sau khi biết kích thước frame đầu tiên (video có thể là bất kỳ độ phân giải nào)
- `src.frames()` dùng context manager `with source as src:` — `VideoSource` đã implement `__enter__`/`__exit__`; nếu chưa, dùng `try/finally` với `source.close()`
- Nếu `save_path` là `null` trong config và không truyền `--save`, không tạo VideoWriter
- Nếu source là webcam index (int), `args.source` là chuỗi "0" → cast thành `int`
- Khi `display=False` và `save_path=None`, pipeline vẫn chạy hữu ích (benchmark FPS)
- `cv2.waitKey(1)` — blocking 1ms; nếu `waitKey(0)` sẽ pause mỗi frame

---

### Task 1.3 — Xác nhận `perception/detection.py` và thêm CLI arg `--conf`

**Mục đích:** Đảm bảo `YOLODetector` hoạt động đúng end-to-end; xác minh fallback `yolov8n.pt`.

**File:** `src/drivevision/perception/detection.py` (không thay đổi logic, chỉ xem lại)

**Checklist xác nhận:**

```
[x] __init__ lazy import ultralytics — nếu ImportError thì raise rõ ràng
[x] Kiểm tra Path(weights).exists() — nếu không có thì dùng _PRETRAINED_FALLBACK
[x] model.predict() truyền đúng conf, iou, classes, imgsz, verbose=False
[x] Duyệt res.boxes lấy xyxy, cls, conf
[x] Trả về List[Detection] đúng type
```

**Test thủ công nhanh:**

```bash
cd /home/quan/DriveVision
PYTHONPATH=src python - <<'EOF'
import numpy as np
from drivevision.types import Frame
from drivevision.perception.detection import YOLODetector

det = YOLODetector(conf=0.3)
frame = Frame(0, 0.0, np.zeros((640, 640, 3), dtype=np.uint8))
dets = det.detect(frame)
print(f"OK — {len(dets)} detections trên ảnh đen (mong đợi: 0)")
EOF
```

**Lưu ý:** Ảnh đen không có object → `len(dets) == 0` là đúng. Dùng video thật để kiểm tra phát hiện.

---

### Task 1.4 — Cập nhật `configs/default.yaml`

**Mục đích:** Làm rõ các key `output.*` và thêm key mới để hỗ trợ Phase 1.

**File:** `configs/default.yaml`

**Thay đổi cần thực hiện:**

Thêm / chỉnh sửa section `output`:

```yaml
output:
  display: false         # true = hiển thị cửa sổ cv2 (cần màn hình)
  save_path: null        # null = không lưu; ví dụ: "output/demo.mp4"
  fps: 30                # FPS khi ghi video output (không liên quan đến FPS inference)
  window_name: "DriveVision"  # tên cửa sổ cv2.imshow (mới)

# Logging
logging:
  level: INFO            # DEBUG | INFO | WARNING | ERROR
  log_interval: 30       # in log mỗi N frames (mới)
```

Thêm key `perception.detection.device` để hỗ trợ sau:

```yaml
perception:
  detection:
    enabled: true
    weights: models/weights/yolo.pt
    conf: 0.35
    iou: 0.5
    classes: null          # null = tất cả 80 class COCO
    imgsz: 640
    device: null           # null = auto (CUDA nếu có, sinon CPU); "cpu" | "cuda:0"
```

**Sau thay đổi, cần cập nhật `DEFAULT_CONFIG` trong `config.py` để đồng bộ:**

```python
DEFAULT_CONFIG: Dict[str, Any] = {
    ...
    "output": {
        "display": False,
        "save_path": None,
        "fps": 30,
        "window_name": "DriveVision",
    },
    "logging": {
        "level": "INFO",
        "log_interval": 30,
    },
}
```

---

### Task 1.5 — Cập nhật `pipeline/builder.py` — truyền `device` vào YOLODetector

**Mục đích:** Cho phép chỉ định device (cpu/cuda) qua config.

**File:** `src/drivevision/pipeline/builder.py`

**Thay đổi nhỏ trong `build_pipeline`:**

```python
# Trong block if get_path(cfg, "perception.detection.enabled"):
detector = YOLODetector(
    weights=get_path(cfg, "perception.detection.weights", "models/weights/yolo.pt"),
    conf=get_path(cfg, "perception.detection.conf", 0.35),
    iou=get_path(cfg, "perception.detection.iou", 0.5),
    classes=get_path(cfg, "perception.detection.classes"),
    imgsz=get_path(cfg, "perception.detection.imgsz", 640),
    # device=get_path(cfg, "perception.detection.device"),  # Phase 1 chưa cần, để comment
)
```

Không thay đổi gì thêm — `YOLODetector.__init__` hiện tại không nhận `device` nhưng ultralytics tự chọn CUDA nếu có.

---

### Task 1.6 — Viết tests

**Mục đích:** Đảm bảo `Annotator`, `YOLODetector` (với mock), và pipeline end-to-end hoạt động đúng.

**File:** `tests/test_phase1.py` (tạo mới)

**Toàn bộ nội dung test:**

```python
"""Tests cho Phase 1: Object Detection + Visualization + Output pipeline.

Chạy: PYTHONPATH=src pytest tests/test_phase1.py -v
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
    """Frame 640x480 màu đen."""
    return Frame(index=42, timestamp=1.4, image=np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.fixture
def sample_detections() -> list[Detection]:
    """Một số Detection giả lập."""
    return [
        Detection(BoundingBox(10, 20, 200, 150), class_id=2, class_name="car",    confidence=0.87),
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


# ─────────────────────────── Test BoundingBox ────────────────────

class TestBoundingBox:
    def test_as_int(self):
        bb = BoundingBox(10.7, 20.3, 200.9, 150.1)
        assert bb.as_int() == (10, 20, 200, 150)

    def test_width_height(self):
        bb = BoundingBox(0, 0, 100, 50)
        assert bb.width == 100
        assert bb.height == 50

    def test_area(self):
        bb = BoundingBox(0, 0, 10, 10)
        assert bb.area == 100.0

    def test_center(self):
        bb = BoundingBox(0, 0, 100, 60)
        assert bb.center == (50.0, 30.0)

    def test_iou_identical(self):
        bb = BoundingBox(0, 0, 10, 10)
        assert abs(bb.iou(bb) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        assert bb.iou(b) == 0.0  # type: ignore


# ─────────────────────────── Test Annotator ──────────────────────

class TestAnnotator:
    def test_draw_returns_same_shape(self, pipeline_result, dummy_frame):
        from drivevision.viz.annotator import Annotator

        ann = Annotator()
        out = ann.draw(pipeline_result, fps=15.0)

        assert isinstance(out, np.ndarray)
        assert out.shape == dummy_frame.image.shape
        assert out.dtype == np.uint8

    def test_draw_does_not_modify_original(self, pipeline_result):
        from drivevision.viz.annotator import Annotator

        original = pipeline_result.frame.image.copy()
        ann = Annotator()
        ann.draw(pipeline_result, fps=0.0)
        assert np.array_equal(pipeline_result.frame.image, original), \
            "Annotator không được sửa ảnh gốc"

    def test_draw_empty_detections(self, dummy_frame):
        from drivevision.viz.annotator import Annotator

        scene = SceneState(frame_index=0, timestamp=0.0)
        result = PipelineResult(frame=dummy_frame, scene=scene)
        ann = Annotator()
        out = ann.draw(result, fps=30.0)
        # Không crash, shape đúng
        assert out.shape == dummy_frame.image.shape

    def test_draw_many_detections(self, dummy_frame):
        """Stress test: 20 detections không crash."""
        from drivevision.viz.annotator import Annotator

        dets = [
            Detection(BoundingBox(i * 10, i * 5, i * 10 + 80, i * 5 + 60),
                      class_id=i % 12, class_name=f"obj{i}", confidence=0.5 + i * 0.02)
            for i in range(20)
        ]
        scene = SceneState(frame_index=0, timestamp=0.0, detections=dets)
        result = PipelineResult(frame=dummy_frame, scene=scene)
        ann = Annotator()
        out = ann.draw(result)
        assert out.shape == dummy_frame.image.shape

    def test_draw_fps_zero(self, pipeline_result):
        """fps=0 không gây division by zero hay exception."""
        from drivevision.viz.annotator import Annotator

        ann = Annotator()
        out = ann.draw(pipeline_result, fps=0.0)
        assert out is not None

    def test_draw_returns_new_array(self, pipeline_result):
        """Trả về bản copy, không phải reference đến ảnh gốc."""
        from drivevision.viz.annotator import Annotator

        ann = Annotator()
        out = ann.draw(pipeline_result)
        assert out is not pipeline_result.frame.image


# ─────────────────────────── Test YOLODetector (mock) ────────────

class TestYOLODetectorMock:
    """Test logic xử lý kết quả YOLO — mock YOLO model để không cần GPU."""

    def test_detect_empty_results(self, dummy_frame):
        """Nếu YOLO trả về kết quả rỗng, detect() phải trả về []."""
        pytest.importorskip("ultralytics")  # skip nếu không cài

        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.names = {}

        with patch("ultralytics.YOLO") as MockYOLO:
            MockYOLO.return_value.predict.return_value = [mock_result]
            from drivevision.perception.detection import YOLODetector

            det = YOLODetector.__new__(YOLODetector)
            det.model = MockYOLO.return_value
            det.conf = 0.35
            det.iou = 0.5
            det.classes = None
            det.imgsz = 640

            results = det.detect(dummy_frame)
            assert results == []

    def test_detect_parses_boxes(self, dummy_frame):
        """detect() phải parse xyxy, cls, conf thành Detection đúng."""
        pytest.importorskip("ultralytics")

        import torch
        from unittest.mock import MagicMock

        from drivevision.perception.detection import YOLODetector

        # Tạo mock box
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 20.0, 200.0, 150.0])]
        mock_box.cls  = [torch.tensor(2.0)]
        mock_box.conf = [torch.tensor(0.87)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {2: "car"}

        det = YOLODetector.__new__(YOLODetector)
        det.conf = 0.35
        det.iou = 0.5
        det.classes = None
        det.imgsz = 640
        det.model = MagicMock()
        det.model.predict.return_value = [mock_result]

        dets = det.detect(dummy_frame)
        assert len(dets) == 1
        assert dets[0].class_name == "car"
        assert abs(dets[0].confidence - 0.87) < 0.001
        assert dets[0].bbox.as_int() == (10, 20, 200, 150)


# ─────────────────────────── Test Pipeline end-to-end ────────────

class TestPipelinePhase1:
    def test_pipeline_no_detector(self, dummy_frame):
        """Pipeline không có detector vẫn chạy, detections rỗng."""
        from drivevision.pipeline.pipeline import Pipeline

        result = Pipeline().process(dummy_frame)
        assert result.scene.detections == []
        assert result.scene.tracks == []
        assert result.risk is None
        assert result.decision is None

    def test_pipeline_with_mock_detector(self, dummy_frame, sample_detections):
        """Pipeline với mock detector trả đúng detections."""
        from unittest.mock import MagicMock

        from drivevision.pipeline.pipeline import Pipeline
        from drivevision.perception.base import Detector

        class MockDetector(Detector):
            def detect(self, frame):
                return sample_detections

        result = Pipeline(detector=MockDetector()).process(dummy_frame)
        assert len(result.scene.detections) == 3
        assert result.scene.detections[0].class_name == "car"

    def test_annotator_integrates_with_pipeline(self, dummy_frame, sample_detections):
        """Annotator nhận PipelineResult từ Pipeline và không crash."""
        from unittest.mock import MagicMock

        from drivevision.pipeline.pipeline import Pipeline
        from drivevision.perception.base import Detector
        from drivevision.viz.annotator import Annotator

        class MockDetector(Detector):
            def detect(self, frame):
                return sample_detections

        result = Pipeline(detector=MockDetector()).process(dummy_frame)
        ann = Annotator()
        out = ann.draw(result, fps=25.0)
        assert out.shape == dummy_frame.image.shape


# ─────────────────────────── Test FPS Counter ────────────────────

class TestFPSCounter:
    def test_fps_zero_on_single_tick(self):
        """Chỉ 1 tick → FPS = 0 (chưa đủ dữ liệu)."""
        import time
        from drivevision.cli import _FPSCounter  # type: ignore

        counter = _FPSCounter(window=30)
        counter.tick()
        assert counter.fps == 0.0

    def test_fps_positive_after_two_ticks(self):
        """Sau 2 tick cách nhau 0.1s → FPS khoảng 10."""
        import time
        from drivevision.cli import _FPSCounter  # type: ignore

        counter = _FPSCounter(window=30)
        counter.tick()
        time.sleep(0.1)
        counter.tick()
        assert counter.fps > 5.0   # ít nhất 5 FPS (tolerant về sleep accuracy)


# ─────────────────────────── Test config ─────────────────────────

class TestConfig:
    def test_load_default_config(self):
        from drivevision.config import load_config

        cfg = load_config()
        assert cfg["source"]["type"] == "video"
        assert cfg["output"]["display"] is False

    def test_get_path(self):
        from drivevision.config import get_path

        cfg = {"perception": {"detection": {"conf": 0.35}}}
        assert get_path(cfg, "perception.detection.conf") == 0.35
        assert get_path(cfg, "perception.detection.missing", 99) == 99

    def test_get_path_missing_key_returns_default(self):
        from drivevision.config import get_path

        cfg = {}
        assert get_path(cfg, "a.b.c", "default") == "default"
```

**Lưu ý test:**
- `pytest.importorskip("ultralytics")` — bỏ qua test nếu ultralytics chưa cài (CI-friendly)
- `pytest.importorskip("torch")` — tương tự cho torch
- `_FPSCounter` import từ `cli.py` — nếu không muốn expose, chuyển thành module `utils/fps.py` sau
- Test `TestBoundingBox.test_iou_no_overlap` có typo (`bb.iou` thay vì `a.iou`) — sửa trong code thực

---

### Task 1.7 — Tạo thư mục output và data/samples

**Mục đích:** Đảm bảo cấu trúc thư mục cần thiết tồn tại.

**Các bước:**

```bash
# Tạo thư mục output cho video kết quả
mkdir -p /home/quan/DriveVision/output

# Tạo thư mục data/samples cho video mẫu
mkdir -p /home/quan/DriveVision/data/samples

# Tạo .gitkeep để git track thư mục trống
touch /home/quan/DriveVision/output/.gitkeep
touch /home/quan/DriveVision/data/samples/.gitkeep
touch /home/quan/DriveVision/data/.gitkeep
```

**Thêm vào `.gitignore`:**

```gitignore
# Output videos
output/*.mp4
output/*.avi

# Sample data (không commit video)
data/samples/*.mp4
data/samples/*.avi
data/samples/*.mkv

# YOLO auto-download cache
~/.config/Ultralytics/
```

---

### Task 1.8 — Cập nhật `scripts/run_pipeline.py` (không bắt buộc, đã đủ)

**File:** `scripts/run_pipeline.py`

Script hiện tại đã đúng — chỉ delegate sang `cli.main()`. Không cần thay đổi. Đảm bảo người dùng biết dùng:

```bash
# Chạy với video, lưu kết quả, không display
python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 \
    --save output/demo.mp4

# Chạy với display (cần màn hình)
python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 \
    --display

# Chạy headless, đo FPS
python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 \
    --max-frames 100

# Chạy với webcam
python scripts/run_pipeline.py --source 0 --display
```

---

## 8. Thay đổi cấu hình

### 8.1 `configs/default.yaml` — các key thay đổi/thêm mới

```yaml
# Thay đổi: thêm window_name, làm rõ ý nghĩa từng key
output:
  display: false           # bool — hiển thị cửa sổ OpenCV (cần màn hình)
  save_path: null          # str | null — đường dẫn lưu .mp4; null = không lưu
  fps: 30                  # float — FPS khi ghi file output (không ảnh hưởng inference)
  window_name: "DriveVision"  # [MỚI] tên cửa sổ cv2.imshow

# Thêm mới: điều khiển logging
logging:
  level: INFO              # [MỚI] DEBUG | INFO | WARNING | ERROR
  log_interval: 30         # [MỚI] in log mỗi N frames

# Thay đổi: thêm device (chưa dùng ở Phase 1, chuẩn bị cho Phase 8)
perception:
  detection:
    enabled: true
    weights: models/weights/yolo.pt
    conf: 0.35
    iou: 0.5
    classes: null
    imgsz: 640
    device: null           # [MỚI] null=auto | "cpu" | "cuda:0"
```

### 8.2 `src/drivevision/config.py` — `DEFAULT_CONFIG` cần đồng bộ

```python
DEFAULT_CONFIG: Dict[str, Any] = {
    "source": { ... },   # không thay đổi
    "perception": {
        "detection": {
            "enabled": True,
            "weights": "models/weights/yolo.pt",
            "conf": 0.35,
            "iou": 0.5,
            "classes": None,
            "imgsz": 640,
            "device": None,  # THÊM
        },
        ...
    },
    "output": {
        "display": False,
        "save_path": None,
        "fps": 30,
        "window_name": "DriveVision",  # THÊM
    },
    "logging": {           # THÊM
        "level": "INFO",
        "log_interval": 30,
    },
    ...
}
```

---

## 9. Kiểm thử

### 9.1 Unit tests

| Test | File | Mô tả | Kỳ vọng |
|---|---|---|---|
| `test_bbox_as_int` | `test_phase1.py` | BoundingBox.as_int() | (10,20,200,150) |
| `test_bbox_iou_identical` | `test_phase1.py` | IoU box với chính nó | 1.0 |
| `test_bbox_iou_no_overlap` | `test_phase1.py` | IoU 2 box không chồng | 0.0 |
| `test_annotator_shape` | `test_phase1.py` | Output cùng shape input | pass |
| `test_annotator_no_modify` | `test_phase1.py` | Không sửa ảnh gốc | pass |
| `test_annotator_empty` | `test_phase1.py` | 0 detection không crash | pass |
| `test_annotator_many` | `test_phase1.py` | 20 detection không crash | pass |
| `test_yolo_empty_result` | `test_phase1.py` | YOLO trả [] → detect() = [] | pass |
| `test_yolo_parses_boxes` | `test_phase1.py` | Parse xyxy/cls/conf đúng | Detection match |
| `test_fps_zero_single` | `test_phase1.py` | 1 tick → FPS=0 | pass |
| `test_fps_positive` | `test_phase1.py` | 2 tick cách 0.1s → FPS>5 | pass |

### 9.2 Integration tests

| Test | Mô tả | Chạy |
|---|---|---|
| Pipeline với MockDetector | Detections đúng số lượng và class | `pytest tests/test_phase1.py::TestPipelinePhase1` |
| Annotator nhận result từ Pipeline | Không crash, shape đúng | `pytest tests/test_phase1.py::TestPipelinePhase1::test_annotator_integrates` |
| Pipeline không có detector | result.scene.detections == [] | `pytest tests/test_phase1.py::TestPipelinePhase1::test_pipeline_no_detector` |

### 9.3 End-to-end manual test

```bash
# Test 1: pipeline headless 100 frames (kiểm tra FPS và không crash)
PYTHONPATH=src python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 \
    --max-frames 100

# Kỳ vọng output log:
# INFO drivevision: Pipeline sẵn sàng. Detector: YOLOv8 | Save: không lưu | Display: False
# INFO drivevision: Frame    30 | Dets:  5 | FPS:  12.4
# INFO drivevision: Frame    60 | Dets:  3 | FPS:  13.1
# ...
# INFO drivevision: Hoàn thành: 100 frames | 421 detections tổng | FPS trung bình: 12.8 | Thời gian: 7.8s

# Test 2: lưu video output
PYTHONPATH=src python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 \
    --save output/demo.mp4 \
    --max-frames 200

# Kỳ vọng: file output/demo.mp4 tồn tại và mở được

# Test 3: graceful khi thiếu ultralytics
pip uninstall ultralytics -y
PYTHONPATH=src python scripts/run_pipeline.py --source data/samples/sample.mp4 --max-frames 10
# Kỳ vọng: WARNING Detection disabled (...). `pip install ultralytics` to enable.
# pipeline vẫn chạy, detections = 0 mỗi frame
pip install "ultralytics>=8.2"

# Test 4: display mode (cần màn hình)
PYTHONPATH=src python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 \
    --display
# Kỳ vọng: cửa sổ "DriveVision" xuất hiện, nhấn 'q' để đóng
```

### 9.4 Edge cases cần kiểm tra

| Edge case | Mô tả | Xử lý |
|---|---|---|
| Video rỗng / corrupt | `VideoSource` không đọc được | `RuntimeError` rõ ràng ở `VideoSource._open()` |
| `save_path` là thư mục không tồn tại | `output/subdir/demo.mp4` | `mkdir -p` trong `_make_video_writer()` |
| Video 1 frame | Chạy xong ngay | FPS=0 (1 tick), không crash |
| `conf=1.0` (threshold rất cao) | Không có detection nào | `[]` — pipeline và annotator xử lý bình thường |
| `conf=0.0` (threshold rất thấp) | Rất nhiều detection | Annotator vẽ tất cả, chậm hơn nhưng không crash |
| Ảnh grayscale (1 channel) | YOLO cần 3 channel | `YOLODetector` pass thẳng numpy array — ultralytics tự xử lý |
| KeyboardInterrupt giữa chừng | Ctrl+C trong vòng lặp | `finally` đảm bảo `writer.release()` |

---

## 10. Tiêu chí hoàn thành (Definition of Done)

### Chức năng cốt lõi

- [ ] `python scripts/run_pipeline.py --source data/samples/sample.mp4 --save output/demo.mp4` chạy không crash
- [ ] File `output/demo.mp4` tồn tại, mở được, có bounding box và nhãn màu
- [ ] Log cuối pipeline in FPS trung bình là số dương
- [ ] Bounding box bao quanh đúng đối tượng (car, person, truck, bus...)
- [ ] Nhãn hiển thị đúng `{class_name} {confidence:.2f}` (ví dụ: "car 0.87")
- [ ] HUD góc trên-trái hiển thị FPS và số detection

### Khả năng cấu hình

- [ ] `--save <path>` ghi video ra đường dẫn chỉ định
- [ ] `--display` / `--no-display` bật/tắt cửa sổ cv2
- [ ] `--max-frames N` dừng sau N frames
- [ ] `--conf 0.5` thay đổi threshold detection
- [ ] `output.save_path` trong YAML hoạt động
- [ ] `output.display` trong YAML hoạt động

### Độ bền

- [ ] Chạy không crash khi `ultralytics` chưa được cài (chỉ warning)
- [ ] Chạy không crash khi `output.save_path = null`
- [ ] Ctrl+C dừng sạch, VideoWriter được release
- [ ] Chạy trên video 0 frame không crash

### Chất lượng code

- [ ] `PYTHONPATH=src pytest tests/ -q` — 0 failed
- [ ] `tests/test_phase1.py` có ít nhất 15 test case
- [ ] Không có import vòng tròn
- [ ] Type hints đầy đủ trên tất cả function public
- [ ] Docstring trên class và method public

### Khả năng mở rộng cho Phase 2

- [ ] `Annotator.draw(result, fps=0.0)` không cần thay đổi chữ ký khi Phase 2 thêm track IDs
- [ ] `Pipeline.process()` không bị sửa
- [ ] `PipelineResult` không bị sửa

---

## 11. Rủi ro & cách xử lý

| # | Rủi ro | Khả năng | Tác động | Cách xử lý |
|---|---|---|---|---|
| R1 | `ultralytics` không cài được do conflict dep | Trung bình | Cao | Dùng virtualenv sạch; thử `pip install ultralytics --no-deps` + cài từng dep thủ công |
| R2 | Không có video mẫu | Thấp | Trung bình | Dùng ffmpeg tạo video synthetic (Cách B trong Task 1.4); hoặc webcam `--source 0` |
| R3 | Máy không có GPU → FPS quá thấp (< 5) | Cao | Thấp | Dùng `imgsz=320` thay vì 640; tăng conf lên 0.5 để ít detection; Phase 1 không cần real-time |
| R4 | `cv2.VideoWriter` không ghi được MP4 trên Linux thiếu codec | Trung bình | Trung bình | Thử codec `XVID` với `.avi`; hoặc cài `ffmpeg` và dùng pipe |
| R5 | `cv2.imshow` crash trên headless server | Thấp | Thấp | Mặc định `display: false`; cảnh báo rõ trong log nếu `DISPLAY` không set |
| R6 | Memory leak: VideoWriter không được release | Thấp | Trung bình | Đảm bảo `finally` block luôn gọi `writer.release()` |
| R7 | YOLO tải weights lần đầu mất nhiều thời gian | Thấp | Thấp | Cache `~/.config/Ultralytics/` — lần sau nhanh; preload bằng `YOLO("yolov8n.pt")` |
| R8 | Pipeline builder không tìm thấy `perception/detection.py` | Thấp | Cao | Kiểm tra `PYTHONPATH=src` và `pip install -e .` |

### Giải pháp codec VideoWriter trên Linux

```python
# Thử lần lượt nếu mp4v không hoạt động
CODECS_TO_TRY = ["mp4v", "avc1", "XVID", "MJPG"]
for codec in CODECS_TO_TRY:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    ext = ".mp4" if codec in ("mp4v", "avc1") else ".avi"
    writer = cv2.VideoWriter(path_with_ext, fourcc, fps, (w, h))
    if writer.isOpened():
        break
```

---

## 12. Hiệu năng & tài nguyên

### Kỳ vọng FPS (Phase 1, không có GPU)

| Cấu hình | CPU (Intel i7) | GPU (Kaggle T4) |
|---|---|---|
| `imgsz=640`, `yolov8n` | 8–15 FPS | 60–100 FPS |
| `imgsz=320`, `yolov8n` | 20–40 FPS | 100+ FPS |
| `imgsz=640`, `yolov8s` | 4–8 FPS | 40–70 FPS |

**Lưu ý:** Phase 1 không ưu tiên real-time. FPS 8–15 trên CPU là đủ tốt cho demo.

### Tối ưu nếu cần

```yaml
# Trong configs/default.yaml — giảm tải khi test nhanh
perception:
  detection:
    imgsz: 320   # giảm từ 640 → tăng ~2-3x FPS, giảm accuracy nhẹ
    conf: 0.5    # tăng threshold → ít detection → nhanh hơn
```

### Bộ nhớ

- Model YOLOv8n: ~15 MB RAM (CPU) / ~40 MB VRAM (GPU)
- Video frame 1280x720 BGR: ~2.6 MB / frame (numpy array)
- VideoWriter buffer: cv2 quản lý nội bộ, không tích lũy
- Vòng lặp frame không lưu list toàn bộ frame → memory constant

### Profiling nhanh

```bash
# Đo FPS thực tế
PYTHONPATH=src python - <<'EOF'
import time, numpy as np
from drivevision.config import load_config
from drivevision.pipeline.builder import build_pipeline
from drivevision.types import Frame

cfg = load_config()
pipeline = build_pipeline(cfg)

frames = [Frame(i, i/30.0, np.random.randint(0,255,(720,1280,3),dtype=np.uint8)) for i in range(50)]

t0 = time.perf_counter()
for f in frames:
    pipeline.process(f)
t1 = time.perf_counter()
print(f"FPS: {50/(t1-t0):.1f}")
EOF
```

---

## 13. Sản phẩm bàn giao

| # | Sản phẩm | Mô tả |
|---|---|---|
| D1 | `src/drivevision/viz/annotator.py` | Hoàn chỉnh: vẽ bbox, nhãn, HUD, FPS |
| D2 | `src/drivevision/cli.py` | Hoàn chỉnh: args, VideoWriter, FPS counter, display |
| D3 | `configs/default.yaml` | Cập nhật: `output.window_name`, `logging.*`, `perception.detection.device` |
| D4 | `src/drivevision/config.py` | `DEFAULT_CONFIG` đồng bộ với YAML |
| D5 | `tests/test_phase1.py` | 15+ test case (unit + integration) |
| D6 | `output/demo.mp4` | Video mẫu annotated (không commit vào git) |
| D7 | `data/samples/.gitkeep` | Thư mục placeholder |
| D8 | `output/.gitkeep` | Thư mục placeholder |

**KHÔNG giao** (ngoài phạm vi):
- Fine-tuned weights (Phase 8)
- API endpoints (Phase 7)
- Tracking IDs (Phase 2)

---

## 14. Điểm nhấn cho Portfolio

### Điểm kỹ thuật đáng mention

1. **Kiến trúc 2 mặt phẳng** (Research/Runtime): giải thích rõ ràng tại sao tách biệt — weights là artifact duy nhất trao đổi giữa Kaggle (training) và local (inference). Phù hợp với MLOps best practice.

2. **Pipeline composable hoàn toàn** (`detector=None` bỏ qua stage): cho thấy thiết kế forward-looking — Phase 1 chỉ có detector nhưng pipeline không cần refactor khi thêm tracker, lane, traffic light.

3. **Graceful degradation**: khi thiếu `ultralytics`, pipeline vẫn chạy (với `detector=None`). Không crash ở import time.

4. **Type safety**: toàn bộ pipeline dùng dataclass typed — `Detection`, `BoundingBox`, `PipelineResult` — không có `dict` nặc danh truyền giữa các stage.

5. **FPS sliding window**: dùng `deque(maxlen=30)` thay vì EMA hay counter đơn giản — cho số FPS ổn định hơn trong 30 frame gần nhất.

### Cách trình bày trong Portfolio

```markdown
## Demo: Object Detection Pipeline (Phase 1)
- Model: YOLOv8n pretrained (COCO 80 classes)
- Input: BDD100K driving video clip
- Output: Annotated video với bounding box màu phân biệt theo class,
          confidence score, HUD FPS real-time
- FPS: ~12 FPS trên CPU (Intel i7), ~80 FPS trên GPU (T4)
- Thiết kế: Pipeline composable, graceful degradation khi thiếu ML deps
```

### Screenshot / GIF demo

Ghi lại GIF bằng:

```bash
# Dùng ffmpeg cắt 10s đầu để tạo GIF nhẹ cho README
ffmpeg -i output/demo.mp4 -t 10 -vf "fps=10,scale=640:-1" \
  -gifflags +transdiff demo_phase1.gif
```

---

## 15. Tham khảo

| Tài nguyên | URL / Path | Dùng cho |
|---|---|---|
| Ultralytics YOLOv8 docs | https://docs.ultralytics.com/ | API reference cho `YOLO.predict()` |
| COCO class list (80 classes) | https://cocodataset.org/#explore | Mapping `class_id → tên` |
| OpenCV VideoWriter docs | https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html | Codec, params |
| BDD100K dataset | https://bdd-data.berkeley.edu/ | Video mẫu giao thông |
| YOLOv8n pretrained weights | Tự tải bởi `ultralytics` từ GitHub | `yolov8n.pt` |
| Phase 0 spec | `/home/quan/DriveVision/phase_0.md` | Interface đã xác lập |
| `types.py` | `/home/quan/DriveVision/src/drivevision/types.py` | Contract của pipeline |
| `pipeline.py` | `/home/quan/DriveVision/src/drivevision/pipeline/pipeline.py` | Không thay đổi |
| `builder.py` | `/home/quan/DriveVision/src/drivevision/pipeline/builder.py` | Wiring config → objects |

---

## 16. Checklist tổng kết

### Trước khi bắt đầu

- [ ] Python 3.12 đang active: `python --version`
- [ ] Venv kích hoạt: `which python` trỏ vào `.venv`
- [ ] `pip install -e ".[dev]"` thành công
- [ ] `PYTHONPATH=src pytest tests/ -q` pass (P0 tests)
- [ ] Có ít nhất 1 video trong `data/samples/`

### Trong quá trình implement

- [ ] Task 1.1: `viz/annotator.py` hoàn chỉnh
  - [ ] `_class_color()` hoạt động cho 80 class
  - [ ] `_draw_label()` vẽ text trên nền tối
  - [ ] `_draw_hud()` hiển thị FPS + frame index + n_dets
  - [ ] `draw(result, fps)` gọi đúng thứ tự layers
- [ ] Task 1.2: `cli.py` hoàn chỉnh
  - [ ] `argparse` có đủ args: `--config`, `--source`, `--save`, `--display`, `--no-display`, `--max-frames`, `--conf`
  - [ ] `_FPSCounter` implement đúng với `deque`
  - [ ] `_make_video_writer()` tạo thư mục cha nếu chưa tồn tại
  - [ ] Vòng lặp có `try/KeyboardInterrupt/finally` đảm bảo release
  - [ ] Log định kỳ mỗi 30 frame
  - [ ] Tóm tắt cuối: FPS, tổng frame, tổng detection
- [ ] Task 1.3: `detection.py` đã xác nhận (không cần sửa)
- [ ] Task 1.4: `configs/default.yaml` cập nhật
- [ ] Task 1.5: `config.py DEFAULT_CONFIG` đồng bộ
- [ ] Task 1.6: `tests/test_phase1.py` có đủ test cases
- [ ] Task 1.7: thư mục `output/` và `data/samples/` tồn tại

### Sau khi implement

- [ ] `PYTHONPATH=src pytest tests/ -q` — **0 failed**
- [ ] Chạy end-to-end: `python scripts/run_pipeline.py --source <video> --save output/demo.mp4`
- [ ] Mở `output/demo.mp4` bằng trình phát — bounding box và nhãn hiển thị đúng
- [ ] Thử `--display` trên máy có màn hình — cửa sổ hiện, 'q' đóng được
- [ ] Thử `--conf 0.7` — ít detection hơn, FPS tăng nhẹ
- [ ] Thử `--source 0` (webcam nếu có)
- [ ] Uninstall ultralytics → pipeline vẫn chạy với warning → reinstall
- [ ] Tạo GIF demo 10 giây cho portfolio
- [ ] Commit tất cả thay đổi với message rõ ràng: `feat(phase1): complete detection viz + output pipeline`
- [ ] Xác nhận Phase 2 có thể cắm vào: `Pipeline(detector=..., tracker=SimpleTracker(...))` hoạt động với Annotator hiện tại
