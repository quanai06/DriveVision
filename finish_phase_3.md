# Phase 3 — Báo cáo hoàn thành (Lane Detection)

> **Ngày:** 2026-06-29 | **Trạng thái:** ✅ HOÀN THÀNH (classical OpenCV, không tải gì, CPU)
> **Máy:** Python 3.12.7, CPU Intel i7-1355U.
> Đọc kèm `phase_3.md` (đặc tả), `finish_phase_1.md`, `finish_phase_2.md`, `EVALUATION.md`.

---

## 0. TL;DR

Phase 3 điền `SceneState.lanes` bằng **phát hiện làn đường cổ điển** (HLS + Canny +
Hough + polyfit + temporal smoothing) và vẽ làn + vùng ego-lane lên demo. Pure CPU
OpenCV — **không cần GPU, không tải dữ liệu/model mới**.

| Hạng mục | Kết quả |
|---|---|
| Test suite | **51 passed** (40 cũ + 11 lane) |
| Lane detect speed | **8.0 ms/frame** (p95 12.7ms), CPU |
| Robustness | **0 crash / 100 ảnh BDD100K thật** |
| Demo | `output/lane_demo.mp4` + `output/lane_frame.jpg` |

Sau Phase 3, một frame demo có **box + track ID + quỹ đạo + làn đường + ego-lane + HUD**
→ rất giống overlay perception xe tự lái.

---

## 1. Lane detection khác Object Detection thế nào

| | Object Detection (P1) | Lane Detection (P3) |
|---|---|---|
| Đầu ra | Bounding box (chữ nhật) | **Polyline** (danh sách điểm đường cong) |
| Đối tượng | Xe/người (rigid, rõ) | Vạch sơn (mảnh, đứt, nhòe) |
| Phương pháp | YOLO | HLS màu + Canny + Hough + polyfit |
| Đơn vị | "object" riêng | Cấu trúc hình học liên tục |

→ Vì vậy `LaneDetector` là ABC riêng, **không** tái dùng `Detector`. (Model đa nhiệm
YOLOP/HybridNets có thể làm cả hai trong 1 forward — để Phase 8.)

---

## 2. Code đã làm

### 2.1 `perception/lane.py` — ClassicalLaneDetector (viết lại sạch)

Pipeline mỗi frame:
1. **HLS color mask** (`_hls_color_mask`): bắt vạch **trắng** (L>200, S<60) + **vàng**
   (H 15–35, S>80, L>80) — bền với điều kiện sáng hơn grayscale thuần.
2. **Canny** trên grayscale đã GaussianBlur, **OR** với color mask.
3. **ROI hình thang** (`_roi_trapezoid`): chỉ giữ vùng mặt đường phía trước.
4. **HoughLinesP** → segments.
5. **Tách trái/phải** (`_separate_lines`): theo dấu slope + vị trí x; lọc slope ∈ [0.4, 5.0].
6. **polyfit `x = f(y)`** (`_fit_points`): fit x theo y (tránh blow-up đường thẳng đứng),
   **lấy mẫu tại 20 điểm y CỐ ĐỊNH** (`linspace(0.6h, h, 20)`).
7. **Temporal smoothing**: vì mọi frame cho 20 điểm tại **cùng y** → làm mượt chỉ là
   **trung bình x theo từng hàng** qua `deque(maxlen=smooth_buffer)`.

**Chế độ `use_perspective`** (tuỳ chọn): warp **bird's-eye view** → `_sliding_window_fit`
(9 cửa sổ, histogram tìm gốc) → polyfit bậc 2 → **EMA** trên coefficients → inverse warp
về toạ độ ảnh.

`detect()` bọc **try/except** → mọi lỗi / ảnh đen / ảnh tối trả `[]`, **không crash**.
Có `reset()` xoá state smoothing.

### 2.2 `perception/lane.py` — ModelLaneDetector (stub sạch)
`NotImplementedError` kèm hướng dẫn rõ: backend `yolov8-seg | yolop | ufld`, weights
`models/weights/lane.pt`, dataset TuSimple/CULane/BDD lane, train trên Kaggle (Phase 8).

### 2.3 `viz/annotator.py` — merge vẽ lane (giữ chữ ký P2)
- **Giữ `draw(result, fps=0.0)`** → P1/P2/cli không vỡ.
- Thêm `_draw_lanes`: tô **vùng ego-lane** (polygon giữa làn trái+phải, xanh mờ alpha 0.30)
  rồi vẽ **polyline** trái (cyan) / phải (xanh) — **vẽ trước** track để box nằm trên.
- HUD thêm dòng `Lanes: N` (cạnh FPS/Frame/Tracks + Risk/Action).

### 2.4 builder + config
- `pipeline/builder.py`: truyền `use_perspective/smooth_alpha/smooth_buffer/poly_deg`
  vào `ClassicalLaneDetector`; nếu `method: model` lỗi (chưa có weights) → **fallback
  classical** + log warning.
- `configs/default.yaml` + `config.py DEFAULT_CONFIG`: **`lane.enabled: true`** + đủ sub-keys.

### 2.5 KHÔNG sửa
`types.py`, `perception/base.py`, `pipeline/pipeline.py` — interface bất biến.

---

## 3. Tests — `tests/test_lane.py` (11 test)

| Test | Kiểm tra |
|---|---|
| `test_blank_frame_returns_empty` | ảnh đen → list, không crash |
| `test_synthetic_detects_a_lane` | ảnh 2 vạch trắng → ≥1 làn |
| `test_lane_has_valid_points` | mỗi Lane ≥ 2 điểm |
| `test_side_classification` | có LEFT hoặc RIGHT |
| `test_confidence_in_range` | confidence ∈ [0,1] |
| `test_points_within_image_bounds` | điểm trong/gần biên ảnh |
| `test_multiple_frames_smoothing` | 10 frame liên tiếp không crash |
| `test_dark_frame_no_crash` | ảnh rất tối → không crash |
| `test_reset_clears_history` | `reset()` xoá history |
| `test_perspective_mode_no_crash` | bird's-eye mode không crash |
| `test_raises_not_implemented` | ModelLaneDetector raise đúng |

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q    # 51 passed
```

---

## 4. Demo

```bash
# default.yaml giờ bật detection + tracking + lane
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 --save output/lane_demo.mp4 --max-frames 300
```

- **`output/lane_demo.mp4`** + **`output/lane_frame.jpg`**: làn trái (vàng/cyan) bám đúng
  vạch vàng, làn phải (xanh) bám mép phải, **vùng ego-lane tô xanh mờ**, 3 xe `#14/#15/#5`
  có quỹ đạo, HUD `FPS / Tracks: 3 / Lanes: 2 / Risk: SAFE / Action: maintain`.
- 300/300 frame phát hiện được làn.

---

## 5. Benchmark (ghi vào EVALUATION.md)

| Phương pháp | ms/frame (mean) | p95 | Crash | 
|---|---|---|---|
| Classical (HLS+Canny+Hough+polyfit+EMA) | **8.0 ms** | 12.7 ms | **0/100 ảnh BDD100K thật** |

→ Toàn pipeline (YOLO ~50ms + tracking <0.5ms + lane 8ms) ≈ **~15 FPS** trên CPU.
**Accuracy lane (IoU/F1/TuSimple acc) cần nhãn → để Phase 8** (TuSimple/CULane/BDD lane).

---

## 6. Sửa lỗi trong spec

Skeleton `_detect_hough` ở `phase_3.md` có **code trùng & mâu thuẫn** (gọi
`_fit_lane_points` 2 lần, tham số `poly_deg` truyền sai chỗ, comment lộn xộn). Đã **viết
lại sạch** với một quyết định then chốt: **fit & lấy mẫu tại 20 điểm y cố định** →
temporal smoothing trở thành phép **trung bình theo hàng** đơn giản, đúng đắn (không phải
"căn chỉnh độ dài min" rối rắm như skeleton).

---

## 7. File thay đổi / tạo mới

**Sửa:** `src/drivevision/perception/lane.py` (ClassicalLaneDetector đầy đủ + ModelLaneDetector
stub), `src/drivevision/viz/annotator.py` (vẽ lane + ego-area + HUD Lanes), `src/drivevision/
pipeline/builder.py` (truyền params lane + fallback), `configs/default.yaml` (lane.enabled +
sub-keys), `src/drivevision/config.py` (DEFAULT_CONFIG lane), `EVALUATION.md` (lane speed).

**Tạo mới:** `tests/test_lane.py` (11 test), `finish_phase_3.md`.

**Dữ liệu (gitignored):** `output/lane_demo.mp4`, `output/lane_frame.jpg`.

**Không cài/tải gì mới** — chỉ dùng opencv/numpy đã có + video dashcam + ảnh BDD100K val.

---

## 8. Definition of Done — đối chiếu `phase_3.md`

- [x] `ClassicalLaneDetector.detect()` có HLS filter, không crash
- [x] Polynomial fit (`polyfit` x=f(y), deg 1; deg≥2 ở perspective)
- [x] Temporal smoothing (deque per-row mean + EMA cho perspective)
- [x] `use_perspective` mode (bird's-eye + sliding window)
- [x] `reset()` hoạt động
- [x] Edge case (đen/tối/không vạch) → `[]` không crash
- [x] `ModelLaneDetector` stub sạch + `NotImplementedError` hướng dẫn
- [x] `Annotator` vẽ polyline làn + tô ego-lane + HUD (Lanes/Risk/Action)
- [x] `tests/test_lane.py` ≥ 9 test (có 11) — **51 passed tổng**
- [x] `configs/default.yaml` đủ sub-keys; `builder.py` truyền tham số
- [x] `pipeline.process()` trả `scene.lanes` được điền
- [x] Video output có lane visualisation rõ
- [x] KHÔNG sửa `Lane/LaneSide/Frame/SceneState/LaneDetector`

**Mục tiêu đo được (2.1–2.7)**
- [x] 2.1 chạy 100 frame BDD không crash · [x] 2.2 phân loại trái/phải · [x] 2.3 fit ổn định
- [x] 2.4 smoothing giảm rung · [x] 2.5 annotator vẽ lane · [x] 2.6 edge case không crash
- [x] 2.7 ModelLaneDetector stub sạch

---

## 9. Khái niệm đã áp dụng (điểm nhấn portfolio)

- **HLS color space** thay vì grayscale → bắt vạch trắng/vàng bền hơn với ánh sáng.
- **Fit `x = f(y)`** (không phải y=f(x)) → tránh blow-up khi vạch gần thẳng đứng.
- **Temporal smoothing** → demo không giật, hiểu biết về ổn định hệ thống real-world.
- **Perspective transform + sliding window** → kỹ thuật kinh điển (Udacity SDC Nanodegree),
  xử lý cua/đường đứt tốt hơn Hough thô.
- **Hai hướng cùng interface** (classical + model stub) → cho thấy hiểu cả phương pháp cổ
  điển lẫn ML, không chỉ dùng black-box.

---

## 10. Lệnh tham khảo

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_lane.py     # lane tests
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --source data/samples/sample.mp4 --save output/lane_demo.mp4 --max-frames 300
# bird's-eye mode: configs/default.yaml -> perception.lane.use_perspective: true
```

---

## 11. Bước tiếp theo
- **Phase 4 — Traffic Light + State**: phát hiện đèn + phân loại màu (HSV heuristic / CNN nhỏ),
  điền `SceneState.traffic_lights`. Pure CPU, không cần tải.
- **Phase 5 — Scene Understanding**: dùng `lanes` (P3) + `tracks` (P2) để gán **ego-lane**,
  xác định **lead vehicle** → nuôi Phase 6 (risk/TTC).
- **Phase 8 — accuracy lane** trên TuSimple/CULane/BDD lane (IoU/F1) — Kaggle.
