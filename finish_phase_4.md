# Phase 4 — Báo cáo hoàn thành (Traffic Light Detection + State)

> **Ngày:** 2026-06-29 | **Trạng thái:** ✅ HOÀN THÀNH (HSV heuristic, CPU, không tải gì)
> **Máy:** Python 3.12.7, CPU Intel i7-1355U.
> Đọc kèm `phase_4.md`, `finish_phase_1..3.md`, `EVALUATION.md`.

---

## 0. TL;DR

Phase 4 điền `SceneState.traffic_lights`: **tái dùng box đèn từ YOLO** (COCO class 9)
rồi **phân loại màu** (đỏ/vàng/xanh/UNKNOWN) bằng **HSV heuristic** — không train,
không model mới, pure CPU.

| Hạng mục | Kết quả |
|---|---|
| Test suite | **66 passed** (51 cũ + 15 traffic light) |
| HSV classify speed | **0.10 ms/đèn** (DoD ≤2ms ✅) |
| Functional (250 ảnh BDD val) | red 35 / yellow 38 / green 18 |
| Demo | `output/tl_best_{0,1,2}.jpg` — 3 đèn RED+GREEN+GREEN phân loại đúng |

---

## 1. Hai bài toán tách biệt

1. **Đèn ở đâu? (Detection)** → YOLO COCO class `traffic light` (id 9) đã giải, **không
   cần model riêng**.
2. **Đèn màu gì? (Classification)** → bài toán riêng trên **crop nhỏ** (~20×50px). HSV
   heuristic đủ tốt ở điều kiện sáng tốt; CNN tiny là nâng cấp Phase 8.

Tách rõ → thay classifier (Phase 8) không đụng pipeline.

---

## 2. Code đã làm

### 2.1 `perception/traffic_light.py` — SimpleTrafficLightDetector
- `__init__(config)` đọc `perception.traffic_light.hsv` (merge default).
- `detect(frame, detections)`: lọc `class_id == 9` → `_safe_crop` → `_classify_state`
  → `List[TrafficLight]`.
- `_safe_crop`: clamp toạ độ trong frame; crop < `min_crop_size` (10px) → `None` → UNKNOWN.
- `_classify_state`: **chiến lược B trước, fallback A**:
  - **B `_color_mask_classify`** (color mask): đếm pixel đỏ/vàng/xanh trên toàn crop,
    chọn màu trội. **Đỏ dùng 2 dải H** (wrap quanh 0°/180°). conf = pixel_trội / diện tích.
  - **A `_region_based_classify`** (region split): chia 3 vùng dọc (đỏ-trên / vàng-giữa
    / xanh-dưới), chọn vùng **sáng nhất** (mean kênh V). Dùng khi B không chắc.
- Edge cases (crop nhỏ, tối, ngược sáng) → **UNKNOWN** thay vì đoán bừa.

### 2.2 `viz/annotator.py` — vẽ traffic light (merge)
- Giữ `draw(result, fps=)`; thêm `_draw_traffic_lights`: **box dày màu theo state**
  (RED đỏ / GREEN xanh / YELLOW vàng / UNKNOWN xám) + nhãn `STATE conf`. HUD thêm `Lights: N`.

### 2.3 builder + config
- `builder.py`: `SimpleTrafficLightDetector(config=tl_cfg)` (truyền cả dict con).
- `configs/default.yaml` + `config.py`: **`traffic_light.enabled: true`** + `classifier`
  + `hsv.{min_saturation, min_value, confidence_threshold, min_crop_size}`.
- `configs/phase4.yaml`: config chạy riêng Phase 4 (spec parity).

### 2.4 KHÔNG sửa
`types.py`, `perception/base.py`, `pipeline/pipeline.py`.

---

## 3. Tests — `tests/test_traffic_light.py` (15 test)

- `TestColorMaskClassify`: red/green/yellow nhận đúng, ảnh tối → conf thấp.
- `TestRegionBasedClassify`: vùng trên sáng → RED, dưới sáng → GREEN, crop <3px → UNKNOWN.
- `TestDetectEndToEnd`: lọc non-class-9, detect RED, rỗng, bbox ngoài frame không crash,
  nhiều đèn, bbox quá nhỏ → UNKNOWN.
- `TestConfigOverride`: override `min_saturation`, default values.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q   # 66 passed
```

---

## 4. Demo (ảnh BDD100K thật có đèn)

Dashcam Udacity (cao tốc) không có đèn → demo dùng **ảnh BDD100K val thật**.
- `output/tl_best_1.jpg`: ngã tư — đèn **GREEN** lớn (phải) + **RED** (trái) + **GREEN**
  (xa) phân loại **đúng cả 3**, kèm 5 xe; HUD `Dets: 8 / Lights: 3`.
- `output/tl_best_0.jpg` (RED), `output/tl_best_2.jpg` (YELLOW).
- 250 ảnh: 115 đèn rõ được phân loại màu.

---

## 5. Benchmark (EVALUATION.md)

| Phương pháp | ms/đèn | DoD | Functional |
|---|---|---|---|
| HSV heuristic | **0.10 ms** | ≤2ms ✅ | red 35 / yellow 38 / green 18 trên 250 ảnh BDD |

**Hạn chế đã biết:** đèn nhỏ/xa dễ nhầm (vd green↔yellow ở ảnh `tl_demo_1`) — đúng bản
chất HSV. **Accuracy có nhãn (LISA/BTTSD) → Phase 8.**

---

## 6. Khái niệm / điểm nhấn

- **Tái dùng > tạo mới:** không train detector đèn riêng, dùng lại YOLO COCO.
- **HSV color space:** tách hue khỏi độ sáng → phân loại màu tốt hơn RGB. Đỏ cần **2 dải H**.
- **Dual-strategy + fallback:** color-mask (chính) + region-split (dự phòng) = robust.
- **Fail-safe:** thà UNKNOWN còn hơn đoán sai (đèn nhỏ/tối/ngược sáng).
- **Interface-driven:** cùng `TrafficLightDetector` ABC → Phase 8 thay bằng CNN không đụng pipeline.

---

## 7. File thay đổi / tạo mới

**Sửa:** `src/drivevision/perception/traffic_light.py` (đầy đủ), `src/drivevision/viz/
annotator.py` (vẽ TL + HUD Lights), `src/drivevision/pipeline/builder.py` (truyền config),
`configs/default.yaml` + `src/drivevision/config.py` (bật TL + hsv keys), `EVALUATION.md`.

**Tạo mới:** `tests/test_traffic_light.py` (15), `configs/phase4.yaml`, `finish_phase_4.md`.

**Dữ liệu (gitignored):** `output/tl_best_*.jpg`, `output/tl_demo_*.jpg`.

**Không cài/tải gì mới.**

---

## 8. Definition of Done — đối chiếu `phase_4.md`

- [x] M1 `detect()` không còn `return []` · [x] M2 lọc đúng class 9
- [x] M3 phân loại đỏ/xanh đúng · [x] M4 vàng (HSV) · [x] M5 edge case → UNKNOWN không crash
- [x] M6 annotator vẽ box màu + label · [x] M7 pipeline end-to-end chạy · [x] M8 ≤2ms/đèn (0.10ms)
- [x] `__init__(config)`, `_safe_crop`, `_classify_state` (B→A), `_color_mask_classify`
  (2 dải H đỏ), `_region_based_classify` (crop<3 → UNKNOWN)
- [x] builder truyền config; annotator vẽ đúng màu/label; configs/phase4.yaml
- [x] Tests color/region/e2e/config đều pass — **66 passed tổng**
- [x] KHÔNG sửa types/base/pipeline

---

## 9. Lệnh tham khảo

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_traffic_light.py
# pipeline đầy đủ giờ gồm cả traffic light (default.yaml)
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --source data/samples/sample.mp4 --save output/demo.mp4 --max-frames 300
```

---

## 10. Bước tiếp theo
- **Phase 5 — Scene Understanding:** gộp `detections`+`tracks`+`lanes`+`traffic_lights`
  → gán **ego-lane**, **lead vehicle**, liên kết đèn với hướng đi → nuôi Phase 6.
- **Phase 6 — Risk + Decision:** ra quyết định **STOP** khi `TrafficLightState.RED` phía trước.
- **Phase 8 — accuracy đèn** (LISA/BTTSD) + thay HSV bằng CNN tiny — Kaggle.
