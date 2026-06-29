# Phase 1 — Báo cáo hoàn thành (Object Detection + Visualization + Output + Benchmark)

> **Ngày thực hiện:** 2026-06-29
> **Trạng thái:** ✅ HOÀN THÀNH (code + demo + speed benchmark + accuracy baseline)
> **Máy:** Linux, Python 3.12.7, CPU Intel i7-1355U (không GPU CUDA)
> File này ghi lại **toàn bộ** những gì đã làm trong Phase 1, theo đúng thứ tự,
> kèm kết quả thật và lệnh tái lập. Đọc kèm `phase_1.md` (đặc tả) và `EVALUATION.md`.

---

## 0. Tóm tắt nhanh (TL;DR)

Phase 1 biến skeleton thành demo chạy được: đọc video lái xe → YOLOv8n phát hiện
vật thể → vẽ annotation → xuất video + đo FPS. Ngoài đặc tả gốc, còn làm thêm:

- **Speed benchmark** trên video thật (FPS + ms từng stage).
- **Accuracy baseline** của YOLOv8n pretrained trên **BDD100K val (10k ảnh)** chạy
  local trên CPU — để biết model "chưa làm gì" thì chính xác đến đâu trên cảnh lái xe.

**Con số chốt:**
| Hạng mục | Kết quả |
|---|---|
| Unit/integration tests | **28 passed** (0 failed) |
| Demo FPS (dashcam, imgsz640, CPU) | **~16–19 FPS** |
| Speed benchmark imgsz640 / imgsz320 | **19.4 / 45.1 FPS** |
| Accuracy baseline BDD100K val | **mAP@50 = 0.225**, mAP@50-95 = 0.128, P = 0.371, R = 0.248 |

---

## 1. Môi trường & cài đặt

### 1.1 Trạng thái ban đầu
- Dùng virtualenv `.venv/` (Python 3.12.7). Chạy code bằng `PYTHONPATH=src` (package
  không cài editable; các script trong `scripts/` tự `sys.path.insert` thư mục `src`).
- `.venv` ban đầu chỉ có: `opencv-python-headless 4.13`, `numpy 2.5.0`, `pyyaml 6.0.3`,
  `pytest 9.1.1`. **Thiếu** `ultralytics`, `torch`.

### 1.2 Đã cài thêm (sau khi xác nhận với user)
```bash
.venv/bin/pip install --timeout 300 --retries 5 torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --timeout 300 --retries 5 "ultralytics>=8.2"
```
Kết quả: **`torch 2.12.1+cpu`**, **`ultralytics 8.4.82`**, `cuda=False`.
> Lần đầu tải torch CPU bị timeout do wheel lớn + CDN chậm → cài lại với `--timeout 300 --retries 5` thì xong.

### 1.3 Lưu ý quan trọng về môi trường
- **OpenCV là bản `headless`** → `cv2.imshow` / `--display` **KHÔNG chạy** (không có
  GUI). `cli.py` bắt `cv2.error` và cảnh báo thay vì crash. Demo dùng `--save` (ghi mp4).
  Muốn cửa sổ live thì cần `pip install opencv-python` (bản non-headless).
- **Không có GPU** → mọi inference chạy CPU. Phase 1 không yêu cầu real-time nên ổn.

---

## 2. Code đã implement

### 2.1 `src/drivevision/viz/annotator.py` (từ stub → đầy đủ)
- Bảng màu **12 màu BGR** phân biệt, ánh xạ theo `class_id` (`_class_color`).
- `_draw_label()` — vẽ nhãn `"{class_name} {conf:.2f}"` trên nền tối; tự lật xuống
  dưới nếu box sát mép trên.
- `_draw_hud()` — HUD góc trên-trái: `DriveVision FPS: x` + `Frame: n Dets: k`.
- `Annotator.draw(result, fps=0.0) -> np.ndarray` — vẽ lên **bản copy** của frame
  (không sửa ảnh gốc), trả về BGR cùng shape.
- **Chữ ký `draw(result, fps=)` giữ ổn định** để Phase 2 (tracking) thêm track-id
  mà không phải đổi interface.

### 2.2 `src/drivevision/cli.py` (từ stub → vòng lặp đầy đủ)
- CLI args mới: `--save`, `--display` / `--no-display`, `--max-frames`, `--conf`
  (kèm `--config`, `--source` cũ).
- `_FPSCounter` — FPS theo **sliding window** (`deque(maxlen=30)`).
- `_make_video_writer()` — VideoWriter **lazy** (khởi tạo sau khi biết kích thước frame
  đầu), **codec fallback**: thử lần lượt `mp4v → avc1 → XVID → MJPG` (đổi đuôi `.avi`
  khi cần), tự `mkdir -p` thư mục cha.
- Vòng lặp: `process → draw → write/show`, log mỗi `log_interval` frame, **headless-safe**
  (gặp lỗi imshow thì tắt display, không crash), `try/except KeyboardInterrupt/finally`
  đảm bảo `writer.release()`.
- In tóm tắt cuối: tổng frame, tổng detection, FPS trung bình, thời gian.

### 2.3 Config — đồng bộ 2 nơi
- `src/drivevision/config.py` `DEFAULT_CONFIG` + `configs/default.yaml` thêm:
  - `output.window_name: DriveVision`
  - `logging: { level: INFO, log_interval: 30 }`
  - `perception.detection.device: null` (auto CPU/CUDA; chưa dùng ở P1, chuẩn bị P8)

### 2.4 `tests/test_phase1.py` (mới, 24 test)
Nhóm test: `BoundingBox` (as_int/iou/area/center), `Annotator` (shape, không sửa ảnh
gốc, vẽ ra pixel, rỗng, 20 box, box sát mép, fps=0), `YOLODetector` (mock parse boxes),
`Pipeline` end-to-end (no-detector, mock detector, tích hợp Annotator), `_FPSCounter`,
`Config` (load default, yaml-defaults đồng bộ, get_path).
> Đã **sửa lỗi typo** trong skeleton của `phase_1.md` (`bb.iou` → `a.iou` ở test no-overlap).

### 2.5 `perception/detection.py`, `pipeline/*`, `io/*`, `types.py`
**KHÔNG sửa** — chỉ dùng. Xác nhận `YOLODetector` (lazy import ultralytics, fallback
`yolov8n.pt`) chạy đúng end-to-end.

---

## 3. Kiểm thử

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
# 28 passed in ~0.2s   (2 test cũ test_pipeline.py + 24 test_phase1.py + ... )
```
Chạy pass **cả khi chưa cài ultralytics** (test detector dùng mock/`importorskip`) lẫn
sau khi cài. Đã verify **graceful degradation**: gỡ ultralytics → pipeline vẫn chạy,
`detector=None`, 0 detection, không crash.

---

## 4. Dữ liệu — quá trình tải (có vài lần đổi nguồn)

> Nguyên tắc: **xác nhận với user trước khi tải**. Tải về thư mục `data/`.

### 4.1 Video mẫu — đi qua 3 nguồn
1. **BDD100K clip** (`dl.yf.io`, theo `phase_1.md`): ❌ **host chết** (timeout) → file 0 byte.
2. **Intel sample-videos** `person-bicycle-car-detection.mp4` (~6MB): ✅ tải được, có
   person/bicycle/car — NHƯNG user nhận xét đúng: đó là **camera giám sát bãi đỗ** (góc
   cao, cố định), **không giống lái xe**.
3. **Udacity `project_video.mp4`** (`github.com/udacity/CarND-Advanced-Lane-Lines`,
   ~25MB): ✅ **dashcam cao tốc thật** (1280×720, 1260 frame) → chốt làm `data/samples/sample.mp4`.

Ngoài ra tự tạo `data/samples/synthetic.mp4` (ô vuông di chuyển, OpenCV) để smoke-test
không cần ML deps.

### 4.2 Weights YOLO
`models/weights/yolo.pt` = `yolov8n.pt` (~6MB, auto-download từ ultralytics assets).

### 4.3 BDD100K (cho accuracy) — xem mục 7
User tự tải full BDD100K trên Kaggle → `~/Downloads/archive.zip` (8.16GB).

---

## 5. Demo (detection thật)

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 --save output/demo.mp4 --max-frames 300
```
**Kết quả:**
- `output/demo.mp4` — 300 frame, **294 detection**, **~16.5 FPS** (CPU).
- `output/demo_best_frame.jpg` — khung đông nhất: 3 ô tô (car 0.71 / 0.64 / 0.41) +
  HUD góc trên-trái.
- Lớp phát hiện trên clip: **car (290), truck (3), person (1)** — đúng cảnh cao tốc.

Hình ảnh annotation đúng thiết kế: box màu theo class (car = xanh dương, person = đỏ,
bicycle = xanh lá), nhãn `class conf`, HUD FPS/Frame/Dets.

---

## 6. Speed Benchmark — `scripts/benchmark.py`

Đo **tốc độ** (không cần nhãn). Trên dashcam Udacity (1280×720), 150 frame + 10 warmup, CPU:

| Cấu hình | imgsz | Detection (ms) | Tổng (ms) | FPS |
|---|---|---|---|---|
| PyTorch CPU | 640 | 50.3 | 51.7 | **19.4** |
| PyTorch CPU | 320 | 21.0 | 22.2 | **45.1** |

**Nhận xét:** Detection chiếm **~98%** thời gian; `annotate` chỉ ~0.8–1.3ms; mọi stage
khác (tracking/scene/risk/decision) < 0.1ms. Giảm imgsz 640→320 nhanh **~2.3×** (đổi lại
accuracy giảm). OpenVINO để dành Phase 8/10.

```bash
.venv/bin/python scripts/benchmark.py --source data/samples/sample.mp4 --frames 150 --imgsz 640
.venv/bin/python scripts/benchmark.py --source data/samples/sample.mp4 --frames 150 --imgsz 320
```

---

## 7. Accuracy Baseline trên BDD100K (điểm nhấn của session)

### 7.1 Vì sao cần & những hiểu lầm đã làm rõ
- `benchmark.py` chỉ đo **tốc độ**. **Độ chính xác (mAP)** cần **nhãn ground-truth** →
  không đo được trên video demo (không nhãn).
- **COCO không phải dataset tự lái.** Ban đầu thử đo trên COCO val2017 (~1GB) để minh
  họa cơ chế, nhưng user chỉ ra đúng: COCO là dataset **vật thể tổng quát**, không phải
  cảnh lái xe → **đã xóa COCO** (`data/coco` + `configs/coco_val.yaml`).
- Chốt: đo **baseline của model pretrained (chưa train gì) trên BDD100K**, chạy **local
  CPU**, lặp lại được mỗi phase. **Finetune** là việc riêng → **chỉ Kaggle (Phase 8)**.

### 7.2 Nguồn BDD100K
- Mirror chính thức không tải được từ máy này: Berkeley `dl.yf.io` chết; ETH
  `dl.cv.ethz.ch` bị **chặn DNS**. HF chỉ có mirror định dạng lạ.
- → User tải full BDD100K trên **Kaggle** → `~/Downloads/archive.zip` (**8.16GB**,
  cấu trúc `bdd100k/bdd100k/images/100k/{train,val,test}` + `bdd100k_labels_release/...
  /bdd100k_labels_images_{train,val}.json`).

### 7.3 Xử lý dữ liệu
1. Giải nén **chỉ tập val**: 10,000 ảnh → `data/bdd100k/images/val/` + nhãn
   `bdd100k_labels_images_val.json` (định dạng cũ).
2. Convert nhãn BDD JSON → YOLO txt bằng `scripts/convert_bdd_to_yolo.py --target coco`:
   - Vì model pretrained xuất **id COCO** nên nhãn GT phải map sang id COCO mới so đúng.
   - Alias: `person→pedestrian`, `bike→bicycle`, `motor→motorcycle`.
   - Kết quả: **10,000 ảnh, 149,969 box**; bỏ **35,557 box** lớp `rider`/`traffic sign`
     (không có trong COCO) — đây là cách đo zero-shot COCO→BDD chuẩn.
3. Set `ultralytics settings datasets_dir = /home/quan/DriveVision` để path tương đối
   trong yaml resolve đúng.
4. (Sửa lỗi nhỏ) ultralytics bắt buộc có key `train` trong data yaml ngay cả khi chỉ
   val → thêm `train: images/val` (placeholder) vào `configs/bdd_val_coco.yaml`.

### 7.4 Lệnh đo
```bash
# 1) convert nhãn (đã chạy)
.venv/bin/python scripts/convert_bdd_to_yolo.py \
    --labels <...>/bdd100k_labels_images_val.json \
    --out data/bdd100k/labels/val --target coco

# 2) đo baseline trên CPU (~14 phút cho 10k ảnh)
PYTHONPATH=src .venv/bin/python scripts/eval_accuracy.py \
    --weights models/weights/yolo.pt --data configs/bdd_val_coco.yaml \
    --imgsz 640 --device cpu
```

### 7.5 KẾT QUẢ — YOLOv8n pretrained zero-shot trên BDD100K val (10k ảnh, 149,969 box)

| Chỉ số | Giá trị |
|---|---|
| **mAP@50** | **0.225** |
| **mAP@50-95** | **0.128** |
| Precision | 0.371 |
| Recall | 0.248 |
| Tốc độ val | ~41ms inference/ảnh (CPU) |

**Per-class mAP@50:**
| car | person | bus | truck | bicycle | motorcycle | traffic light |
|---|---|---|---|---|---|---|
| 0.488 | 0.363 | 0.292 | 0.238 | 0.182 | 0.136 | 0.102 |

Chi tiết (PR curve, confusion matrix): `runs/detect/val-2/`.

### 7.6 Diễn giải
- **mAP@50 ~0.23 là thấp** — đúng kỳ vọng: model học trên COCO (ảnh tổng quát) chạy
  thẳng lên cảnh lái xe (vật nhỏ ở xa, đêm/mưa, góc dashcam, quy ước nhãn khác).
- `car` tốt nhất (0.49 — to, rõ), `traffic light` tệ nhất (0.10 — vật rất nhỏ).
- **Recall thấp (0.248)** vì BDD gán nhãn cả vật rất nhỏ/bị che mà model COCO bỏ sót.
- → Đây là **mốc (baseline)** để chứng minh giá trị của **finetune ở Phase 8**: train
  tiếp trên ảnh BDD (đúng miền lái xe) rồi đo lại bằng đúng tool này để thấy Δ.

### 7.7 Khái niệm cốt lõi đã chốt
> **COCO dạy model NHẬN DẠNG VẬT (car/person/... là gì). BDD100K dạy model NHẬN DẠNG
> VẬT TRONG LÚC LÁI XE.** Vấn đề không phải "chưa từng thấy ô tô" mà là **lệch miền
> (domain shift)** + khác quy ước nhãn.

---

## 8. Bộ công cụ mới (tái dùng cho Phase 8)

| File | Vai trò |
|---|---|
| `scripts/eval_accuracy.py` | `model.val()` → mAP@50, mAP@50-95, P, R (+ per-class giao thông). Mặc định CPU. Có `--md`. |
| `scripts/convert_bdd_to_yolo.py` | BDD det JSON → YOLO txt. `--target coco` (baseline pretrained) / `--target bdd` (10-class native, cho model đã finetune). |
| `configs/bdd_val_coco.yaml` | Dataset config id-COCO — dùng đo baseline pretrained. |
| `configs/bdd_det.yaml` | Dataset config 10-class native — để dành sau finetune. |

---

## 9. Tất cả file đã tạo / sửa

**Tạo mới:**
- `src` không tạo file mới (sửa stub).
- `tests/test_phase1.py`
- `scripts/eval_accuracy.py`
- `scripts/convert_bdd_to_yolo.py`
- `configs/bdd_val_coco.yaml`, `configs/bdd_det.yaml`
- `finish_phase_1.md` (file này)
- `data/samples/synthetic.mp4`, `models/weights/.gitkeep`, `output/.gitkeep`, `data/samples/.gitkeep`

**Sửa:**
- `src/drivevision/viz/annotator.py` (stub → đầy đủ)
- `src/drivevision/cli.py` (stub → vòng lặp đầy đủ)
- `src/drivevision/config.py` (`DEFAULT_CONFIG` thêm output/logging/device)
- `configs/default.yaml` (window_name, logging, device)
- `EVALUATION.md` (điền bảng speed + bảng accuracy baseline + per-class)

**Dữ liệu (gitignored — không commit):**
- `data/samples/sample.mp4` (dashcam Udacity, 25MB)
- `models/weights/yolo.pt` (yolov8n, 6.5MB)
- `data/bdd100k/` (val 10k ảnh + nhãn YOLO, ~618MB)
- `output/demo.mp4`, `output/demo_best_frame.jpg`, `output/smoke.mp4`
- `runs/detect/val-2/` (kết quả val của ultralytics)

> `.gitignore` đã bỏ qua `data/`, `output/`, `*.pt`, `*.onnx` từ trước.

---

## 10. Definition of Done — đối chiếu `phase_1.md`

**Chức năng cốt lõi**
- [x] `run_pipeline.py --source ... --save output/demo.mp4` chạy không crash
- [x] `output/demo.mp4` tồn tại, có box + nhãn màu
- [x] Log cuối in FPS trung bình dương
- [x] Box bao đúng đối tượng (car/person/truck)
- [x] Nhãn `{class} {conf:.2f}` (vd "car 0.71")
- [x] HUD góc trên-trái FPS + số detection

**Cấu hình**
- [x] `--save`, `--display/--no-display`, `--max-frames`, `--conf` hoạt động
- [x] `output.save_path`, `output.display` trong YAML hoạt động

**Độ bền**
- [x] Không crash khi thiếu `ultralytics` (chỉ warning)
- [x] Không crash khi `save_path=null`
- [x] Ctrl+C dừng sạch, VideoWriter release (khối `finally`)
- [x] Video 0/1 frame không crash

**Chất lượng code**
- [x] `pytest -q` — **0 failed** (28 passed)
- [x] `test_phase1.py` ≥ 15 test (có 24)
- [x] Type hints + docstring public

**Mở rộng Phase 2**
- [x] `Annotator.draw(result, fps=)` không cần đổi chữ ký
- [x] `Pipeline.process()` / `PipelineResult` không bị sửa

**Mục tiêu đo được (M1–M7)**
- [x] M1 pipeline end-to-end (0 exception)
- [x] M2 phát hiện ≥6 class giao thông (car/person/truck/bus/motorcycle/bicycle)
- [x] M3 xuất video annotated xem được
- [x] M4 FPS in ra stdout (dương)
- [x] M5 graceful khi thiếu ultralytics
- [x] M6 test suite pass
- [~] M7 `--display`: **bị chặn bởi OpenCV headless** (không có GUI trên máy này) — code
  đã xử lý an toàn; cần bản `opencv-python` non-headless mới mở cửa sổ.

**Ngoài đặc tả (làm thêm)**
- [x] Speed benchmark trên video thật (điền `EVALUATION.md`)
- [x] Accuracy baseline YOLOv8n trên BDD100K val (10k) — local CPU

---

## 11. Lệnh tham khảo nhanh

```bash
# Test
PYTHONPATH=src .venv/bin/python -m pytest -q

# Demo (lưu video)
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py \
    --source data/samples/sample.mp4 --save output/demo.mp4 --max-frames 300

# Speed benchmark
.venv/bin/python scripts/benchmark.py --source data/samples/sample.mp4 --frames 150 --imgsz 640

# Accuracy baseline (BDD100K val, CPU)
PYTHONPATH=src .venv/bin/python scripts/eval_accuracy.py \
    --weights models/weights/yolo.pt --data configs/bdd_val_coco.yaml --device cpu
```

---

## 12. Bước tiếp theo

- **Phase 2 — Multi-Object Tracking:** cắm `SimpleTracker`/ByteTrack vào pipeline; vẽ
  track-id (chữ ký `Annotator.draw` đã sẵn sàng, không cần đổi).
- **Phase 8 — Fine-tuning (Kaggle GPU):** train YOLOv8 trên BDD100K (dùng
  `configs/bdd_det.yaml`, `--target bdd`), xuất `best.pt` → `models/weights/`, rồi đo
  lại bằng `eval_accuracy.py` để so **baseline (0.225) vs fine-tuned**.
- (Tùy chọn) Commit Phase 1: `feat(phase1): detection + viz + output + speed/accuracy benchmark`.
