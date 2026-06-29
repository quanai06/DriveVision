# DriveVision — Khung đánh giá (Evaluation)

> **Nguyên tắc:** ĐỪNG chỉ benchmark Detection. Mỗi thành phần có bản chất khác
> nhau nên phải đo bằng **đúng metric** của nó. File này định nghĩa: đo gì, đo
> bằng công cụ nào, cần dữ liệu/nhãn gì, ở phase nào — và bảng kết quả để điền.

---

## 0. Hai loại đánh giá

| Loại | Đo gì | Cần nhãn (ground-truth)? | Khi nào |
|---|---|---|---|
| **Tốc độ (Speed)** | FPS, độ trễ từng stage | ❌ Không | Mọi phase (P1+) |
| **Độ chính xác (Accuracy)** | mAP, MOTA, IoU... | ✅ Có | Khi có dataset nhãn (P8) |

- **Speed** → `scripts/benchmark.py` (đã có, chạy ngay).
- **Accuracy** → từng thành phần dùng công cụ riêng (mục 2–7 bên dưới).

---

## 1. Tốc độ — `scripts/benchmark.py`

Đo FPS tổng + độ trễ (ms) **từng stage** để biết điểm nghẽn.

```bash
# trên video thật
python scripts/benchmark.py --source data/samples/sample.mp4 --frames 200
# không có video -> frame ngẫu nhiên
python scripts/benchmark.py --frames 100 --imgsz 320
# in 1 dòng markdown để dán vào bảng dưới
python scripts/benchmark.py --source x.mp4 --md --label "OpenVINO 320"
```

### Bảng kết quả tốc độ

> Đo ngày 2026-06-29 | CPU (torch 2.12.1+cpu, CUDA=False) | YOLOv8n | nguồn:
> `data/samples/sample.mp4` — dashcam cao tốc Udacity (1280x720) | 150 frames + 10 warmup |
> lệnh: `python scripts/benchmark.py --source data/samples/sample.mp4 --frames 150 --imgsz <N>`

| Cấu hình | imgsz | Detection (ms) | Tổng (ms) | FPS |
|---|---|---|---|---|
| PyTorch CPU | 640 | 50.3 | 51.7 | 19.4 |
| PyTorch CPU | 320 | 21.0 | 22.2 | 45.1 |
| OpenVINO | 640 | _ | _ | _ |
| OpenVINO | 320 | _ | _ | _ |

> **Nhận xét (Phase 1):** Detection chiếm ~98% thời gian (annotate chỉ ~0.8 ms);
> mọi stage khác (tracking/scene/risk/decision/annotate) gộp lại < 1 ms. Giảm
> imgsz 640→320 tăng ~2.2× FPS. OpenVINO để dành Phase 8/10 (phần cứng Intel).
>
> Mục tiêu: cho thấy hiểu trade-off tốc độ ↔ độ chính xác, và tối ưu OpenVINO cho phần cứng Intel.

---

## 2. Object Detection (Phase 1, đo accuracy ở Phase 8)

| Metric | Ý nghĩa | Khoảng | Cần nhãn? | Công cụ |
|---|---|---|---|---|
| **mAP@50** | Độ chính xác trung bình ở IoU≥0.5 | 0–1 (cao = tốt) | ✅ | `ultralytics model.val()` |
| **mAP@50-95** | Trung bình mAP ở IoU 0.5→0.95 (khắt khe hơn) | 0–1 | ✅ | ultralytics |
| **Precision** | Trong các box dự đoán, bao nhiêu % đúng | 0–1 | ✅ | ultralytics |
| **Recall** | Trong các vật thật, bắt được bao nhiêu % | 0–1 | ✅ | ultralytics |
| **F1** | Cân bằng Precision/Recall | 0–1 | ✅ | ultralytics |
| **FPS / latency** | Tốc độ | — | ❌ | benchmark.py |

**Dữ liệu:** BDD100K (hoặc KITTI) tập `val`.
**Cách đo:** `from ultralytics import YOLO; YOLO("weights.pt").val(data="bdd.yaml")`
→ trả mAP per-class + confusion matrix + PR curve sẵn.

### Baseline NGAY — pretrained, chạy LOCAL trên CPU, CHƯA finetune

> Mục đích: biết model HIỆN TẠI (YOLOv8n pretrained, chưa train gì) chính xác đến
> đâu trên cảnh lái xe thật. Đo được **local CPU**, lặp lại mỗi phase để theo dõi.
> (Finetune để cải thiện là việc RIÊNG, chạy trên **Kaggle GPU** ở Phase 8.)

Khi đã có BDD100K val (ảnh + nhãn `det_val.json`) trong `data/bdd100k/`:

```bash
# 1) nhãn BDD val -> YOLO txt theo id COCO (vì model pretrained xuất id COCO)
python scripts/convert_bdd_to_yolo.py \
    --labels data/bdd100k/labels/det_20/det_val.json \
    --out    data/bdd100k/labels/val --target coco   # ultralytics tìm nhãn ở /labels/val/
# 2) đo baseline trên CPU (KHÔNG train)
python scripts/eval_accuracy.py \
    --weights models/weights/yolo.pt --data configs/bdd_val_coco.yaml \
    --imgsz 640 --device cpu
```

| Model | Dataset | imgsz | mAP@50 | mAP@50-95 | Precision | Recall | Ngày |
|---|---|---|---|---|---|---|---|
| YOLOv8n pretrained (baseline) | BDD100K val (10k ảnh, 149,969 box) | 640 | **0.225** | **0.128** | 0.371 | 0.248 | 2026-06-29 |

Per-class mAP@50 (zero-shot COCO→BDD, CPU i7-1355U, ~14 phút):

| car | person | bus | truck | bicycle | motorcycle | traffic light |
|---|---|---|---|---|---|---|
| 0.488 | 0.363 | 0.292 | 0.238 | 0.182 | 0.136 | 0.102 |

> **Đọc số này:** mAP@50 ~0.23 là **thấp** — đúng như kỳ vọng khi lấy model học trên
> COCO (ảnh tổng quát) đem chạy thẳng lên cảnh lái xe BDD (vật nhỏ ở xa, đêm/mưa,
> quy ước nhãn khác). `car` tốt nhất (0.49), `traffic light` tệ nhất (0.10 — vật rất
> nhỏ). Đây chính là **lý do cần finetune** (Phase 8, Kaggle GPU) → đo lại để thấy Δ.

> Lưu ý: model detect KHÔNG đổi từ Phase 1→7, nên con số này ổn định cho tới khi
> finetune (P8). Đo 1 lần ở P1 là đủ làm mốc; đo lại sau finetune để thấy Δ.

### Bảng so sánh baseline vs fine-tuned (Phase 8 — Kaggle GPU)

| Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| Pretrained (baseline) | _ | _ | _ | _ |
| Fine-tuned (của bạn) | _ | _ | _ | _ |
| **Cải thiện (Δ)** | _ | _ | _ | _ |

---

## 3. Multi-Object Tracking (Phase 2)

| Metric | Ý nghĩa | Khoảng | Cần nhãn? |
|---|---|---|---|
| **MOTA** | Multiple Object Tracking Accuracy — gộp lỗi miss + false positive + ID switch | -∞→1 (cao=tốt) | ✅ |
| **IDF1** | F1 trên việc giữ ĐÚNG danh tính (ID) qua thời gian | 0–1 | ✅ |
| **HOTA** | Higher Order Tracking Accuracy — cân bằng detection & association (chuẩn hiện đại) | 0–1 | ✅ |
| **MT / ML** | % track theo được hầu hết / mất hầu hết | 0–1 | ✅ |
| **ID switches** | Số lần đổi nhầm ID | đếm (thấp=tốt) | ✅ |
| **FPS** | Tốc độ tracker | — | ❌ |

**Dữ liệu nhãn tracking:** BDD100K MOT, hoặc MOT17/MOT20 (định dạng MOTChallenge).
**Công cụ:**
- [`TrackEval`](https://github.com/JonathonLuiten/TrackEval) — chuẩn chính thức cho HOTA/MOTA/IDF1.
- [`py-motmetrics`](https://github.com/cheind/py-motmetrics) — nhẹ hơn, tính MOTA/IDF1.

**Đánh giá nhanh không có nhãn (Phase 2):** đếm ID-switch bằng mắt trên video, kiểm tra ID có ổn định khi vật bị che một phần (unit test chuỗi giả lập trong `tests/test_tracking.py`).

### Bảng kết quả tracking

| Tracker | MOTA | IDF1 | HOTA | ID-sw | FPS |
|---|---|---|---|---|---|
| SimpleTracker (IoU) | _ | _ | _ | _ | _ |
| ByteTrack | _ | _ | _ | _ | _ |

---

## 4. Lane Detection (Phase 3)

| Metric | Ý nghĩa | Khoảng | Cần nhãn? |
|---|---|---|---|
| **IoU** (pixel) | Giao/Hợp giữa vùng làn dự đoán và nhãn (cho hướng segmentation) | 0–1 | ✅ |
| **Accuracy** | % điểm/pixel làn phân loại đúng | 0–1 | ✅ |
| **Precision / Recall / F1** | Theo pixel hoặc theo điểm làn | 0–1 | ✅ |
| **TuSimple Accuracy** | Metric chuẩn của benchmark TuSimple (theo điểm) | 0–1 | ✅ |
| **CULane F1** | Metric chuẩn CULane (IoU≥0.5 theo dải làn) | 0–1 | ✅ |

**Dữ liệu nhãn lane:** BDD100K lane, TuSimple, CULane.
**Công cụ:** tự viết IoU/accuracy trên mask (numpy), hoặc dùng eval script chính thức của TuSimple/CULane.
**Đánh giá nhanh (classical, Phase 3):** trực quan — vạch vẽ có bám đúng làn không; ổn định qua frame (đỡ rung) nhờ smoothing.

### Bảng kết quả lane

| Phương pháp | IoU | Accuracy | F1 |
|---|---|---|---|
| Classical (OpenCV) | _ | _ | _ |
| Model (YOLOP/UFLD) | _ | _ | _ |

---

## 5. Traffic Light — phát hiện + nhận màu (Phase 4)

| Metric | Ý nghĩa | Cần nhãn? |
|---|---|---|
| **Detection mAP** | Phát hiện hộp đèn (nếu dùng model riêng) | ✅ |
| **Classification Accuracy** | % phân loại đúng màu RED/YELLOW/GREEN | ✅ |
| **Per-class accuracy** | Đúng theo từng màu (đỏ thường quan trọng nhất) | ✅ |
| **Confusion matrix** | Nhầm lẫn giữa các màu | ✅ |

**Dữ liệu:** tự gán nhãn một bộ nhỏ (vài trăm crop đèn cắt từ video) — đủ để đo accuracy màu.
**Công cụ:** `sklearn.metrics` (accuracy_score, confusion_matrix).

### Bảng kết quả đèn

| Phương pháp | Accuracy | Acc(RED) | Acc(YELLOW) | Acc(GREEN) |
|---|---|---|---|---|
| HSV heuristic | _ | _ | _ | _ |
| CNN nhỏ | _ | _ | _ | _ |

---

## 6. Scene Understanding (Phase 5) — đánh giá LOGIC

Không có "metric chuẩn ngành"; đánh giá bằng **độ đúng của suy luận** trên tình huống kiểm thử:

| Tiêu chí | Cách đo | Cần nhãn? |
|---|---|---|
| Gán ego-lane đúng | % object gán đúng làn trên bộ test tự dựng | ✅ (tự gán) |
| Chọn đúng lead vehicle | Unit test cảnh giả lập (xe gần nhất trong ego-lane) | ❌ |
| Liên kết đèn đúng | Unit test | ❌ |
| Đếm vật thể đúng | So với số detection | ❌ |

**Công cụ:** `tests/test_scene.py` (pytest) trên SceneState giả lập.

---

## 7. Risk + Decision (Phase 6) — đánh giá LOGIC theo kịch bản

Cũng không có metric ngành; đánh giá bằng **scenario-based tests** (đúng/sai theo kỳ vọng an toàn):

| Kịch bản | Kỳ vọng | Cần nhãn? |
|---|---|---|
| bbox phình nhanh trong ego-lane | TTC giảm → WARNING → DANGER → BRAKE | ❌ |
| đèn đỏ phía trước | Decision = STOP | ❌ |
| vật đứng yên ngoài làn | SAFE / MAINTAIN | ❌ |
| nhiều vật nguy hiểm | lấy mức rủi ro cao nhất | ❌ |

**Chỉ số tổng hợp:** *Scenario pass rate* = số kịch bản đúng / tổng kịch bản (mục tiêu 100%).
**Công cụ:** `tests/test_risk.py`, `tests/test_decision.py`.

> Lưu ý: TTC bằng "looming" (P6) là ước lượng; chỉ khi có **depth từ CARLA (P9)** mới đo được **sai số TTC tính bằng giây/mét** so với ground-truth thật.

---

## 8. Hệ thống tổng thể (Phase 7 + 10)

| Metric | Ý nghĩa | Công cụ |
|---|---|---|
| **End-to-end FPS** | Toàn pipeline | benchmark.py |
| **Latency p50 / p95** | Độ trễ trung vị & đuôi | benchmark.py (mở rộng) |
| **Bộ nhớ (RAM)** | Mức tiêu thụ | `tracemalloc` / `psutil` |
| **API latency** | Thời gian phản hồi `/api/scene` | `locust` / `ab` / TestClient |
| **Test coverage** | % code có test | `pytest --cov` |

---

## 9. Bảng tổng hợp toàn dự án (đưa vào README ở Phase 10)

| Thành phần | Metric chính | Kết quả | Dataset |
|---|---|---|---|
| Detection | mAP@50 | _ | BDD100K |
| Tracking | HOTA / IDF1 | _ | BDD100K MOT |
| Lane | IoU | _ | BDD100K lane |
| Traffic light | Accuracy | _ | tự gán |
| Risk/Decision | Scenario pass rate | _ | kịch bản tự dựng |
| Hệ thống | FPS (OpenVINO) | _ | — |

---

## 10. Bản đồ "phase → đánh giá"

| Phase | Đánh giá ngay được (không nhãn) | Đánh giá đầy đủ (cần nhãn, làm ở P8+) |
|---|---|---|
| 1 Detection | FPS, trực quan box | mAP/P/R |
| 2 Tracking | FPS, ID-switch bằng mắt, unit test | MOTA/IDF1/HOTA |
| 3 Lane | trực quan, ổn định | IoU/Accuracy |
| 4 Traffic light | trực quan | Accuracy màu (bộ nhỏ) |
| 5 Scene | unit test logic | ego-lane accuracy |
| 6 Risk/Decision | scenario tests | sai số TTC (cần depth, P9) |
| 7 API | latency, functional | load test |
| 8 Fine-tune | — | ⭐ mAP baseline vs fine-tuned |
| 9 CARLA | — | sai số khoảng cách (depth) |
| 10 Polish | bảng tổng hợp + coverage | — |

---

## 11. Tái lập (Reproducibility)

- Cố định seed, ghi version thư viện, lưu config dùng để đo.
- Mỗi lần đo ghi rõ: phần cứng (CPU/GPU), imgsz, model, backend (PyTorch/OpenVINO).
- Lưu kết quả thô (CSV/JSON) cạnh bảng tổng hợp.
