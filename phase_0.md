# Phase 0 — Nền tảng & Bản đồ lộ trình (ĐÃ HOÀN THÀNH)

**Mục tiêu:** Khóa kiến trúc, dựng bộ khung (skeleton) modular chạy được end-to-end, và định nghĩa toàn bộ lộ trình từ skeleton → project portfolio hoàn chỉnh.

> File này là **chỉ mục (index)** cho toàn bộ đặc tả. Mỗi phase tiếp theo nằm trong `phase_<N>.md` với N tăng dần. Đọc file này trước để nắm bức tranh tổng, sau đó đi sâu từng phase.

---

## 1. DriveVision là gì

**DriveVision — High-Performance Autonomous Driving Perception Platform**: một nền tảng *perception* (nhận thức môi trường) cho xe tự hành, xử lý ảnh/video đầu vào và sinh ra hiểu biết về cảnh đường: vật thể, làn đường, đèn giao thông, theo vết đối tượng, đánh giá rủi ro và gợi ý quyết định — kèm dashboard trực quan.

Đây là **project portfolio cá nhân**, nên tiêu chí thành công cuối cùng là:
- Kiến trúc sạch, modular, dễ đọc, dễ mở rộng.
- Chạy được thật, demo trực quan ấn tượng (video annotated + dashboard).
- Có đo đạc (metrics), có tài liệu, có thể tái lập (reproducible).

---

## 2. Các ràng buộc đã chốt (rất quan trọng — mọi phase phải tuân thủ)

| Yếu tố | Quyết định | Hệ quả |
|---|---|---|
| Ngôn ngữ | Python 3.12 | Giữ venv hiện tại; CARLA live cần env riêng (xem Phase 9) |
| Tài nguyên | **Chỉ có Kaggle** (GPU T4/P100, giới hạn giờ) | Mọi training/fine-tune chạy trên Kaggle notebook; runtime chạy local trên video |
| Nguồn dữ liệu | Video/dataset thường trước (BDD100K/KITTI), CARLA sau | Thiết kế `DataSource` để hoán đổi nguồn |
| Model | Fine-tune từ pretrained, **không train from scratch**; baseline-first | Pretrained chạy trước → đo baseline → chỉ fine-tune khi cần |
| Hiệu năng | Kiến trúc sạch + accuracy khá + demo xịn | **Không** đu real-time 30FPS/TensorRT giai đoạn đầu |
| Dashboard | FastAPI + web nhẹ (REST + MJPEG stream) | Tách API lõi khỏi UI |

---

## 3. Kiến trúc tổng thể — hai mặt phẳng tách biệt

```
┌─ RESEARCH PLANE (Kaggle notebooks) ─────────────────────────┐
│   chuẩn bị data → fine-tune model → xuất weights (.pt/.onnx) │
│                         │                                    │
│                         ▼ (chỉ giao tiếp qua FILE WEIGHTS)   │
├─ RUNTIME PLANE (máy local, xử lý video) ────────────────────┤
│                                                              │
│  DataSource ─▶ Pipeline ─▶ Annotator ─▶ API / Dashboard     │
│  (video/CARLA)   │                                           │
│                  ├─ Detection ─┐                             │
│                  ├─ Tracking   │                             │
│                  ├─ Lane       ├─▶ SceneState ─▶ Risk ─▶ Decision
│                  ├─ TrafficLt  │                             │
│                  └─ Fusion ────┘                             │
└──────────────────────────────────────────────────────────────┘
```

**Nguyên tắc vàng:** hai plane chỉ trao đổi qua file weights trong `models/weights/`. Notebook không import runtime, runtime không import code training. Nhờ vậy bạn train trên Kaggle, tải `best.pt` về, thả vào thư mục, runtime tự nhận.

---

## 4. Cấu trúc thư mục (skeleton đã dựng)

```
src/drivevision/
  types.py        # HỢP ĐỒNG CHUNG: BoundingBox, Detection, Track, Lane(+LaneSide),
                  #   TrafficLight(+TrafficLightState), Frame, SceneState,
                  #   RiskLevel, ObjectRisk, RiskReport, Action, Decision, PipelineResult
  config.py       # load_config(path)->dict (deep-merge DEFAULT_CONFIG); get_path(cfg,"a.b.c")
  io/
    base.py            # DataSource (ABC): frames()->Iterator[Frame]
    video_source.py    # VideoSource — ĐÃ HOẠT ĐỘNG (OpenCV)
    carla_source.py    # CarlaSource — stub (Phase 9)
  perception/
    base.py            # ABC: Detector / Tracker / LaneDetector / TrafficLightDetector
    detection.py       # YOLODetector (ultralytics, lazy import) — có impl tham khảo
    tracking.py        # SimpleTracker (IoU) — có impl tham khảo
    lane.py            # ClassicalLaneDetector (impl) + ModelLaneDetector (stub)
    traffic_light.py   # SimpleTrafficLightDetector — stub
  fusion/fusion.py             # SensorFusion — stub (no-op)
  scene/scene_understanding.py # SceneBuilder.build(...) -> SceneState
  risk/risk_assessment.py      # RiskAssessor.assess(scene) -> RiskReport — stub
  decision/decision_support.py # DecisionSupport.decide(scene, risk) -> Decision — stub
  pipeline/
    pipeline.py        # Pipeline(stages...).process(frame)->PipelineResult; stage=None -> bỏ qua
    builder.py         # build_source(cfg), build_pipeline(cfg) — wiring theo config
  viz/annotator.py             # Annotator.draw(result)->np.ndarray (BGR) — stub
  api/
    server.py          # FastAPI app — stub
    stream.py          # PipelineRunner (thread + MJPEG) — stub
  cli.py               # entry: load config -> build -> loop frames
configs/default.yaml   scripts/run_pipeline.py   tests/   dashboard/index.html
notebooks/   models/weights/   data/samples/
```

Lệnh cơ bản:
```bash
PYTHONPATH=src pytest -q                                   # smoke test (không cần ML dep)
python scripts/run_pipeline.py --source data/samples/x.mp4 # chạy pipeline
```

---

## 5. "Hợp đồng chung" — `types.py` (đọc kỹ trước mọi phase)

Tất cả module nói chuyện với nhau qua các dataclass này. **Không sửa interface tùy tiện**; nếu cần thêm trường, thêm có chủ đích và cập nhật các phase liên quan.

- `BoundingBox(x1,y1,x2,y2)` + `.width/.height/.area/.center/.iou()/.as_int()`
- `Detection(bbox, class_id, class_name, confidence)`
- `Track(track_id, detection, age, time_since_update, history)` + `.velocity`
- `Lane(points, side: LaneSide, confidence)`
- `TrafficLight(bbox, state: TrafficLightState, confidence)`
- `Frame(index, timestamp, image, depth?, semantic?)` + `.width/.height`
- `SceneState(frame_index, timestamp, detections, tracks, lanes, traffic_lights)`
- `RiskReport(level: RiskLevel, score, object_risks, factors)`
- `Decision(action: Action, reason, confidence)`
- `PipelineResult(frame, scene, risk?, decision?)`

---

## 6. Bản đồ các Phase

| Phase | Tên | Mục tiêu rút gọn | Cần train? |
|---|---|---|---|
| 0 | Nền tảng & Roadmap | Khóa kiến trúc + skeleton chạy được | — |
| 1 | Object Detection + Visualization | Pretrained YOLO → video annotated (demo đầu tiên) | Không |
| 2 | Multi-Object Tracking | ID ổn định qua frame + quỹ đạo | Không |
| 3 | Lane Detection | Phát hiện làn đường (classical → model) | Tùy chọn |
| 4 | Traffic Light + State | Phát hiện đèn + phân loại màu | Tùy chọn |
| 5 | Scene Understanding | Gán object vào làn, lead vehicle, tổng hợp cảnh | Không |
| 6 | Risk Assessment + Decision | TTC, mức rủi ro, gợi ý hành động (rule-based) | Không |
| 7 | API + Dashboard | FastAPI + MJPEG stream + bảng rủi ro | Không |
| 8 | Data & Fine-tuning (Kaggle) | Baseline → fine-tune → đánh giá → swap weights | **Có** |
| 9 | CARLA Integration | Nguồn CARLA live + depth/semantic + fusion thật | Không |
| 10 | Portfolio Polish | README, demo, benchmark, CI/CD, deploy | — |

**Thứ tự khuyến nghị thực thi:** 1 → 2 → 3 → 4 → 5 → 6 → 7 (đã có một sản phẩm demo đầy đủ chạy trên pretrained). Sau đó 8 (nâng chất lượng model). 9 và 10 là mở rộng nâng cao. Mỗi phase đều để pipeline ở trạng thái **chạy được**, không có "khoảng chết".

---

## 7. Quy ước chung cho mọi phase

1. **Không phá interface** trong `*/base.py` và `types.py` — kế thừa và implement.
2. Mỗi stage độc lập: làm xong thì bật cờ tương ứng trong `configs/default.yaml`.
3. Mỗi phase phải có: test, cập nhật config (nếu cần), tiêu chí hoàn thành rõ ràng.
4. Commit theo phase, message dạng `feat(phaseN): ...`.
5. `grep -rn TODO src/` luôn cho biết phần còn dang dở.

---

## 8. Định nghĩa "hoàn thành" của cả dự án (Phase 10)

- Chạy 1 lệnh ra video annotated đầy đủ (object + track + lane + đèn + rủi ro + gợi ý).
- Dashboard web xem luồng trực tiếp + bảng thông số.
- Có notebook fine-tune tái lập được + bảng so sánh baseline vs fine-tuned (mAP).
- README chuyên nghiệp, có ảnh/GIF demo, sơ đồ kiến trúc, hướng dẫn cài đặt.
- Test pass, có CI; (tùy chọn) Docker + bản demo online.

> Tiếp theo: mở `phase_1.md`.
