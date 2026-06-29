#!/usr/bin/env python
"""DriveVision — Accuracy benchmark (mAP / Precision / Recall).

Đo ĐỘ CHÍNH XÁC của detector bằng cách so dự đoán với NHÃN ground-truth — KHÁC
với scripts/benchmark.py (chỉ đo tốc độ, không cần nhãn). Cần một dataset có nhãn
ở định dạng Ultralytics (data yaml + ảnh + nhãn YOLO txt).

Dùng:
    # YOLOv8n pretrained trên COCO val2017
    python scripts/eval_accuracy.py --weights models/weights/yolo.pt --data configs/coco_val.yaml

    # Phase 8: so weights fine-tune trên BDD100K val
    python scripts/eval_accuracy.py --weights runs/best.pt --data configs/bdd_val.yaml --imgsz 640

In bảng mAP@50 / mAP@50-95 / Precision / Recall (tổng + vài lớp giao thông),
và tùy chọn --md để in một dòng markdown dán vào EVALUATION.md.
"""

from __future__ import annotations

import argparse

# Các lớp COCO quan trọng với cảnh lái xe — in riêng để dễ đọc.
_TRAFFIC = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]


def main() -> None:
    ap = argparse.ArgumentParser(description="DriveVision accuracy (mAP) benchmark")
    ap.add_argument("--weights", default="models/weights/yolo.pt", help="Đường dẫn .pt")
    ap.add_argument("--data", default="configs/coco_val.yaml", help="Dataset yaml (có nhãn)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="conf thấp khi đo mAP (chuẩn)")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU cho NMS khi val")
    ap.add_argument("--device", default=None, help="cpu | cuda:0 | None=auto")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--label", default="YOLOv8n pretrained", help="Nhãn cho dòng --md")
    ap.add_argument("--md", action="store_true", help="In 1 dòng markdown")
    args = ap.parse_args()

    from ultralytics import YOLO  # lazy: chỉ cần khi thật sự đo accuracy

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )

    box = metrics.box
    names = metrics.names  # {id: name}

    if args.md:
        print("| Model | imgsz | mAP@50 | mAP@50-95 | Precision | Recall |")
        print("|---|---|---|---|---|---|")
        print(f"| {args.label} | {args.imgsz} | {box.map50:.3f} | {box.map:.3f} | "
              f"{box.mp:.3f} | {box.mr:.3f} |")
        return

    print(f"\nWeights : {args.weights}")
    print(f"Data    : {args.data}  (imgsz={args.imgsz})")
    print("\n=== Tổng thể (all classes) ===")
    print(f"  mAP@50      : {box.map50:.4f}")
    print(f"  mAP@50-95   : {box.map:.4f}")
    print(f"  Precision   : {box.mp:.4f}")
    print(f"  Recall      : {box.mr:.4f}")

    # Per-class cho các lớp giao thông (nếu có trong dataset).
    name_to_id = {n: i for i, n in names.items()}
    rows = []
    for cls in _TRAFFIC:
        cid = name_to_id.get(cls)
        if cid is None or cid not in getattr(box, "ap_class_index", []):
            continue
        idx = list(box.ap_class_index).index(cid)
        rows.append((cls, box.ap50[idx], box.ap[idx]))
    if rows:
        print("\n=== Lớp giao thông (per-class) ===")
        print(f"  {'class':<14}{'mAP@50':>10}{'mAP@50-95':>12}")
        for cls, ap50, ap in rows:
            print(f"  {cls:<14}{ap50:>10.3f}{ap:>12.3f}")
    print(f"\nKết quả chi tiết (PR curve, confusion matrix) lưu ở: {metrics.save_dir}")


if __name__ == "__main__":
    main()
