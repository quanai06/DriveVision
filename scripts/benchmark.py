#!/usr/bin/env python
"""DriveVision — Speed benchmark (FPS + độ trễ từng stage).

Đo TỐC ĐỘ, KHÔNG cần nhãn — chạy được từ Phase 1 trở đi. Đánh giá độ CHÍNH XÁC
(mAP / MOTA / IDF1 / HOTA / IoU...) xem EVALUATION.md, làm bằng công cụ riêng.

Cách dùng:
    # đo trên video thật
    python scripts/benchmark.py --source data/samples/sample.mp4 --frames 200

    # không có video -> tự sinh frame ngẫu nhiên (vẫn đo được tốc độ inference)
    python scripts/benchmark.py --frames 100 --imgsz 320

    # in ra dạng dòng markdown để dán vào EVALUATION.md
    python scripts/benchmark.py --source x.mp4 --md
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevision.config import load_config  # noqa: E402
from drivevision.pipeline.builder import build_pipeline  # noqa: E402
from drivevision.types import Frame, PipelineResult  # noqa: E402
from drivevision.viz.annotator import Annotator  # noqa: E402

# Thứ tự stage trùng với Pipeline.process() — đo riêng từng khâu để biết nghẽn ở đâu.
STAGE_ORDER = [
    "detection",
    "fusion",
    "tracking",
    "lane",
    "traffic_light",
    "scene",
    "risk",
    "decision",
    "annotate",
]


def _frame_iter(source: str | None, n: int, size: tuple[int, int]):
    """Sinh tối đa n Frame: từ video nếu có --source, ngược lại frame ngẫu nhiên."""
    if source:
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {source!r}")
        i = 0
        while i < n:
            ok, img = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop để đủ n frame
                continue
            yield Frame(index=i, timestamp=i / 30.0, image=img)
            i += 1
        cap.release()
    else:
        h, w = size
        rng = np.random.default_rng(0)
        for i in range(n):
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            yield Frame(index=i, timestamp=i / 30.0, image=img)


def benchmark(cfg: dict, source: str | None, frames: int, warmup: int) -> dict:
    pipe = build_pipeline(cfg)
    annotator = Annotator()
    timings: dict[str, list[float]] = {s: [] for s in STAGE_ORDER}

    def tick(stage: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        timings[stage].append((time.perf_counter() - t0) * 1000.0)  # ms
        return out

    total = frames + warmup
    counted = 0
    for k, frame in enumerate(_frame_iter(source, total, (720, 1280))):
        # Bỏ qua warmup (frame đầu chậm do nạp model / cấp phát bộ nhớ).
        record = k >= warmup
        local: dict[str, list[float]] = timings if record else {s: [] for s in STAGE_ORDER}

        def t(stage, fn, *a):
            t0 = time.perf_counter()
            out = fn(*a)
            local[stage].append((time.perf_counter() - t0) * 1000.0)
            return out

        dets = t("detection", pipe.detector.detect, frame) if pipe.detector else []
        if pipe.fusion and pipe.fusion.enabled:
            dets = t("fusion", pipe.fusion.apply, frame, dets)
        tracks = t("tracking", pipe.tracker.update, frame, dets) if pipe.tracker else []
        lanes = t("lane", pipe.lane_detector.detect, frame) if pipe.lane_detector else []
        lights = (
            t("traffic_light", pipe.traffic_light_detector.detect, frame, dets)
            if pipe.traffic_light_detector
            else []
        )
        scene = t("scene", pipe.scene_builder.build, frame, dets, tracks, lanes, lights)
        risk = t("risk", pipe.risk_assessor.assess, scene) if pipe.risk_assessor else None
        decision = (
            t("decision", pipe.decision_support.decide, scene, risk)
            if pipe.decision_support
            else None
        )
        result = PipelineResult(frame=frame, scene=scene, risk=risk, decision=decision)
        t("annotate", annotator.draw, result)

        if record:
            counted += 1

    return {"timings": timings, "counted": counted}


def _stats(ms: list[float]) -> tuple[float, float]:
    if not ms:
        return 0.0, 0.0
    return statistics.mean(ms), (statistics.stdev(ms) if len(ms) > 1 else 0.0)


def print_table(result: dict, cfg: dict) -> None:
    timings = result["timings"]
    means = {s: _stats(v)[0] for s, v in timings.items()}
    total_ms = sum(means.values())
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    print(f"\n{'Stage':<16}{'ms/frame':>12}{'± std':>10}{'% tổng':>10}")
    print("-" * 48)
    for s in STAGE_ORDER:
        if not timings[s]:
            continue
        mean, std = _stats(timings[s])
        pct = (mean / total_ms * 100.0) if total_ms else 0.0
        print(f"{s:<16}{mean:>10.2f}  {std:>8.2f}  {pct:>8.1f}%")
    print("-" * 48)
    print(f"{'TỔNG':<16}{total_ms:>10.2f}  {'':>8}  {'':>8}   →  {fps:.1f} FPS")
    print(f"\nFrames đo: {result['counted']}  |  imgsz={cfg['perception']['detection']['imgsz']}")


def print_md(result: dict, cfg: dict, label: str) -> None:
    means = {s: _stats(v)[0] for s, v in result["timings"].items()}
    total = sum(means.values())
    fps = 1000.0 / total if total else 0.0
    det = means.get("detection", 0.0)
    print(f"| {label} | {cfg['perception']['detection']['imgsz']} | "
          f"{det:.1f} | {total:.1f} | {fps:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description="DriveVision speed benchmark")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default=None, help="Video (bỏ trống = frame ngẫu nhiên)")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=None, help="Ghi đè imgsz")
    ap.add_argument("--md", action="store_true", help="In 1 dòng markdown")
    ap.add_argument("--label", default="PyTorch CPU", help="Nhãn cấu hình cho dòng md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.imgsz:
        cfg["perception"]["detection"]["imgsz"] = args.imgsz

    if not args.md:
        src = args.source or "(frame ngẫu nhiên)"
        print(f"Benchmark | nguồn: {src} | frames: {args.frames} (+{args.warmup} warmup)")

    result = benchmark(cfg, args.source, args.frames, args.warmup)

    if args.md:
        print("| Cấu hình | imgsz | Detection (ms) | Tổng (ms) | FPS |")
        print("|---|---|---|---|---|")
        print_md(result, cfg, args.label)
    else:
        print_table(result, cfg)


if __name__ == "__main__":
    main()
