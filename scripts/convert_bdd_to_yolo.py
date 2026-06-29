#!/usr/bin/env python
"""Convert BDD100K detection labels (Scalabel JSON) -> YOLO txt labels.

BDD100K ships detection labels as one big JSON (e.g. ``det_val.json`` or the
older ``bdd100k_labels_images_val.json``). Ultralytics ``model.val()`` /
training needs one ``.txt`` per image in YOLO format:

    <class_id> <x_center> <y_center> <width> <height>   # all normalised 0..1

Two target id-spaces (`--target`):

* ``bdd``  — BDD's own 10 detection classes (0..9). Use this for FINE-TUNING and
  for evaluating a model that was fine-tuned on BDD. Pair with ``configs/bdd_det.yaml``.
* ``coco`` — remap BDD categories onto COCO ids so a COCO-PRETRAINED model
  (YOLOv8n straight out of the box, which predicts COCO ids) can be evaluated
  fairly. Classes with no COCO equivalent (rider, traffic sign) are dropped.
  Pair with ``configs/bdd_val_coco.yaml``.

BDD100K images are a fixed 1280x720; override with --img-width/--img-height if needed.

Usage:
    python scripts/convert_bdd_to_yolo.py \
        --labels data/bdd100k/labels/det_20/det_val.json \
        --out    data/bdd100k/labels/val2017 \
        --target coco
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Canonical BDD100K detection classes (det_20 order) -> id used in `bdd` target.
_BDD_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
]
_BDD_ID = {name: i for i, name in enumerate(_BDD_CLASSES)}

# Older label files use slightly different names — normalise to canonical.
_ALIAS = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
}

# BDD category -> COCO id (for evaluating a COCO-pretrained model). Unmapped
# categories (rider, traffic sign) are intentionally absent and get skipped.
_BDD_TO_COCO = {
    "pedestrian": 0,      # person
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "bicycle": 1,
    "traffic light": 9,
}


def _canon(category: str) -> str:
    return _ALIAS.get(category, category)


def convert(labels_json: Path, out_dir: Path, target: str, w: float, h: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(labels_json.read_text(encoding="utf-8"))

    n_imgs = n_boxes = n_skipped = 0
    for frame in data:
        name = frame.get("name")
        if not name:
            continue
        stem = Path(name).stem
        lines: list[str] = []
        for lab in frame.get("labels") or []:
            box = lab.get("box2d")
            if not box:
                continue  # lane/drivable/poly labels — not detection boxes
            cat = _canon(lab.get("category", ""))
            if target == "coco":
                if cat not in _BDD_TO_COCO:
                    n_skipped += 1
                    continue
                cid = _BDD_TO_COCO[cat]
            else:  # bdd
                if cat not in _BDD_ID:
                    n_skipped += 1
                    continue
                cid = _BDD_ID[cat]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1
        # Write even empty files so val() counts the image as a background negative.
        (out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_imgs += 1

    print(f"Converted {n_imgs} images, {n_boxes} boxes -> {out_dir}  "
          f"(target={target}, skipped {n_skipped} boxes of unmapped classes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="BDD100K det JSON -> YOLO txt")
    ap.add_argument("--labels", required=True, help="BDD det JSON (det_val.json / det_train.json)")
    ap.add_argument("--out", required=True, help="Output dir for YOLO .txt labels")
    ap.add_argument("--target", choices=["bdd", "coco"], default="coco",
                    help="bdd=10-class native (fine-tune); coco=remap to COCO ids (pretrained eval)")
    ap.add_argument("--img-width", type=float, default=1280.0)
    ap.add_argument("--img-height", type=float, default=720.0)
    args = ap.parse_args()

    convert(Path(args.labels), Path(args.out), args.target, args.img_width, args.img_height)


if __name__ == "__main__":
    main()
