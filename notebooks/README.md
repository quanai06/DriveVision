# Notebooks — Research / Training plane (Kaggle)

This folder holds Kaggle training notebooks. They are **decoupled** from the
runtime: the only artifact that crosses over is the trained weights file.

Workflow:
1. Fine-tune a model on Kaggle (GPU) — e.g. YOLO on a CARLA dataset.
2. Download the resulting `best.pt`.
3. Drop it into `../models/weights/yolo.pt`.
4. The runtime picks it up automatically (see `configs/default.yaml`).

Planned notebooks:
- `01_finetune_yolo_carla.ipynb` — fine-tune detection on a CARLA dataset (M5).
- `02_lane_segmentation.ipynb`   — lane model training (optional).

> Note: CARLA itself does not run on Kaggle. Use a pre-existing CARLA dataset
> from Kaggle, or fine-tune on BDD100K/KITTI for the dashcam domain.
