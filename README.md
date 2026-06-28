# DriveVision

**High-Performance Autonomous Driving Perception Platform** — a modular
perception pipeline (object detection, tracking, lanes, traffic lights, scene
understanding, risk assessment, decision support) with a FastAPI dashboard.

> Status: **skeleton / framework**. Stages are wired together; most contain
> `TODO` stubs to fill in milestone by milestone.

## Design: two decoupled planes

```
RESEARCH PLANE (Kaggle notebooks)  ──fine-tune──▶  weights (.pt/.onnx)
                                                         │ drop into models/weights/
RUNTIME PLANE (local, on video)  DataSource ▶ Pipeline ▶ Viz ▶ API/Dashboard
```

The two only communicate through a weights file. The `DataSource` seam lets you
start on plain video/dataset now and plug in CARLA later (`io/carla_source.py`).

## Layout

```
src/drivevision/
  types.py        core dataclasses (shared contract)
  config.py       YAML config + defaults
  io/             DataSource: video (M0), carla (M6 stub)
  perception/     base ABCs + detection / tracking / lane / traffic_light
  fusion/         depth/semantic fusion (stub)
  scene/          SceneBuilder
  risk/           RiskAssessor (rule-based, stub)
  decision/       DecisionSupport (rule-based, stub)
  pipeline/       Pipeline orchestrator + builder (wiring)
  viz/            Annotator (stub)
  api/            FastAPI server + MJPEG runner (stub)
configs/  notebooks/  models/weights/  data/  scripts/  tests/  dashboard/
```

## Quick start

```bash
source .venv/bin/activate
pip install -e .                 # core (numpy, opencv, pyyaml)
pip install -e ".[perception]"   # + ultralytics (real detection)
pip install -e ".[api]"          # + fastapi/uvicorn (dashboard)

# Run the pipeline over a video
python scripts/run_pipeline.py --source data/samples/sample.mp4

pytest -q                        # smoke tests (no ML deps needed)
```

## Milestones

| # | Goal |
|---|------|
| M0 | Skeleton: types, config, CLI runs a video through an (empty) pipeline |
| M1 | Object detection (YOLO) + tracking -> annotated video |
| M2 | Lane + traffic-light detection |
| M3 | Scene understanding + risk + decision (rule-based) |
| M4 | FastAPI + dashboard (MJPEG stream + risk panel) |
| M5 | Fine-tune on Kaggle (CARLA dataset) -> swap weights |
| M6 | CARLA live source |

To add real behaviour, search the code for `TODO` — each marks a stage to implement.
