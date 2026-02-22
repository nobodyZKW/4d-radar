# Baseline (VoD 4D Radar)

This folder contains an isolated baseline pipeline using the current dataset at
`E:\毕设\code\vod-min`.

## Structure

- `__init__.py`, `dataset.py`, `model.py`, `train.py`, `eval.py`: baseline code
- `configs/vod_baseline.yaml`: baseline config
- `scripts/train_baseline.ps1`: one-command training
- `scripts/eval_baseline.ps1`: one-command evaluation
- `requirements.txt`: python dependencies
- `outputs/`: checkpoints and logs (created by training)

## Setup

```powershell
cd E:\毕设\code\4d-radar
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r baseline\requirements.txt
```

## Train

```powershell
cd E:\毕设\code\4d-radar
.\baseline\scripts\train_baseline.ps1
```

Outputs are written to `baseline/outputs/vod_baseline`.

## Eval

```powershell
cd E:\毕设\code\4d-radar
.\baseline\scripts\eval_baseline.ps1
```

## Notes

- This version uses a center-distance matching metric for fast iteration.
- For paper-level 3D mAP, integrate the official VoD/KITTI evaluator next.
