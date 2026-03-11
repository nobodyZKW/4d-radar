# Baseline (VoD 4D Radar)

This folder contains an isolated baseline pipeline using the current dataset at
`E:\毕设\code\vod-min`.

## Structure

- `__init__.py`, `dataset.py`, `model.py`, `train.py`, `eval.py`: baseline code
- `configs/vod_baseline.yaml`: baseline config
- `scripts/train_baseline.ps1`: one-command training
- `scripts/eval_baseline.ps1`: one-command evaluation
- `scripts/generate_test_visuals.ps1`: batch export test visualizations (left image, right BEV)
- `requirements.txt`: python dependencies
- `outputs/`: checkpoints and logs (created by training)

## Split Policy

- Training uses `ImageSets/train.txt`
- Testing uses `ImageSets/test.txt` (configured via `val_split: test`)
- `val.txt` is currently synced to `test.txt` for compatibility

## Clean Previous Results

```powershell
cd E:\毕设\code\4d-radar
cmd /c "rmdir /s /q baseline\outputs 2>nul & rmdir /s /q baseline\results 2>nul & mkdir baseline\outputs & mkdir baseline\results"
```

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

## Generate Test Result Images (50% image + 50% BEV)

```powershell
cd E:\毕设\code\4d-radar
.\baseline\scripts\generate_test_visuals.ps1
```

Output folder:

- `baseline/results/test_vis/*.png`

## Visualization

Quickly visualize one sample (left: image + 2D boxes, right: radar BEV):

```powershell
python E:\毕设\code\4d-radar\baseline\scripts\visualize_sample.py `
  --data-root E:\毕设\code\vod-min `
  --id 00000 `
  --save E:\毕设\code\4d-radar\baseline\vis_00000.png
```

You can also run directly with Python:

```powershell
python -m baseline.eval
```

If you run with file path (not recommended), it is now supported too:

```powershell
python E:\毕设\code\4d-radar\baseline\eval.py
```

## PowerShell note

`$PSScriptRoot` only exists inside `.ps1` files.  
Do not paste the script internals into terminal line-by-line.  
Run the script file itself, for example:

```powershell
.\baseline\scripts\train_baseline.ps1
```

## Notes

- This version uses a center-distance matching metric for fast iteration.
- For paper-level 3D mAP, integrate the official VoD/KITTI evaluator next.
