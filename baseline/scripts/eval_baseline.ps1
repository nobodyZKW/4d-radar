$ErrorActionPreference = "Stop"

$BASELINE_ROOT = Split-Path -Parent $PSScriptRoot
$PROJECT_ROOT = Split-Path -Parent $BASELINE_ROOT
Set-Location $PROJECT_ROOT

python -m baseline.eval `
  --config baseline/configs/vod_baseline.yaml `
  --ckpt baseline/outputs/vod_baseline/best.pt
