$ErrorActionPreference = "Stop"

$BASELINE_ROOT = Split-Path -Parent $PSScriptRoot
$PROJECT_ROOT = Split-Path -Parent $BASELINE_ROOT
Set-Location $PROJECT_ROOT

python .\baseline\scripts\generate_split_visuals.py `
  --data-root ..\vod-min `
  --split test `
  --max-samples 200 `
  --output-dir .\baseline\results\test_vis

