$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Resolve-Path (Join-Path $SCRIPT_DIR "..\..")
Set-Location $ROOT

python .\improve-v1\eval.py `
  --config .\improve-v1\configs\vod_centerpoint_radar_v1.yaml `
  --ckpt .\improve-v1\outputs\vod_centerpoint_radar_v1\best.pt `
  --use-ema

