param(
  [string]$Config = "configs/dataset_configs/vod_radar_dataset.yaml"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $Root

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

$env:PYTHONPATH = "$Root;$Root\external\OpenPCDet"
python pcdet_ext/datasets/vod_radar/vod_radar_dataset.py create_vod_radar_infos $Config
