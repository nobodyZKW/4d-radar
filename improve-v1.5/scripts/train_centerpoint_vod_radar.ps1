param(
  [string]$Config = "configs/model_configs/vod_centerpoint_radar_v1_5.yaml",
  [string]$ExtraTag = "v1_5",
  [int]$Workers = 4,
  [int]$Epochs = 0
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $Root

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

$env:PYTHONPATH = "$Root;$Root\external\OpenPCDet"
$cmd = @(
  "external/OpenPCDet/tools/train.py",
  "--cfg_file", $Config,
  "--extra_tag", $ExtraTag,
  "--workers", "$Workers"
)
if ($Epochs -gt 0) {
  $cmd += @("--epochs", "$Epochs")
}
python @cmd
