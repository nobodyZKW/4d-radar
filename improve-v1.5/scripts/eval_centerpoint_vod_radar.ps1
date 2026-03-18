param(
  [string]$Config = "configs/model_configs/vod_centerpoint_radar_v1_5.yaml",
  [string]$Ckpt = "",
  [string]$TrainTag = "v1_5",
  [string]$EvalTag = "v1_5_eval",
  [int]$Workers = 4
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $Root

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

$env:PYTHONPATH = "$Root;$Root\external\OpenPCDet"

if ($Ckpt -eq "") {
  $cfgName = [System.IO.Path]::GetFileNameWithoutExtension($Config)
  $ckptDir = Join-Path $Root "external\OpenPCDet\output\model_configs\$cfgName\$TrainTag\ckpt"
  $latest = Get-ChildItem -Path $ckptDir -Filter *.pth | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($null -eq $latest) { throw "No checkpoint found in $ckptDir" }
  $Ckpt = $latest.FullName
}

python external/OpenPCDet/tools/test.py --cfg_file $Config --ckpt $Ckpt --workers $Workers --save_to_file --extra_tag $EvalTag
