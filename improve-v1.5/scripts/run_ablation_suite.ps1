param(
  [switch]$RunTrain,
  [int]$Workers = 4,
  [int]$Epochs = 0,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $Root

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

$env:PYTHONPATH = "$Root;$Root\external\OpenPCDet"

$cmd = @(
  "-m", "pcdet_ext.utils.ablation_runner",
  "--root", ".",
  "--workers", "$Workers"
)
if ($RunTrain) { $cmd += "--run-train" }
if ($Epochs -gt 0) { $cmd += @("--epochs", "$Epochs") }
if ($DryRun) { $cmd += "--dry-run" }
python @cmd
