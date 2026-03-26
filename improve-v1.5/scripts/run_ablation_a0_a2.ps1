param(
  [switch]$RunTrain,
  [int]$Workers = 4,
  [int]$Epochs = 0,
  [string]$OutputDir = "outputs/ablations/a0_a2",
  [string]$TagPrefix = "ablation_a0a2",
  [switch]$SkipMissingCkpt,
  [switch]$List,
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
  "--workers", "$Workers",
  "--output-dir", "$OutputDir",
  "--extra-tag-prefix", "$TagPrefix",
  "--preset", "a0_a2"
)
if ($RunTrain) { $cmd += "--run-train" }
if ($Epochs -gt 0) { $cmd += @("--epochs", "$Epochs") }
if ($SkipMissingCkpt) { $cmd += "--skip-missing-ckpt" }
if ($List) { $cmd += "--list" }
if ($DryRun) { $cmd += "--dry-run" }

python @cmd
