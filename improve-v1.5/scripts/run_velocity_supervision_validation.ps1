param(
  [switch]$RunTrain,
  [int]$Workers = 4,
  [int]$Epochs = 0,
  [string]$OutputDir = "outputs/ablations/velocity",
  [string]$TagPrefix = "vel_ablation",
  [switch]$SkipMissingCkpt,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $Root

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONPATH = "$Root;$Root\external\OpenPCDet"

$ablationCmd = @(
  "-m", "pcdet_ext.utils.ablation_runner",
  "--root", ".",
  "--workers", "$Workers",
  "--output-dir", "$OutputDir",
  "--extra-tag-prefix", "$TagPrefix",
  "--preset", "velocity"
)
if ($RunTrain) { $ablationCmd += "--run-train" }
if ($Epochs -gt 0) { $ablationCmd += @("--epochs", "$Epochs") }
if ($SkipMissingCkpt) { $ablationCmd += "--skip-missing-ckpt" }
if ($DryRun) { $ablationCmd += "--dry-run" }
python @ablationCmd

$reportCmd = @(
  "-m", "pcdet_ext.utils.velocity_validation_report",
  "--root", ".",
  "--input-dir", "$OutputDir",
  "--output-csv", "$OutputDir/velocity_validation_summary.csv",
  "--output-md", "$OutputDir/velocity_validation_summary.md",
  "--output-plot", "$OutputDir/velocity_validation_summary.png"
)
python @reportCmd

