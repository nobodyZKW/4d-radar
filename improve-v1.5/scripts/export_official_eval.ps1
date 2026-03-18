param(
  [string]$ResultPkl = "",
  [string]$OutputDir = "outputs/official_eval",
  [string]$DevkitCmd = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $Root

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

if ($ResultPkl -eq "") {
  throw "Please provide --ResultPkl path to result.pkl"
}

$env:PYTHONPATH = "$Root;$Root\external\OpenPCDet"
$cmd = @(
  "-m", "pcdet_ext.eval.vod_official_eval_adapter",
  "--result-pkl", $ResultPkl,
  "--output-dir", $OutputDir
)
if ($DevkitCmd -ne "") {
  $cmd += @("--devkit-cmd", $DevkitCmd)
}
python @cmd
