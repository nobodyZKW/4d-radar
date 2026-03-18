param(
  [string]$DataRoot = "../../vod-min"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $Root

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

$DataPath = (Resolve-Path $DataRoot).Path
Write-Host "Data root: $DataPath"

$Required = @(
  "radar_5frames/training/velodyne",
  "lidar/training/label_2",
  "lidar/training/calib",
  "lidar/ImageSets"
)

foreach ($rel in $Required) {
  $p = Join-Path $DataPath $rel
  if (-not (Test-Path $p)) {
    throw "Missing required path: $p"
  }
}

$trainFile = Join-Path $DataPath "lidar/ImageSets/train.txt"
$valFile = Join-Path $DataPath "lidar/ImageSets/val.txt"
$testFile = Join-Path $DataPath "lidar/ImageSets/test.txt"

if (-not (Test-Path $testFile)) {
  Write-Host "test.txt not found, creating from tail 10% of train split"
  $trainIds = Get-Content $trainFile | Where-Object { $_.Trim() -ne "" }
  $n = $trainIds.Count
  $nTest = [Math]::Max(1, [Math]::Floor($n * 0.1))
  $trainNew = $trainIds[0..($n - $nTest - 1)]
  $testNew = $trainIds[($n - $nTest)..($n - 1)]
  $trainNew | Set-Content $trainFile -Encoding UTF8
  $testNew | Set-Content $testFile -Encoding UTF8
  Write-Host "Created test split: $nTest / $n"
}

$counts = @{
  train = (Get-Content $trainFile | Where-Object { $_.Trim() -ne "" }).Count
  val   = (Get-Content $valFile | Where-Object { $_.Trim() -ne "" }).Count
  test  = (Get-Content $testFile | Where-Object { $_.Trim() -ne "" }).Count
}
Write-Host ("Split counts -> train={0}, val={1}, test={2}" -f $counts.train, $counts.val, $counts.test)
