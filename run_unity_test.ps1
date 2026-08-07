param(
    [string]$UnityData = "C:\Users\yusuf\OneDrive\Belgeler\Unity Projects\My project\ShelfSystemData",
    [string]$CameraId = "UNITY"
)
$metadata = Get-ChildItem -LiteralPath (Join-Path $UnityData "input") -Filter "frame_*.json" | Sort-Object Name | Select-Object -First 1
if (-not $metadata) { throw "ShelfSystemData\input altında frame metadata bulunamadı." }
& py -3.11 "$PSScriptRoot\run_shelf_gap_cascade.py" `
  --source (Join-Path $UnityData "input") --camera-id $CameraId --id-mode pose_geometry `
  --frame-metadata $metadata.FullName --store-map (Join-Path $UnityData "store_map.json") `
  --shelf-model "$PSScriptRoot\best.pt" --empty-model "$PSScriptRoot\empty_shelf_yolo11m_best.pt" `
  --save-annotated
exit $LASTEXITCODE
