param([string]$Source,[string]$StoreId="MARKET-001",[string]$MapDir="")
if(-not $Source){throw "-Source zorunludur."}
if(-not $MapDir){$MapDir=Join-Path $PSScriptRoot "maps\$StoreId"}
& py -3.11 "$PSScriptRoot\run_shelf_gap_cascade.py" --operation-mode localization --source $Source `
  --store-id $StoreId --map-dir $MapDir --shelf-model "$PSScriptRoot\best.pt" `
  --empty-model "$PSScriptRoot\empty_shelf_yolo11m_best.pt"
exit $LASTEXITCODE
