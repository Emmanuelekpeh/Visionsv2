
<#
.SYNOPSIS
    Prepare T+1 training dataset for Kaggle upload
.DESCRIPTION
    Creates a properly structured directory and compresses it for Kaggle dataset upload
#>

param(
    [string]$DatasetPath = "dataset\t_plus_1_training",
    [string]$OutputName = "t_plus_1_video_pairs",
    [switch]$SkipCompression
)

Write-Host "=" -NoNewline; Write-Host ("=" * 59)
Write-Host "Kaggle Dataset Preparation for T+1 VAE Training"
Write-Host "=" -NoNewline; Write-Host ("=" * 59)

# Check dataset exists
if (-not (Test-Path $DatasetPath)) {
    Write-Host "`n[ERROR] Dataset directory not found: $DatasetPath" -ForegroundColor Red
    Write-Host "`nRun extract_frame_pairs.py first to create the dataset"
    exit 1
}

# Check manifest exists
$manifestPath = Join-Path $DatasetPath "t_plus_1_pairs.jsonl"
if (-not (Test-Path $manifestPath)) {
    Write-Host "`n[ERROR] Manifest file not found: $manifestPath" -ForegroundColor Red
    Write-Host "`nDataset is incomplete - manifest missing"
    exit 1
}

# Count files
$pngFiles = Get-ChildItem $DatasetPath -Filter *.png
$pairCount = (Get-Content $manifestPath | Measure-Object -Line).Lines

Write-Host "`n[OK] Dataset validation:" -ForegroundColor Green
Write-Host "  Location: $DatasetPath"
Write-Host "  PNG files: $($pngFiles.Count)"
Write-Host "  Manifest pairs: $pairCount"

# Calculate size
$totalSize = ($pngFiles | Measure-Object -Property Length -Sum).Sum
$sizeMB = [math]::Round($totalSize / 1MB, 2)
Write-Host "  Total size: $sizeMB MB"

# Create metadata file for Kaggle
$metadata = @{
    title = $OutputName
    id = "meowkittypaws/$OutputName"
    licenses = @(
        @{ name = "other" }
    )
    description = @"
# T+1 VAE Video Frame Pairs Dataset

Consecutive video frame pairs extracted for temporal dynamics training.

## Contents
- Frame pairs at 64x64 resolution
- JSONL manifest with metadata
- Total pairs: $pairCount

## Usage
```python
import json
from PIL import Image

# Load manifest
with open('t_plus_1_pairs.jsonl', 'r') as f:
    pairs = [json.loads(line) for line in f]

# Load a frame pair
pair = pairs[0]
prev_frame = Image.open(pair['prev'])
next_frame = Image.open(pair['next'])
```

## Training
Use with Phase 2 of T+1 VAE training (temporal dynamics).

Generated: $(Get-Date -Format "yyyy-MM-dd")
"@
}

$metadataJson = $metadata | ConvertTo-Json -Depth 10
$metadataPath = Join-Path $DatasetPath "dataset-metadata.json"
$metadataJson | Out-File -FilePath $metadataPath -Encoding UTF8

Write-Host "`n[OK] Created Kaggle metadata: dataset-metadata.json" -ForegroundColor Green

if (-not $SkipCompression) {
    Write-Host "`n[COMPRESSING] Creating archive..." -ForegroundColor Cyan
    
    $zipPath = "$OutputName.zip"
    
    # Remove existing zip
    if (Test-Path $zipPath) {
        Remove-Item $zipPath -Force
        Write-Host "  Removed existing archive"
    }
    
    # Create zip
    try {
        Compress-Archive -Path $DatasetPath -DestinationPath $zipPath -CompressionLevel Optimal
        $zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
        
        Write-Host "`n[OK] Archive created!" -ForegroundColor Green
        Write-Host "  File: $zipPath"
        Write-Host "  Size: $zipSize MB"
        
    } catch {
        Write-Host "`n[ERROR] Compression failed: $_" -ForegroundColor Red
        exit 1
    }
}

# Upload instructions
Write-Host "`n" -NoNewline; Write-Host ("=" * 60)
Write-Host "READY FOR KAGGLE UPLOAD" -ForegroundColor Green
Write-Host "=" -NoNewline; Write-Host ("=" * 59)

Write-Host "`nUpload Steps:"
Write-Host "  1. Go to: https://www.kaggle.com/datasets"
Write-Host "  2. Click 'New Dataset'"
Write-Host "  3. Upload: $zipPath"
Write-Host "  4. Title: $OutputName"
Write-Host "  5. Make it Public (or Private for testing)"
Write-Host "  6. Click 'Create'"

Write-Host "`nKaggle Dataset Path:"
Write-Host "  /kaggle/input/meowkittypaws/$OutputName" -ForegroundColor Yellow

Write-Host "`nAdd to notebook:"
Write-Host "  1. Edit notebook"
Write-Host "  2. Add Data > Search: $OutputName"
Write-Host "  3. Add to notebook"

Write-Host "`n[OK] All done!" -ForegroundColor Green
