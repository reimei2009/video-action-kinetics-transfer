# Quick API Test Script
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "🧪 Video Action Recognition API - Quick Test" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Test 1: Health Check
Write-Host "🏥 Testing Health Endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -Method Get
    Write-Host "✅ Status: $($health.status)" -ForegroundColor Green
    Write-Host "✅ Model: $($health.model_name)" -ForegroundColor Green
    Write-Host "✅ Device: $($health.device)" -ForegroundColor Green
} catch {
    Write-Host "❌ Health check failed: $_" -ForegroundColor Red
    exit 1
}

# Test 2: Predict
Write-Host "`n📹 Testing Predict Endpoint..." -ForegroundColor Yellow
Write-Host "   Uploading video: 4088191-hd_1920_1080_25fps.mp4" -ForegroundColor Gray

$videoPath = "D:\Đại học\Ki1Nam4\Project\video-action-kinetics-transfer\4088191-hd_1920_1080_25fps.mp4"

if (Test-Path $videoPath) {
    try {
        # Create multipart form
        $boundary = [System.Guid]::NewGuid().ToString()
        $bodyLines = @(
            "--$boundary",
            "Content-Disposition: form-data; name=`"file`"; filename=`"4088191-hd_1920_1080_25fps.mp4`"",
            "Content-Type: video/mp4",
            "",
            [System.IO.File]::ReadAllText($videoPath),
            "--$boundary--"
        )
        
        # Use curl instead (simpler)
        Write-Host "   Processing..." -ForegroundColor Gray
        
        $result = & curl.exe -X POST "http://127.0.0.1:8000/api/v1/predict?top_k=5" `
            -F "file=@$videoPath" `
            -H "accept: application/json" `
            --silent
        
        $data = $result | ConvertFrom-Json
        
        if ($data.success) {
            Write-Host "`n✅ SUCCESS!" -ForegroundColor Green
            Write-Host "`n🎯 TOP PREDICTIONS:" -ForegroundColor Cyan
            Write-Host "============================================================" -ForegroundColor Cyan
            
            foreach ($pred in $data.predictions) {
                $rank = $pred.rank
                $label = $pred.label
                $conf = [math]::Round($pred.confidence * 100, 2)
                
                # Color based on confidence
                $color = if ($conf -gt 15) { "Green" } elseif ($conf -gt 10) { "Yellow" } else { "Gray" }
                
                Write-Host "#$rank. " -NoNewline
                Write-Host "$label" -NoNewline -ForegroundColor $color
                Write-Host " - $conf%" -ForegroundColor $color
            }
            
            Write-Host "`n📊 METADATA:" -ForegroundColor Cyan
            Write-Host "============================================================" -ForegroundColor Cyan
            Write-Host "Model: $($data.model_name)"
            Write-Host "Processing Time: $($data.processing_time)s"
            Write-Host "Video Duration: $($data.video_metadata.duration)s"
            Write-Host "============================================================`n" -ForegroundColor Cyan
            
            Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
        } else {
            Write-Host "❌ Prediction failed" -ForegroundColor Red
        }
        
    } catch {
        Write-Host "❌ Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Video file not found: $videoPath" -ForegroundColor Red
}
