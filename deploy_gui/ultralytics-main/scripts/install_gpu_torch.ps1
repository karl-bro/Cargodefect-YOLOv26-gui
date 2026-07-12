# Install GPU PyTorch for RTX 4060 (CUDA 12.4)
# Run in PowerShell: .\scripts\install_gpu_torch.ps1

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "=== Current PyTorch ===" -ForegroundColor Cyan
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

Write-Host "`n=== Installing CUDA PyTorch (约 2.5GB，请耐心等待) ===" -ForegroundColor Cyan
Write-Host "Method 1: pip (recommended)" -ForegroundColor Yellow

pip install torch==2.5.1 torchvision==0.20.1 `
    --index-url https://download.pytorch.org/whl/cu124 `
    --force-reinstall

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPip failed, trying conda..." -ForegroundColor Yellow
    conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
}

Write-Host "`n=== Verify ===" -ForegroundColor Cyan
python -c @"
import torch
ok = torch.cuda.is_available()
print('torch:', torch.__version__)
print('CUDA:', ok)
if ok:
    print('GPU:', torch.cuda.get_device_name(0))
    print('SUCCESS - run: python scripts/train_fusion.py')
else:
    print('FAILED - retry install or check network')
"@
