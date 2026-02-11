@echo off
setlocal
REM PDF Remaster_v1_0_0 - GPU install helper (Windows)
REM 先に CUDA に合う PyTorch を公式手順で入れてから、この bat を実行してください。
REM (例) CUDA11.8: py -3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install -r requirements-gpu.txt

echo.
echo Done.
pause
