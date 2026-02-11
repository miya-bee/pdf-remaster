@echo off
setlocal
REM PDF Remaster_v1_0_0 - CPU install helper (Windows)
REM 推奨: Python 3.10 (py -3.10 が使える状態)

py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install -r requirements.txt

echo.
echo Done.
pause
