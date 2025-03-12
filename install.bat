@echo off
echo Installing dependencies...

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install numpy and scipy first (pre-built wheels)
pip install numpy==1.24.3
pip install scipy>=1.10.0

REM Install other dependencies
pip install -r requirements.txt

REM Install the package in development mode
pip install -e .

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Error: Installation failed!
    echo Please make sure you have Visual Studio Build Tools installed.
    echo You can download them from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    exit /b 1
) else (
    echo Installation completed successfully!
)

pause
