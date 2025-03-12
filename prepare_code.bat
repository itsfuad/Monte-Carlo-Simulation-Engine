@echo off
echo Preparing code for push...

echo.
echo 1. Formatting code with Black...
black src tests
if %ERRORLEVEL% NEQ 0 (
    echo Error: Black formatting failed
    exit /b 1
)

echo.
echo 2. Running flake8 linting...
flake8 src tests --max-line-length=100 --extend-ignore=E203
if %ERRORLEVEL% NEQ 0 (
    echo Error: Flake8 linting failed
    exit /b 1
)

echo.
echo 3. Running tests...
pytest tests/test_simulation.py -v "--cov=src" "--cov-report=term-missing"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Tests failed
    exit /b 1
)

echo.
echo All checks passed! Code is ready to push.

pause 