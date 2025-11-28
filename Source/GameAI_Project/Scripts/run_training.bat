@echo off
REM SBDAPM RLlib Training Runner
REM Activates game_ai conda environment and runs training script

echo ============================================================
echo SBDAPM RLlib Training
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda not found in PATH
    echo Please ensure Anaconda/Miniconda is installed and in PATH
    pause
    exit /b 1
)

REM Activate game_ai environment and run training
echo Activating conda environment: game_ai
echo.

REM Default parameters
set ITERATIONS=10
set PORT=50051

REM Parse command line arguments (optional)
if not "%1"=="" set ITERATIONS=%1
if not "%2"=="" set PORT=%2

echo Training parameters:
echo   - Iterations: %ITERATIONS%
echo   - Port: %PORT%
echo.

REM Run training with conda
call conda run -n game_ai python train_rllib.py --iterations %ITERATIONS% --port %PORT%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Training completed successfully!
pause
