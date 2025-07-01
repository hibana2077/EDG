@echo off
echo AugNet Visualization Tool
echo =========================

rem Check if a checkpoint file was provided
if "%~1"=="" (
    echo Error: Please provide a checkpoint file path
    echo Usage: visualize_augnet.bat [checkpoint_path] [options]
    echo Options:
    echo   --output [path]        Output path (default: augnet_visualization.png)
    echo   --num-examples [num]   Number of examples to show (default: 4)
    echo   --device [cuda/cpu]    Device to use (default: cuda)
    echo.
    echo Example: visualize_augnet.bat checkpoints\checkpoint_epoch_100.pth --num-examples 6
    exit /b 1
)

rem Set default values
set CHECKPOINT=%~1
set OUTPUT=augnet_visualization.png
set NUM_EXAMPLES=4
set DEVICE=cuda

rem Parse command line arguments
shift
:parse_args
if "%~1"=="" goto run_script
if "%~1"=="--output" (
    set OUTPUT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--num-examples" (
    set NUM_EXAMPLES=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:run_script
echo Generating AugNet visualization...
echo Checkpoint: %CHECKPOINT%
echo Output: %OUTPUT%
echo Number of examples: %NUM_EXAMPLES%
echo Device: %DEVICE%
echo.

python src/generate_augnet_visualization.py --checkpoint "%CHECKPOINT%" --output "%OUTPUT%" --num_examples %NUM_EXAMPLES% --device %DEVICE%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Visualization completed successfully!
    echo Saved to: %OUTPUT%
) else (
    echo.
    echo Error occurred during visualization.
)
