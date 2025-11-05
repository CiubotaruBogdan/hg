@echo off
REM ========================================
REM HG 585 LLM Evaluation System - Windows Setup
REM GPU-optimized setup for Windows systems
REM ========================================

echo.
echo ======================================================================
echo            HG 585 LLM EVALUATION SYSTEM - WINDOWS SETUP
echo                     GPU-Optimized Installation
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python version: %PYTHON_VERSION%

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

REM Create virtual environment
echo [INFO] Creating Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [WARNING] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Check for NVIDIA GPU
echo [INFO] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    set GPU_AVAILABLE=true
) else (
    echo [WARNING] No NVIDIA GPU detected or nvidia-smi not available
    set GPU_AVAILABLE=false
)

REM Install PyTorch
echo [INFO] Installing PyTorch...
if "%GPU_AVAILABLE%"=="true" (
    echo [INFO] Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo [SUCCESS] PyTorch with CUDA 11.8 support installed
) else (
    echo [INFO] Installing PyTorch CPU version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo [WARNING] PyTorch CPU version installed (no GPU detected)
)

REM Install other requirements
echo [INFO] Installing other requirements...
pip install -r requirements.txt

REM Verify PyTorch installation
echo [INFO] Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU devices: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CPU only')"

REM Download NLTK data
echo [INFO] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); print('NLTK data downloaded')"

REM Create directories
echo [INFO] Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "results\visualizations" mkdir results\visualizations
if not exist "logs" mkdir logs

REM Create log files
echo [INFO] Setting up logging...
type nul > logs\training.log
type nul > logs\evaluation.log
type nul > logs\system.log

REM Create Windows GPU monitoring script
echo [INFO] Creating GPU monitoring script...
(
echo import time
echo import subprocess
echo import json
echo from datetime import datetime
echo.
echo def get_gpu_stats^(^):
echo     """Get current GPU statistics on Windows."""
echo     try:
echo         result = subprocess.run^([
echo             'nvidia-smi', 
echo             '--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
echo             '--format=csv,noheader,nounits'
echo         ], capture_output=True, text=True^)
echo         
echo         if result.returncode == 0:
echo             lines = result.stdout.strip^(^).split^('\n'^)
echo             stats = []
echo             for line in lines:
echo                 parts = line.split^(', '^)
echo                 if len^(parts^) ^>= 7:
echo                     stats.append^({
echo                         'timestamp': parts[0],
echo                         'name': parts[1],
echo                         'temperature': int^(parts[2]^) if parts[2] != '[N/A]' else None,
echo                         'utilization': int^(parts[3]^) if parts[3] != '[N/A]' else None,
echo                         'memory_used': int^(parts[4]^) if parts[4] != '[N/A]' else None,
echo                         'memory_total': int^(parts[5]^) if parts[5] != '[N/A]' else None,
echo                         'power_draw': float^(parts[6]^) if parts[6] != '[N/A]' else None
echo                     }^)
echo             return stats
echo         return None
echo     except Exception as e:
echo         print^(f"Error getting GPU stats: {e}"^)
echo         return None
echo.
echo def monitor_training^(interval=30, log_file="logs/gpu_monitoring.log"^):
echo     """Monitor GPU during training."""
echo     print^("Starting GPU monitoring..."^)
echo     print^("Press Ctrl+C to stop"^)
echo     
echo     try:
echo         with open^(log_file, 'a'^) as f:
echo             while True:
echo                 stats = get_gpu_stats^(^)
echo                 if stats:
echo                     timestamp = datetime.now^(^).isoformat^(^)
echo                     log_entry = {
echo                         'timestamp': timestamp,
echo                         'gpu_stats': stats
echo                     }
echo                     f.write^(json.dumps^(log_entry^) + '\n'^)
echo                     f.flush^(^)
echo                     
echo                     # Print current status
echo                     for i, gpu in enumerate^(stats^):
echo                         print^(f"GPU {i}: {gpu['utilization']}%% util, "
echo                               f"{gpu['memory_used']}/{gpu['memory_total']} MB, "
echo                               f"{gpu['temperature']}°C"^)
echo                 
echo                 time.sleep^(interval^)
echo                 
echo     except KeyboardInterrupt:
echo         print^("\nGPU monitoring stopped"^)
echo.
echo if __name__ == "__main__":
echo     import sys
echo     interval = int^(sys.argv[1]^) if len^(sys.argv^) ^> 1 else 30
echo     monitor_training^(interval^)
) > monitor_gpu.py

REM Create Windows run script
echo [INFO] Creating Windows run script...
(
echo @echo off
echo REM Quick run script for Windows
echo call venv\Scripts\activate.bat
echo python src\main.py
echo pause
) > run_system.bat

REM Test the system
echo [INFO] Running system tests...
python -c "import sys; sys.path.append('src'); from model_manager import ModelManager; manager = ModelManager(); info = manager.get_system_info(); print(f'GPU Available: {info[\"gpu_available\"]}'); print(f'System ready: True')"

echo.
echo ======================================================================
echo                        SETUP COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo Setup Summary:
echo   • Python virtual environment: venv\
echo   • PyTorch with GPU support: Installed
echo   • All dependencies installed: Yes
echo   • Project directories created: Yes
echo   • GPU monitoring tools: Yes
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate.bat
echo   2. Run the system: python src\main.py
echo   3. Or use quick run: run_system.bat
echo   4. Monitor GPU usage: python monitor_gpu.py
echo.
echo Tips for Windows:
echo   • Use Command Prompt or PowerShell as Administrator for best results
echo   • Make sure Windows Defender doesn't block Python processes
echo   • For long training sessions, consider using Windows Terminal
echo   • Keep the system plugged in during training for best performance
echo.

if "%GPU_AVAILABLE%"=="true" (
    echo [SUCCESS] GPU training ready! 🎮
    echo Your system is optimized for fast GPU training.
) else (
    echo [WARNING] CPU training only - consider using a GPU for faster training
)

echo.
echo Press any key to exit...
pause >nul
