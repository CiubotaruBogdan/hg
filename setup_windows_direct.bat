@echo off
echo ======================================================================
echo           HG 585 LLM Evaluation System - Direct Installation
echo                    Windows Setup (No Virtual Environment)
echo ======================================================================
echo.

echo 📋 This script will install the LLM evaluation system directly on your system.
echo 📁 Models will be stored in: D:\llm_models\
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found:
python --version

:: Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not available.
    echo Please ensure pip is installed with Python.
    pause
    exit /b 1
)

echo ✅ pip found:
pip --version
echo.

:: Ask user for installation type
echo 📦 Choose installation type:
echo.
echo   1. Minimal (2-3GB) - Basic functionality, testing
echo   2. Full (8-10GB) - Complete features, production ready
echo   3. Chat Only (3-4GB) - For using already trained models
echo.
set /p choice="Enter your choice (1/2/3): "

if "%choice%"=="1" (
    set requirements_file=requirements-minimal.txt
    echo 📥 Installing minimal requirements...
) else if "%choice%"=="2" (
    set requirements_file=requirements-full.txt
    echo 📥 Installing full requirements...
) else if "%choice%"=="3" (
    set requirements_file=requirements-chat.txt
    echo 📥 Installing chat requirements...
) else (
    echo ❌ Invalid choice. Using minimal installation.
    set requirements_file=requirements-minimal.txt
)

echo.
echo 🔄 Installing Python packages...
echo This may take 10-30 minutes depending on your internet connection.
echo.

:: Install requirements
pip install -r %requirements_file%
if %errorlevel% neq 0 (
    echo.
    echo ❌ Installation failed. Please check the error messages above.
    echo.
    echo 💡 Common solutions:
    echo    - Run as Administrator
    echo    - Check internet connection
    echo    - Update pip: python -m pip install --upgrade pip
    pause
    exit /b 1
)

echo.
echo 🎮 Installing GPU support (CUDA)...
echo.

:: Ask about GPU support
set /p gpu_choice="Do you have an NVIDIA GPU and want GPU acceleration? (y/n): "
if /i "%gpu_choice%"=="y" (
    echo 📥 Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if %errorlevel% neq 0 (
        echo ⚠️  GPU installation failed, but CPU version should work.
    ) else (
        echo ✅ GPU support installed successfully!
    )
) else (
    echo ℹ️  Skipping GPU support. CPU-only training will be used.
)

:: Create models directory
echo.
echo 📁 Creating models directory...
if not exist "D:\llm_models" (
    mkdir "D:\llm_models"
    echo ✅ Created D:\llm_models\
) else (
    echo ℹ️  D:\llm_models\ already exists
)

:: Test the installation
echo.
echo 🧪 Testing installation...
python -c "import torch, transformers; print('✅ Core libraries imported successfully')" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  Some libraries may not be fully installed, but basic functionality should work.
)

:: Create run script
echo.
echo 📝 Creating run script...
echo @echo off > run_llm_system.bat
echo cd /d "%~dp0" >> run_llm_system.bat
echo python src/main.py >> run_llm_system.bat
echo pause >> run_llm_system.bat

echo ✅ Created run_llm_system.bat for easy launching

echo.
echo ======================================================================
echo                          🎉 INSTALLATION COMPLETE!
echo ======================================================================
echo.
echo 🚀 To start the system:
echo    • Double-click: run_llm_system.bat
echo    • Or run: python src/main.py
echo.
echo 📁 Models will be downloaded to: D:\llm_models\
echo 📊 System status: Use menu option 9 to check GPU and system info
echo.
echo 💡 Next steps:
echo    1. Run the system using run_llm_system.bat
echo    2. Select option 2 to download models
echo    3. Select option 0 for HuggingFace authentication (if needed)
echo    4. Follow the 6-step workflow for training and evaluation
echo.
echo ======================================================================
pause
