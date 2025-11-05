# 🪟 HG 585 LLM Evaluation System - Windows Setup Guide

## 📋 Windows Requirements

### **System Requirements:**
- **OS**: Windows 10 (1909+) / Windows 11
- **CPU**: Intel/AMD x64 with 8+ cores
- **RAM**: 16GB+ (32GB+ recommended for GPU training)
- **Storage**: 100GB+ free space (500GB+ for multiple models)
- **Python**: 3.8+ (3.10+ recommended)

### **GPU Requirements (Optional but Recommended):**
- **NVIDIA GPU** with 8GB+ VRAM
- **CUDA**: 11.8+ or 12.1+
- **Driver**: 525.60.13+ (for CUDA 12.1)

### **Recommended GPU Configurations:**
```
Entry Level:  RTX 4060 Ti (16GB) - Good for training
Mid Range:    RTX 4070 Ti (12GB) - Very good performance  
High End:     RTX 4090 (24GB)    - Excellent performance
Workstation:  RTX A6000 (48GB)   - Professional grade
```

## 🚀 Quick Windows Setup

### **Method 1: Automated Setup (Recommended)**

#### **1. Download System:**
- Download from: https://github.com/CiubotaruBogdan/hg/archive/main.zip
- Extract to `C:\hg-llm-evaluation\`

#### **2. Run Setup Script:**
```cmd
cd C:\hg-llm-evaluation
setup_windows.bat
```

The script will automatically:
- ✅ Check Python installation
- ✅ Detect NVIDIA GPU and CUDA
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA support
- ✅ Install all dependencies
- ✅ Set up monitoring tools
- ✅ Verify installation

#### **3. Run the System:**
```cmd
run_system.bat
```

### **Method 2: Manual Setup**

#### **1. Install Python:**
- Download from: https://python.org/downloads/
- **Important**: Check "Add Python to PATH" during installation
- Verify: `python --version`

#### **2. Install NVIDIA Drivers (for GPU):**
- Download latest drivers from: https://nvidia.com/drivers
- Or use GeForce Experience for automatic updates

#### **3. Create Virtual Environment:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### **4. Install PyTorch:**
```cmd
# For GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### **5. Install Dependencies:**
```cmd
pip install -r requirements.txt
```

#### **6. Run System:**
```cmd
python src\main.py
```

## 🎮 GPU Setup on Windows

### **CUDA Installation:**

#### **Option 1: CUDA Toolkit (Full Installation)**
```cmd
# Download CUDA 11.8 from:
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# Or CUDA 12.1 from:
# https://developer.nvidia.com/cuda-downloads
```

#### **Option 2: PyTorch CUDA (Recommended)**
```cmd
# PyTorch includes its own CUDA runtime
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Verify GPU Setup:**
```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### **GPU Performance on Windows:**
```cmd
# Check GPU status
nvidia-smi

# Monitor during training
python monitor_gpu.py
```

## 🔧 Windows-Specific Configuration

### **PowerShell Execution Policy:**
```powershell
# If you get execution policy errors
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Windows Defender Exclusions:**
Add these folders to Windows Defender exclusions for better performance:
- `C:\hg-llm-evaluation\`
- `C:\Users\{username}\.cache\huggingface\`
- `C:\Users\{username}\.cache\torch\`

### **Performance Optimization:**
```cmd
# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable Windows Game Mode (can interfere with training)
# Settings > Gaming > Game Mode > Off
```

## 📊 Training Performance on Windows

### **Expected Training Times (per model):**
```
CPU Only (16 cores):  2-4 hours per model
RTX 4060 Ti (16GB):   30-45 minutes per model
RTX 4070 Ti (12GB):   20-35 minutes per model  
RTX 4090 (24GB):      10-25 minutes per model
RTX A6000 (48GB):     8-15 minutes per model
```

### **Memory Usage:**
```
CPU Training:  8-16GB RAM + 2-4GB VRAM
GPU Training:  4-8GB RAM + 6-20GB VRAM (depending on model)
```

## 🔐 Authentication on Windows

### **HuggingFace Setup:**
```cmd
# Install HuggingFace CLI
pip install huggingface-hub[cli]

# Login with token
huggingface-cli login

# Or set environment variable
set HUGGINGFACE_HUB_TOKEN=your_token_here
```

### **Ollama Setup (for GPT-OSS):**
```cmd
# Download Ollama for Windows from:
# https://ollama.ai/download/windows

# Install and run
ollama pull gpt-oss:20b
ollama list
```

## 🚀 Running Training Sessions

### **Command Prompt Method:**
```cmd
cd C:\hg-llm-evaluation
venv\Scripts\activate.bat
python src\main.py
```

### **PowerShell Method:**
```powershell
cd C:\hg-llm-evaluation
.\venv\Scripts\Activate.ps1
python src\main.py
```

### **Quick Run Method:**
```cmd
# Double-click run_system.bat
# Or from command line:
run_system.bat
```

### **Background Training:**
```cmd
# For long training sessions, use Windows Terminal
# Or run in background with:
start /B python src\main.py
```

## 📈 Monitoring on Windows

### **GPU Monitoring:**
```cmd
# Real-time GPU stats
nvidia-smi -l 1

# Continuous logging
python monitor_gpu.py 30
```

### **Task Manager:**
- **Performance tab** - Monitor CPU, RAM, GPU usage
- **Details tab** - Monitor Python processes
- **GPU section** - Real-time VRAM usage

### **Windows Performance Toolkit:**
```cmd
# Install Windows Performance Toolkit (optional)
# Monitor detailed system performance during training
```

## 🔧 Troubleshooting Windows Issues

### **Common Problems:**

#### **"Python not found":**
```cmd
# Reinstall Python with "Add to PATH" checked
# Or manually add to PATH:
# C:\Users\{username}\AppData\Local\Programs\Python\Python311\
# C:\Users\{username}\AppData\Local\Programs\Python\Python311\Scripts\
```

#### **"CUDA out of memory":**
```cmd
# Reduce batch size in training configuration
# Close other GPU applications (games, browsers with hardware acceleration)
# Restart system to clear GPU memory
```

#### **"Access denied" errors:**
```cmd
# Run Command Prompt as Administrator
# Or change folder permissions for the project directory
```

#### **Slow training on GPU:**
```cmd
# Check GPU utilization in Task Manager
# Ensure GPU drivers are up to date
# Verify CUDA installation: nvidia-smi
# Check power settings (High Performance mode)
```

### **Windows-Specific Optimizations:**
```cmd
# Disable Windows Search indexing for project folder
# Set project folder to "Exclude from Windows Defender"
# Use SSD storage for better I/O performance
# Close unnecessary background applications
# Set Windows to High Performance power plan
```

## 💡 Windows Tips

### **Best Practices:**
1. **Use SSD storage** for faster model loading
2. **Close unnecessary applications** during training
3. **Use Windows Terminal** for better command line experience
4. **Monitor temperatures** - keep GPU under 80°C
5. **Use virtual environments** to avoid conflicts
6. **Regular Windows updates** for best GPU driver compatibility

### **Recommended Software:**
- **Windows Terminal** - Better command line interface
- **MSI Afterburner** - GPU monitoring and overclocking
- **HWiNFO64** - Detailed hardware monitoring
- **Process Explorer** - Advanced process monitoring

### **File Paths:**
```
Project:           C:\hg-llm-evaluation\
Virtual Env:       C:\hg-llm-evaluation\venv\
Models Cache:      C:\Users\{username}\.cache\huggingface\
Logs:              C:\hg-llm-evaluation\logs\
Results:           C:\hg-llm-evaluation\results\
```

## 📞 Windows Support

- **GitHub Issues**: https://github.com/CiubotaruBogdan/hg/issues
- **Windows Setup**: Run `setup_windows.bat` for automated setup
- **System Check**: Use option 3 in the main menu
- **GPU Issues**: Check NVIDIA Control Panel and drivers

---

**Ready for Windows GPU training! 🪟🎮**
