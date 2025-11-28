# 📦 Installation Guide - HG 585 LLM Evaluation System

Choose the installation option that best fits your needs and hardware capabilities.

## 🚀 Quick Start (Direct Installation - Recommended)

### **Windows - Direct Installation** 🪟
**Best for**: Direct installation on your Windows workstation
**Models location**: `D:\llm_models\`
**No virtual environment needed**

```cmd
# Run the automated setup
setup_windows_direct.bat

# Or manual installation
pip install -r requirements-minimal.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### **Linux/macOS - Direct Installation** 🐧
**Best for**: Direct installation on your Linux/macOS system
**Models location**: `./models/`
**No virtual environment needed**

```bash
# Run the automated setup
chmod +x setup_linux_direct.sh
./setup_linux_direct.sh

# Or manual installation
pip3 install -r requirements-minimal.txt --user
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 --user
```

---

## 📦 Installation Types

### **Option 1: Minimal Installation** ⚡
**Best for**: Testing, basic functionality, limited hardware
**Size**: ~2-3GB
**Time**: 5-10 minutes

```bash
pip install -r requirements-minimal.txt
```

**What you get:**
- ✅ Document processing (HG585.pdf)
- ✅ Model downloading and management
- ✅ Basic training capabilities
- ✅ Interactive menu system
- ❌ Advanced evaluation metrics
- ❌ Professional visualizations
- ❌ Web interfaces

---

### **Option 2: Full Installation** 🎯
**Best for**: Production use, complete evaluation, research
**Size**: ~8-10GB
**Time**: 15-30 minutes

```bash
pip install -r requirements-full.txt
```

**What you get:**
- ✅ Everything from minimal installation
- ✅ Advanced evaluation metrics (BLEU, ROUGE, etc.)
- ✅ Professional visualizations and charts
- ✅ Complete data science stack
- ✅ Enhanced document processing
- ✅ Performance optimizations

---

### **Option 3: Chat Interface Only** 💬
**Best for**: Using already trained models, deployment
**Size**: ~3-4GB
**Time**: 10-15 minutes

```bash
pip install -r requirements-chat.txt
```

**What you get:**
- ✅ Model inference capabilities
- ✅ Web-based chat interfaces (Gradio, Streamlit)
- ✅ API server functionality
- ✅ Trained model deployment
- ❌ Training capabilities
- ❌ Document processing

---

## 🎮 GPU Support

### **For NVIDIA GPUs (Recommended):**

#### **CUDA 11.8 (Most Compatible):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### **CUDA 12.1 (Latest):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### **Verify GPU Installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

---

## 📁 Models Storage

### **Windows:**
- **Default location**: `D:\llm_models\`
- **Automatic creation**: System creates directory if it doesn't exist
- **Organized structure**: Each model in its own subdirectory

### **Linux/macOS:**
- **Default location**: `./models/`
- **Automatic creation**: System creates directory if it doesn't exist
- **Organized structure**: Each model in its own subdirectory

### **Custom Location:**
You can specify a custom models directory by modifying the ModelManager initialization in the code.

---

## 📋 System Requirements

### **Minimal Installation:**
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 20GB+ free space (plus models storage)
- **CPU**: 4+ cores
- **GPU**: Optional (CPU training supported)

### **Full Installation:**
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ free space (plus models storage)
- **CPU**: 8+ cores recommended
- **GPU**: NVIDIA with 8GB+ VRAM recommended

### **For GPU Training:**
- **NVIDIA GPU**: RTX 4060 Ti or better
- **VRAM**: 8GB minimum, 16GB+ recommended
- **CUDA**: 11.8 or 12.1
- **Drivers**: Latest NVIDIA drivers

---

## 🛠️ Platform-Specific Setup

### **Windows (Direct Installation):**
```cmd
# Automated setup (recommended)
setup_windows_direct.bat

# Manual installation
pip install -r requirements-minimal.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run system
python src/main.py
```

### **Linux/macOS (Direct Installation):**
```bash
# Automated setup (recommended)
chmod +x setup_linux_direct.sh
./setup_linux_direct.sh

# Manual installation
pip3 install -r requirements-minimal.txt --user
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 --user

# Run system
python3 src/main.py
```

### **Server with Virtual Environment (Advanced):**
```bash
# Use the original GPU setup script
chmod +x gpu_setup.sh
./gpu_setup.sh

# Activate environment and run
source venv/bin/activate
python src/main.py
```

---

## 📊 Installation Comparison

| Feature | Direct Install | Virtual Env | Chat Only |
|---------|----------------|-------------|-----------|
| **Setup Complexity** | Simple | Advanced | Simple |
| **System Integration** | Direct | Isolated | Direct |
| **Models Location** | D:\llm_models\ | ./models/ | ./models/ |
| **Environment** | System Python | venv | System Python |
| **Best For** | Workstations | Servers | Deployment |

---

## 🔧 Troubleshooting

### **Common Issues:**

#### **"Models directory not found"**
- **Windows**: System automatically creates `D:\llm_models\`
- **Linux**: System automatically creates `./models/`
- **Custom**: Modify ModelManager initialization

#### **"CUDA out of memory"**
```bash
# System automatically adjusts based on available VRAM
# Check GPU status in menu option 9
```

#### **"No module named 'torch'"**
```bash
# Windows
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Linux/macOS
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 --user
```

#### **"Permission denied" on Windows**
```cmd
# Run Command Prompt as Administrator
# Or use setup_windows_direct.bat
```

#### **Import errors on Linux**
```bash
# Ensure ~/.local/bin is in PATH
export PATH=$PATH:~/.local/bin
```

---

## 💡 Recommendations

### **For Windows Users:**
1. Use **setup_windows_direct.bat** for automated setup
2. Models automatically stored in `D:\llm_models\`
3. No virtual environment needed
4. Direct system integration

### **For Linux/macOS Users:**
1. Use **setup_linux_direct.sh** for automated setup
2. Packages installed with `--user` flag
3. No virtual environment needed
4. System-wide availability

### **For Servers:**
1. Use **gpu_setup.sh** for isolated environment
2. Virtual environment for better control
3. Advanced monitoring and optimization
4. Production deployment ready

---

## 🆘 Support

- **System Status**: Run `python src/main.py` → Option 9
- **GPU Check**: Run `nvidia-smi` (Windows/Linux)
- **Models Location**: Automatically displayed on startup
- **Performance Tips**: See README.md for hardware-specific optimizations

---

**Ready to start? Choose your platform-specific setup above! 🚀**
