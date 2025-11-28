# 📦 Installation Guide - HG 585 LLM Evaluation System

Choose the installation option that best fits your needs and hardware capabilities.

## 🚀 Quick Start (Recommended)

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

## 📋 System Requirements

### **Minimal Installation:**
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 20GB+ free space
- **CPU**: 4+ cores
- **GPU**: Optional (CPU training supported)

### **Full Installation:**
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ free space
- **CPU**: 8+ cores recommended
- **GPU**: NVIDIA with 8GB+ VRAM recommended

### **For GPU Training:**
- **NVIDIA GPU**: RTX 4060 Ti or better
- **VRAM**: 8GB minimum, 16GB+ recommended
- **CUDA**: 11.8 or 12.1
- **Drivers**: Latest NVIDIA drivers

---

## 🛠️ Platform-Specific Setup

### **Windows:**
```cmd
# Run the automated setup
setup_windows.bat

# Or manual installation
pip install -r requirements-minimal.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### **Linux/macOS:**
```bash
# Run the automated setup
chmod +x gpu_setup.sh
./gpu_setup.sh

# Or manual installation
pip install -r requirements-minimal.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Installation Comparison

| Feature | Minimal | Full | Chat Only |
|---------|---------|------|-----------|
| **Size** | 2-3GB | 8-10GB | 3-4GB |
| **Install Time** | 5-10 min | 15-30 min | 10-15 min |
| **Document Processing** | ✅ | ✅ | ❌ |
| **Model Training** | ✅ | ✅ | ❌ |
| **Basic Evaluation** | ✅ | ✅ | ❌ |
| **Advanced Metrics** | ❌ | ✅ | ❌ |
| **Visualizations** | ❌ | ✅ | ❌ |
| **Web Interfaces** | ❌ | ✅ | ✅ |
| **Model Inference** | ✅ | ✅ | ✅ |

---

## 🔧 Troubleshooting

### **Common Issues:**

#### **"CUDA out of memory"**
```bash
# Use smaller batch size
# System automatically adjusts based on available VRAM
```

#### **"No module named 'torch'"**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### **"Permission denied" on Windows**
```cmd
# Run Command Prompt as Administrator
# Or use setup_windows.bat
```

#### **Slow downloads**
```bash
# Install faster download support
pip install huggingface-hub[hf_xet]
```

---

## 💡 Recommendations

### **For Beginners:**
1. Start with **Minimal Installation**
2. Test basic functionality
3. Upgrade to Full if needed

### **For Researchers:**
1. Use **Full Installation**
2. Install on GPU server
3. Enable all optimizations

### **For Deployment:**
1. Train with **Full Installation**
2. Deploy with **Chat Interface Only**
3. Use exported models

---

## 🆘 Support

- **System Status**: Run `python src/main.py` → Option 9
- **GPU Check**: Run `nvidia-smi` (Windows/Linux)
- **Requirements Check**: Each requirements file includes size estimates
- **Performance Tips**: See README.md for hardware-specific optimizations

---

**Ready to start? Choose your installation option above! 🚀**
