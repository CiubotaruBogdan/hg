# 🤖 HG 585 LLM Evaluation System - Windows Edition

A comprehensive system for training and evaluating Large Language Models on Romanian government document HG 585/2002 "National Standards on the Protection of Classified Information". This version is adapted for **Windows Server** environments.

## ✨ Features

- **5 LLM Models**: Llama 3.1, Qwen 3, DeepSeek-V2, Gemma 3, GPT-OSS 20B
- **Interactive Menu**: Simple numbered interface - no command line arguments
- **GPU Optimization**: Native CUDA support with automatic optimization
- **LoRA Training**: Parameter-efficient fine-tuning
- **Real Evaluation**: Before/after training comparison with multiple metrics
- **Professional Visualizations**: Charts and reports for model comparison
- **Model Export**: Save trained models for future use
- **Disk Size Monitoring**: Track model storage usage
- **HuggingFace Integration**: Seamless model downloading and authentication

## 🚀 Quick Start

### **Automated Setup (Recommended)**

1.  **Download and Extract:**
    *   Download the project from the GitHub repository as a ZIP file.
    *   Extract the contents to a directory, for example: `C:\hg-llm-evaluation`

2.  **Run Setup Script:**
    *   Open a Command Prompt (`cmd.exe`) and navigate to the project directory.
    *   Run the setup script:
        ```cmd
        cd C:\hg-llm-evaluation
        setup.bat
        ```
    *   The script will automatically check for Python, detect your GPU, create a virtual environment, and install all necessary dependencies.

3.  **Run the System:**
    *   After the setup is complete, you can run the main application using the provided batch file:
        ```cmd
        run_system.bat
        ```
    *   Alternatively, you can double-click `run_system.bat` in File Explorer.

### **Simple Usage:**
1.  **Run `run_system.bat`**
2.  **Select menu options** 1-6 for the complete workflow.
3.  **No command line arguments** are needed!

## 📋 System Requirements

### **Minimum:**
- **OS**: Windows 10 (1909+) / Windows 11 / Windows Server 2019+
- **CPU**: 4+ cores
- **RAM**: 16GB+ (32GB+ recommended)
- **Storage**: 100GB+ free space
- **Python**: 3.8+

### **Recommended (GPU Training):**
- **GPU**: NVIDIA with 8GB+ VRAM
- **CUDA**: 11.8+ or 12.1+
- **Examples**: RTX 4060 Ti, RTX 4070 Ti, RTX 4090, A100

## 🎮 GPU Performance

### **Training Speed (per model):**
```
CPU Only:     2-8 hours per model
RTX 4060 Ti:  30-60 minutes per model
RTX 4070 Ti:  20-45 minutes per model  
RTX 4090:     10-30 minutes per model
A100:         8-20 minutes per model
```

### **Automatic GPU Optimization:**
- **Memory-based batch sizing** - automatic based on VRAM
- **Mixed precision training** - 2x speed improvement
- **Gradient checkpointing** - memory optimization
- **Temperature monitoring** - safety checks
- **Performance recommendations** - tailored to your GPU

## 🤖 Selected Models

The system evaluates and compares **5 state-of-the-art LLMs**:

1. **Llama 3.1** (Meta) - General-purpose flagship model 🔒
2. **Qwen 3** (Alibaba) - Multilingual powerhouse  
3. **DeepSeek-V2** (DeepSeek AI) - Specialized reasoning model
4. **Gemma 3** (Google) - Latest open-source model 🔒
5. **GPT-OSS 20B** (Ollama) - Open-source GPT model via Ollama

*🔒 = Requires HuggingFace authentication*

## 📱 Interactive Menu Interface

```
======================================================================
           HG 585 LLM EVALUATION SYSTEM
         Interactive Menu Interface
======================================================================

Please select an option:

  1. Preprocess Data Source (HG585.pdf)
  2. Download Models
  3. Check Models Status
  4. Evaluate Initial Models (Before Training)
  5. Train Models
  6. Evaluate Models Post Training

  7. Run Complete Workflow (Steps 1-6)
  8. Create Visualizations
  9. View System Status
  0. Exit

Enter your choice (0-9): 
```

## 🔐 Authentication Setup

### **HuggingFace (for Llama 3.1 & Gemma 3):**
```cmd
# Get token from: https://huggingface.co/settings/tokens
huggingface-cli login

# Or use menu option 2 → 0 for guided setup
```

### **Ollama (for GPT-OSS):**
```cmd
# Download from https://ollama.ai/download/windows
# After installation, run:
ollama pull gpt-oss:20b
```

## 📊 Evaluation Metrics

The system computes comprehensive metrics:

- **Exact Match**: Percentage of exactly matching answers
- **F1 Score**: Token-level overlap between prediction and reference
- **BLEU Score**: Machine translation quality metric
- **ROUGE-L**: Longest common subsequence-based metric
- **Semantic Similarity**: Content-based similarity measure
- **Romanian Accuracy**: Romanian-specific character and diacritic accuracy

## 📁 Project Structure

```
llm-evaluation/
├── data/
│   ├── raw/                 # HG585.pdf document
│   └── processed/           # Generated training data
├── src/
│   ├── preprocessing/       # Document processing
│   ├── training/           # Model training scripts
│   ├── evaluation/         # Evaluation and metrics
│   ├── model_manager.py    # Model downloading and management
│   └── main.py            # Interactive menu interface
├── models/                 # Downloaded and trained models
├── results/               # Evaluation results and visualizations
├── logs/                  # Training and system logs
├── setup.bat              # Windows GPU setup
└── requirements.txt      # Python dependencies
```

## 🔧 Advanced Usage

### **Individual Steps:**
```cmd
# Activate environment
venv\Scripts\activate.bat

# Run specific steps
python src\main.py  # Then select menu options
```

### **GPU Monitoring:**
```cmd
# Monitor GPU during training
python monitor_gpu.py

# Check system status
python src\main.py  # Option 9
```

### **Model Export:**
After training, models are automatically exported to:
```
models/{model_name}_trained_export/
├── model/                  # Complete trained model
├── tokenizer/             # Tokenizer files
├── lora_adapters/         # LoRA adapter weights
├── export_metadata.json  # Export information
└── usage_example.py      # Ready-to-use script
```

## 📈 Results and Visualizations

### **Generated Reports:**
- `results/evaluation_results.json` - Complete metrics data
- `results/comparison_report.md` - Detailed analysis
- `results/visualizations/` - Professional charts

### **Visualization Types:**
- **Metrics Comparison** - Bar charts of all metrics
- **Training Improvement** - Before/after analysis
- **Model Ranking** - Performance leaderboard
- **Performance Radar** - Multi-dimensional comparison

## 🔧 Troubleshooting

### **GPU Issues:**
```cmd
# Check GPU availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA not available, system automatically uses CPU
```

### **Memory Issues:**
- System automatically adjusts batch size based on available VRAM
- Use gradient checkpointing (enabled by default)
- Close other GPU applications during training

### **Model Download Issues:**
- Check internet connection
- Verify HuggingFace authentication for protected models
- System provides clear error messages and fallback options

## 💡 Tips for Best Performance

### **Hardware:**
1.  **Use GPU** for 10-50x faster training
2.  **SSD storage** for faster model loading
3.  **Adequate cooling** - keep GPU under 80°C
4.  **Sufficient VRAM** - 8GB minimum, 16GB+ recommended

### **Software:**
1.  **Close unnecessary applications** during training
2.  **Use virtual environments** to avoid conflicts
3.  **Monitor system resources** during training
4.  **Regular driver updates** for optimal GPU performance

## 🆘 Support

- **GitHub Issues**: https://github.com/CiubotaruBogdan/hg/issues
- **System Check**: Use menu option 9 for comprehensive status
- **GPU Setup**: Run `setup.bat` for automated configuration
- **Logs**: Check `logs/` directory for detailed error information

---

**Ready to evaluate LLMs on Romanian legal documents! 🇷🇴🤖**


## 🔧 Advanced Windows Setup & Troubleshooting

For more detailed Windows setup instructions, including manual installation, GPU driver setup, and performance optimization, please refer to the following sections.

### Manual Installation Steps

1.  **Install Python:**
    *   Download and install Python 3.8+ from [python.org](https://python.org/downloads/).
    *   **Important:** Check the box for "Add Python to PATH" during installation.

2.  **Create Virtual Environment:**
    ```cmd
    python -m venv venv
    venv\Scripts\activate.bat
    ```

3.  **Install PyTorch:**
    *   For GPU (CUDA 11.8):
        ```cmd
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   For CPU only:
        ```cmd
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

4.  **Install Dependencies:**
    ```cmd
    pip install -r requirements.txt
    ```

### Windows Performance Optimization

*   **PowerShell Execution Policy:** If you encounter script execution errors in PowerShell, run:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
*   **Windows Defender Exclusions:** For better performance, add the project folder (e.g., `C:\hg-llm-evaluation`) and Hugging Face cache folder (`C:\Users\{username}\.cache\huggingface`) to Windows Defender exclusions.
*   **High Performance Power Plan:** Ensure your system is using the High Performance power plan, especially during training.

### Common Troubleshooting

*   **"Python not found"**: Reinstall Python and ensure "Add to PATH" is checked. Or, manually add the Python installation and scripts directory to your system's PATH environment variables.
*   **"CUDA out of memory"**: The system automatically adjusts batch sizes, but if you still encounter this, try closing other GPU-intensive applications or restarting your system to clear VRAM.
*   **"Access denied" errors**: Run your Command Prompt or PowerShell as an Administrator.
