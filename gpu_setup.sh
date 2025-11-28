#!/bin/bash
# ========================================
# HG 585 LLM Evaluation System - GPU Setup
# Server deployment script with CUDA optimization
# ========================================

set -e  # Exit on any error

echo "🚀 Setting up HG 585 LLM Evaluation System for GPU training..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# System information
print_status "Checking system information..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

# Check GPU availability
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    GPU_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected or nvidia-smi not available"
    GPU_AVAILABLE=false
fi

# Check CUDA installation
print_status "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "CUDA $CUDA_VERSION detected"
else
    print_warning "CUDA not detected. PyTorch will install its own CUDA runtime."
fi

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
    print_success "Python $PYTHON_VERSION is compatible"
else
    print_error "Python 3.8+ required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
if [[ $GPU_AVAILABLE == true ]]; then
    # Install CUDA version of PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print_success "PyTorch with CUDA 11.8 support installed"
else
    # Install CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_warning "PyTorch CPU version installed (no GPU detected)"
fi

# Install other requirements
print_status "Installing other requirements..."
pip install -r requirements.txt

# Verify PyTorch GPU support
print_status "Verifying PyTorch GPU support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'    Compute capability: {props.major}.{props.minor}')
else:
    print('GPU training not available - will use CPU')
"

# Download NLTK data
print_status "Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download warning: {e}')
"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/raw data/processed models results results/visualizations logs

# Set up logging directory
print_status "Setting up logging..."
touch logs/training.log
touch logs/evaluation.log
touch logs/system.log

# Create GPU monitoring script
print_status "Creating GPU monitoring script..."
cat > monitor_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
GPU monitoring script for training sessions
"""
import time
import subprocess
import json
from datetime import datetime

def get_gpu_stats():
    """Get current GPU statistics."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            stats = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 7:
                    stats.append({
                        'timestamp': parts[0],
                        'name': parts[1],
                        'temperature': int(parts[2]) if parts[2] != '[N/A]' else None,
                        'utilization': int(parts[3]) if parts[3] != '[N/A]' else None,
                        'memory_used': int(parts[4]) if parts[4] != '[N/A]' else None,
                        'memory_total': int(parts[5]) if parts[5] != '[N/A]' else None,
                        'power_draw': float(parts[6]) if parts[6] != '[N/A]' else None
                    })
            return stats
        return None
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def monitor_training(interval=30, log_file="logs/gpu_monitoring.log"):
    """Monitor GPU during training."""
    print("Starting GPU monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        with open(log_file, 'a') as f:
            while True:
                stats = get_gpu_stats()
                if stats:
                    timestamp = datetime.now().isoformat()
                    log_entry = {
                        'timestamp': timestamp,
                        'gpu_stats': stats
                    }
                    f.write(json.dumps(log_entry) + '\n')
                    f.flush()
                    
                    # Print current status
                    for i, gpu in enumerate(stats):
                        print(f"GPU {i}: {gpu['utilization']}% util, "
                              f"{gpu['memory_used']}/{gpu['memory_total']} MB, "
                              f"{gpu['temperature']}°C")
                
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\nGPU monitoring stopped")

if __name__ == "__main__":
    import sys
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    monitor_training(interval)
EOF

chmod +x monitor_gpu.py

# Create system optimization script
print_status "Creating system optimization script..."
cat > optimize_system.sh << 'EOF'
#!/bin/bash
# System optimization for GPU training

echo "🔧 Optimizing system for GPU training..."

# Set GPU performance mode (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Setting GPU performance mode..."
    sudo nvidia-smi -pm 1 2>/dev/null || echo "Could not set persistence mode (may require sudo)"
    sudo nvidia-smi -ac $(nvidia-smi --query-supported-clocks=memory,graphics --format=csv,noheader,nounits | tail -1 | tr ',' ' ') 2>/dev/null || echo "Could not set application clocks (may require sudo)"
fi

# Increase file descriptor limits
echo "Optimizing file descriptor limits..."
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Set memory overcommit
echo "Optimizing memory settings..."
echo 'vm.overcommit_memory = 1' | sudo tee -a /etc/sysctl.conf

# Disable swap if available (for better GPU performance)
if [ -f /swapfile ] || [ -n "$(swapon --show)" ]; then
    echo "Warning: Swap is enabled. Consider disabling for better GPU performance:"
    echo "  sudo swapoff -a"
    echo "  # Comment out swap entries in /etc/fstab"
fi

echo "✅ System optimization completed"
echo "Note: Some changes may require a reboot to take effect"
EOF

chmod +x optimize_system.sh

# Test the system
print_status "Running system tests..."
python3 -c "
import sys
sys.path.append('src')

try:
    from model_manager import ModelManager
    manager = ModelManager()
    system_info = manager.get_system_info()
    
    print('System Test Results:')
    print(f'  GPU Available: {system_info[\"gpu_available\"]}')
    print(f'  HuggingFace Auth: {system_info[\"hf_auth\"]}')
    print(f'  Ollama Available: {system_info[\"ollama_available\"]}')
    
    if system_info['gpu_available']:
        print(f'  GPU Count: {system_info.get(\"gpu_count\", \"Unknown\")}')
        if 'gpu_memory' in system_info:
            for i, mem in enumerate(system_info['gpu_memory']):
                print(f'    GPU {i}: {mem} GB VRAM')
    
    print('✅ System test passed')
    
except Exception as e:
    print(f'⚠️  System test warning: {e}')
"

# Final setup summary
echo ""
print_success "🎉 GPU setup completed successfully!"
echo ""
echo "📋 Setup Summary:"
echo "  • Python virtual environment: venv/"
echo "  • PyTorch with GPU support: ✅"
echo "  • All dependencies installed: ✅"
echo "  • Project directories created: ✅"
echo "  • GPU monitoring tools: ✅"
echo ""
echo "🚀 Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run the system: python src/main.py"
echo "  3. Monitor GPU usage: python monitor_gpu.py"
echo "  4. Optimize system (optional): ./optimize_system.sh"
echo ""
echo "💡 Tips for server deployment:"
echo "  • Use 'screen' or 'tmux' for long training sessions"
echo "  • Monitor GPU temperature and usage during training"
echo "  • Consider using smaller batch sizes if you run out of VRAM"
echo "  • Enable mixed precision training for better performance"
echo ""

if [[ $GPU_AVAILABLE == true ]]; then
    print_success "GPU training ready! 🎮"
else
    print_warning "CPU training only - consider using a GPU server for faster training"
fi
