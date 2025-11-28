#!/bin/bash

echo "======================================================================"
echo "           HG 585 LLM Evaluation System - Direct Installation"
echo "                    Linux/macOS Setup (No Virtual Environment)"
echo "======================================================================"
echo

echo "📋 This script will install the LLM evaluation system directly on your system."
echo "📁 Models will be stored in: ./models/"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "Please install Python 3.8+ using your package manager:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

echo "✅ Python found:"
python3 --version

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not available."
    echo "Installing pip3..."
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install python3-pip
    else
        echo "Please install pip3 manually for your system."
        exit 1
    fi
fi

echo "✅ pip3 found:"
pip3 --version
echo

# Ask user for installation type
echo "📦 Choose installation type:"
echo
echo "  1. Minimal (2-3GB) - Basic functionality, testing"
echo "  2. Full (8-10GB) - Complete features, production ready"
echo "  3. Chat Only (3-4GB) - For using already trained models"
echo
read -p "Enter your choice (1/2/3): " choice

case $choice in
    1)
        requirements_file="requirements-minimal.txt"
        echo "📥 Installing minimal requirements..."
        ;;
    2)
        requirements_file="requirements-full.txt"
        echo "📥 Installing full requirements..."
        ;;
    3)
        requirements_file="requirements-chat.txt"
        echo "📥 Installing chat requirements..."
        ;;
    *)
        echo "❌ Invalid choice. Using minimal installation."
        requirements_file="requirements-minimal.txt"
        ;;
esac

echo
echo "🔄 Installing Python packages directly to system..."
echo "This may take 10-30 minutes depending on your internet connection."
echo

# Install requirements directly to system
pip3 install -r "$requirements_file" --user
if [ $? -ne 0 ]; then
    echo
    echo "❌ Installation failed. Please check the error messages above."
    echo
    echo "💡 Common solutions:"
    echo "   - Check internet connection"
    echo "   - Update pip: python3 -m pip install --upgrade pip --user"
    echo "   - Install build tools: sudo apt install build-essential (Ubuntu/Debian)"
    exit 1
fi

echo
echo "🎮 Installing GPU support (CUDA)..."
echo

# Ask about GPU support
read -p "Do you have an NVIDIA GPU and want GPU acceleration? (y/n): " gpu_choice
if [[ $gpu_choice =~ ^[Yy]$ ]]; then
    echo "📥 Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --user
    if [ $? -ne 0 ]; then
        echo "⚠️  GPU installation failed, but CPU version should work."
    else
        echo "✅ GPU support installed successfully!"
    fi
else
    echo "ℹ️  Skipping GPU support. CPU-only training will be used."
fi

# Create models directory
echo
echo "📁 Creating models directory..."
if [ ! -d "./models" ]; then
    mkdir -p "./models"
    echo "✅ Created ./models/"
else
    echo "ℹ️  ./models/ already exists"
fi

# Test the installation
echo
echo "🧪 Testing installation..."
python3 -c "import torch, transformers; print('✅ Core libraries imported successfully')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some libraries may not be fully installed, but basic functionality should work."
fi

# Create run script
echo
echo "📝 Creating run script..."
cat > run_llm_system.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 src/main.py
EOF

chmod +x run_llm_system.sh
echo "✅ Created run_llm_system.sh for easy launching"

echo
echo "======================================================================"
echo "                          🎉 INSTALLATION COMPLETE!"
echo "======================================================================"
echo
echo "🚀 To start the system:"
echo "   • Run: ./run_llm_system.sh"
echo "   • Or run: python3 src/main.py"
echo
echo "📁 Models will be downloaded to: ./models/"
echo "📊 System status: Use menu option 9 to check GPU and system info"
echo
echo "💡 Next steps:"
echo "   1. Run the system using ./run_llm_system.sh"
echo "   2. Select option 2 to download models"
echo "   3. Select option 0 for HuggingFace authentication (if needed)"
echo "   4. Follow the 6-step workflow for training and evaluation"
echo
echo "📋 Note: Packages installed with --user flag to avoid system conflicts"
echo "If you encounter import errors, ensure ~/.local/bin is in your PATH"
echo
echo "======================================================================"
