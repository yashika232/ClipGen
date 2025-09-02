#!/bin/bash
# Installation script for 3D Avatar Pipeline on macOS

echo "🎭 Installing 3D Avatar Pipeline Dependencies"
echo "============================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew install cmake
brew install colmap
brew install opencv

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "🐍 Using conda for Python environment..."
    
    # Create conda environment
    conda create -n avatar3d python=3.9 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate avatar3d
    
    # Install PyTorch with MPS support
    conda install pytorch torchvision torchaudio -c pytorch -y
    
else
    echo "🐍 Using pip for Python packages..."
    
    # Create virtual environment
    python3 -m venv avatar3d_env
    source avatar3d_env/bin/activate
fi

# Install Python packages
echo "📦 Installing Python packages..."
pip install -r requirements_3d.txt

# Install additional packages for development
pip install click loguru

# Check PyTorch MPS support
echo "🧪 Testing PyTorch MPS support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ MPS acceleration ready')
else:
    print('⚠️  MPS not available, using CPU')
"

# Test MediaPipe
echo "🧪 Testing MediaPipe..."
python3 -c "
import mediapipe as mp
print(f'MediaPipe version: {mp.__version__}')
print('✅ MediaPipe ready')
"

# Optional: Download Blender (for rigging)
echo "📦 Blender installation (optional)..."
if ! command -v blender &> /dev/null; then
    echo "⚠️  Blender not found. Download from: https://www.blender.org/"
    echo "   Or install via: brew install --cask blender"
else
    echo "✅ Blender found"
fi

# Optional: OpenMVS (advanced users)
echo "📦 OpenMVS installation (optional)..."
if ! command -v DensifyPointCloud &> /dev/null; then
    echo "⚠️  OpenMVS not found. Build from: https://github.com/cdcseacave/openMVS"
    echo "   This is optional for basic testing"
else
    echo "✅ OpenMVS found"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Run tests: python test_pipeline.py"
echo "2. Setup models: python setup_models.py"
echo "3. Try CLI: python avatar_pipeline.py --help"
echo ""
echo "If using conda, activate environment with:"
echo "conda activate avatar3d"