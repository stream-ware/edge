#!/bin/bash
#
# Streamware Jetson - Full Installation Script
# For NVIDIA Jetson Orin Nano 8GB with JetPack 6.0+
#

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Streamware Jetson - Installation Script                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: Not running on NVIDIA Jetson"
    echo "   Some features may not work correctly."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo ""
echo "Step 1/6: System packages"
echo "========================="

sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    espeak-ng \
    libespeak-ng-dev \
    libopencv-dev \
    python3-opencv

echo ""
echo "Step 2/6: Python virtual environment"
echo "====================================="

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

echo ""
echo "Step 3/6: Python dependencies"
echo "=============================="

# Install PyTorch for Jetson (if not already installed)
# JetPack 6 comes with PyTorch, but we ensure it's available in venv
pip install numpy pyyaml

# Audio
pip install sounddevice soundfile pyaudio webrtcvad

# STT - Faster Whisper
pip install faster-whisper ctranslate2

# Vision - ultralytics for YOLOv8
pip install ultralytics opencv-python

# LLM client
pip install httpx ollama

# TTS
pip install piper-tts

# Async utilities
pip install aiofiles websockets aiohttp

# Development
pip install pytest pytest-asyncio

echo ""
echo "Step 4/6: Ollama installation"
echo "=============================="

if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed"
fi

# Start Ollama service
echo "Starting Ollama service..."
sudo systemctl enable ollama || true
sudo systemctl start ollama || true

# Wait for Ollama to be ready
sleep 5

# Pull recommended model
echo "Pulling phi3:mini model..."
ollama pull phi3:mini || echo "⚠️ Could not pull model. Run manually: ollama pull phi3:mini"

echo ""
echo "Step 5/6: Download models"
echo "=========================="

# Piper TTS Polish voice
chmod +x scripts/download_piper_pl.sh
./scripts/download_piper_pl.sh medium gosia || echo "⚠️ Piper download failed"

# YOLOv8 - download and optionally export to TensorRT
echo "Downloading YOLOv8n..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo ""
echo "Export YOLOv8 to TensorRT? (recommended for best performance)"
read -p "This takes ~5 minutes. Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 scripts/export_yolo_tensorrt.py --model yolov8n
fi

echo ""
echo "Step 6/6: Permissions and setup"
echo "================================"

# Audio permissions
sudo usermod -aG audio $USER || true

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py
chmod +x main.py

# Create logs directory
mkdir -p logs

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Installation Complete!                                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "To run Streamware Jetson:"
echo ""
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Optional: Run benchmark"
echo "  python scripts/benchmark.py"
echo ""
echo "Notes:"
echo "  - If audio doesn't work, log out and back in (for audio group)"
echo "  - Ensure camera is connected before running"
echo "  - For best performance, use MAX power mode:"
echo "    sudo nvpmodel -m 0"
echo ""
