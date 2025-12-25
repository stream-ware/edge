#!/bin/bash
#
# Download Piper TTS Polish voice
# https://github.com/rhasspy/piper
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models/piper"

# Domyślna jakość
QUALITY="${1:-medium}"

# Głosy polskie
# - gosia (kobieta)
# - darkman (mężczyzna) 
VOICE="${2:-gosia}"

# URL do modeli Piper
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL"

echo "==================================="
echo "Piper TTS - Polski głos"
echo "==================================="
echo "Głos: $VOICE"
echo "Jakość: $QUALITY"
echo ""

# Utwórz katalog
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Nazwa modelu
MODEL_NAME="pl_PL-${VOICE}-${QUALITY}"
ONNX_FILE="${MODEL_NAME}.onnx"
JSON_FILE="${MODEL_NAME}.onnx.json"

# Sprawdź czy już istnieje
if [ -f "$ONNX_FILE" ] && [ -f "$JSON_FILE" ]; then
    echo "✅ Model już istnieje: $MODEL_NAME"
    exit 0
fi

echo "📥 Pobieranie modelu..."

# Download ONNX
if [ ! -f "$ONNX_FILE" ]; then
    echo "  - $ONNX_FILE"
    wget -q --show-progress "${BASE_URL}/${VOICE}/${QUALITY}/${ONNX_FILE}"
fi

# Download config JSON
if [ ! -f "$JSON_FILE" ]; then
    echo "  - $JSON_FILE"
    wget -q --show-progress "${BASE_URL}/${VOICE}/${QUALITY}/${JSON_FILE}"
fi

echo ""
echo "✅ Pobrano: $MODEL_NAME"
echo ""

# Test modelu
echo "🔊 Test syntezy..."
if command -v piper &> /dev/null; then
    echo "Test" | piper --model "$ONNX_FILE" --output_file /tmp/piper_test.wav
    echo "✅ Test OK - /tmp/piper_test.wav"
else
    echo "⚠️ piper CLI nie zainstalowany"
    echo "   Zainstaluj: pip install piper-tts"
fi

echo ""
echo "==================================="
echo "Gotowe!"
echo ""
echo "Użycie w config.yaml:"
echo "  tts:"
echo "    model: \"$MODEL_NAME\""
echo "    model_path: \"models/piper/$ONNX_FILE\""
echo "    config_path: \"models/piper/$JSON_FILE\""
echo "==================================="
