#!/usr/bin/env bash

# ==============================================================================
# Stable Audio 3 - Batch Audio Variation Processor Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Detect Python Environment
PYTHON_BIN="python3"
if [ -f "$SCRIPT_DIR/optimized/mlx/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/optimized/mlx/.venv/bin/python3"
elif [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
elif [ -f "$SCRIPT_DIR/venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"
fi

# 2. Check if arguments were passed directly via CLI
if [ "$#" -gt 0 ]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/batch_processor.py" "$@"
fi

# 3. Interactive Wizard Mode (if launched with no arguments)
echo ""
echo "======================================================================"
echo " 🎛️  STABLE AUDIO 3 - BATCH VARIATION PROCESSOR"
echo "======================================================================"
echo "Processes a folder of audio files (or single file) and generates"
echo "a dedicated subfolder with 5 noisier variants (+0.02 step)."
echo "======================================================================"
echo ""

# Input Folder or File
while true; do
    read -r -p "📁 Enter Audio Folder OR File Path (e.g. /path/to/Google Drive/Audio): " INPUT_PATH
    # Expand tilde ~ if present
    INPUT_PATH="${INPUT_PATH/#\~/$HOME}"
    if [ -d "$INPUT_PATH" ] || [ -f "$INPUT_PATH" ]; then
        break
    else
        echo "❌ Path not found. Please enter a valid directory or audio file path."
    fi
done

# Output Folder
read -r -p "📂 Enter Output Folder [default: ./batch_output]: " OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR:-./batch_output}"
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"

# Test Mode Option (if a directory was selected)
TEST_FLAG=""
if [ -d "$INPUT_PATH" ]; then
    read -r -p "🧪 Run quick TEST on 1 file only? [y/N]: " TEST_CHOICE
    if [[ "$TEST_CHOICE" =~ ^[Yy]$ ]]; then
        TEST_FLAG="--test"
    fi
fi

# Text Prompt
read -r -p "💬 Global Text Prompt (press Enter for unconditional): " PROMPT

# Base Noise Level
read -r -p "🎚️  Base Noise Level [default: 0.30]: " NOISE_LEVEL
NOISE_LEVEL="${NOISE_LEVEL:-0.30}"

# Denoising Steps
read -r -p "🔢 Denoising Steps [default: 8]: " STEPS
STEPS="${STEPS:-8}"

# CFG Scale
read -r -p "🎯 CFG Guidance Scale [default: 4.0]: " CFG
CFG="${CFG:-4.0}"

# Model Selection
echo ""
echo "Select Model Architecture:"
echo "  1) sm-music (50M DiT - fast, optimized for music/melodies)"
echo "  2) sm-sfx   (50M DiT - fast, optimized for sound effects)"
echo "  3) medium   (1.4B DiT - full quality, larger VRAM footprint)"
read -r -p "Select [1-3, default: 1]: " MODEL_CHOICE

case "$MODEL_CHOICE" in
    2) MODEL_NAME="sm-sfx" ;;
    3) MODEL_NAME="medium" ;;
    *) MODEL_NAME="sm-music" ;;
esac

echo ""
echo "🚀 Starting Processor..."
echo "----------------------------------------------------------------------"

exec "$PYTHON_BIN" "$SCRIPT_DIR/batch_processor.py" \
    --input-dir "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --prompt "$PROMPT" \
    --noise "$NOISE_LEVEL" \
    --delta 0.02 \
    --variants 5 \
    --steps "$STEPS" \
    --cfg "$CFG" \
    --model "$MODEL_NAME" \
    $TEST_FLAG
