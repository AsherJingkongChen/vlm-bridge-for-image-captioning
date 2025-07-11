#!/bin/bash
# Setup script for vast.ai GPU instance
# This script prepares the environment and starts training

set -e  # Exit on error

echo "=================================================="
echo "VLM Bridge Training Setup for vast.ai"
echo "=================================================="

# Check if we're on a GPU instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y git curl build-essential python3-dev

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Clone repository
REPO_DIR="vlm-bridge-for-image-captioning"
if [ ! -d "$REPO_DIR" ]; then
    echo "📂 Cloning repository..."
    git clone https://github.com/AsherJingkongChen/vlm-bridge-for-image-captioning.git $REPO_DIR
fi

cd $REPO_DIR

# Set up Python environment
echo "🐍 Setting up Python environment..."
uv sync
source .venv/bin/activate

# Set up HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "🔑 HuggingFace token required for Gemma-2-2B access"
    echo "Please enter your HuggingFace token (HF_TOKEN):"
    read -s HF_TOKEN
    export HF_TOKEN
fi

# Process dataset if not already done
if [ ! -d "data/groundcap/train" ]; then
    echo ""
    echo "📊 Processing GroundCap dataset..."
    uv run vlm-data-pipeline transform
else
    echo "✅ Dataset already processed"
fi

# Training configuration
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/experiment}
LOG_DIR=${LOG_DIR:-logs/experiment}

echo ""
echo "=================================================="
echo "Setup Complete! Ready to train."
echo "=================================================="
echo ""
echo "Training configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Log dir: $LOG_DIR"
echo ""

# Start TensorBoard in background
echo "🔧 Starting TensorBoard on port 6006..."
uv run tensorboard --logdir $LOG_DIR --host 0.0.0.0 --port 6006 &
TENSORBOARD_PID=$!
echo "TensorBoard PID: $TENSORBOARD_PID"

# Start training
uv run vlm-training train \
    --data-dir data/groundcap \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --checkpoint-dir $CHECKPOINT_DIR \
    --log-dir $LOG_DIR

echo ""
echo "=================================================="
echo "✅ Training complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Logs: $LOG_DIR"
echo ""
echo "Best model: $CHECKPOINT_DIR/best_model.pth"
echo ""

# Kill TensorBoard
kill $TENSORBOARD_PID 2>/dev/null || true
