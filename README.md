# Vision-Language Bridge for Image Captioning

A lightweight, trainable bridge module that connects frozen vision and language models for image captioning tasks.

## 🌟 Overview

This project implements an Encoder-Adapter-Decoder architecture that enables effective communication between vision and language models:

-   **Vision Encoder**: DINOv2-large (304.4M params, frozen)
-   **Language Model**: Gemma-2-2B (2.61B params, frozen)
-   **Bridge Module**: Cross-attention adapter (66.1M params, trainable)

Only 2.21% of parameters are trainable, making this approach highly efficient while maintaining strong performance.

## 🚀 Quick Start

### Prerequisites

-   Python 3.12+
-   CUDA-capable GPU (16GB+ VRAM recommended)
-   HuggingFace account with access to Gemma-2-2B

### Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/AsherJingkongChen/vlm-bridge-for-image-captioning.git
cd vlm-bridge-for-image-captioning

# Create virtual environment and install dependencies
uv sync
source .venv/bin/activate

# Set HuggingFace token (required for Gemma-2-2B)
export HF_TOKEN="your_huggingface_token"
```

### Data Preparation

Process the GroundCap dataset (52,350 image-caption pairs):

```bash
uv run vlm-data explore-dataset
# Process the dataset into train/val/test splits
uv run vlm-data transform --output-dir data/groundcap
uv run vlm-data inspect-loader --data-dir data/groundcap
```

### Training

Start training with configuration files:

```bash
# Basic training with default configuration
uv run vlm-training

# Use custom configuration file
uv run vlm-training --config config/training-default.yaml

# Resume from checkpoint
uv run vlm-training --resume checkpoints/latest_checkpoint.pth

# Resume with custom config
uv run vlm-training --config config/training-default.yaml --resume checkpoints/latest_checkpoint.pth
```

### Configuration Management

Training parameters are managed through YAML configuration files:

```bash
# View default configuration
cat config/training-default.yaml

# Create custom configuration by copying and editing
cp config/training-default.yaml config/training-default.yaml
# Edit config/training-default.yaml with your preferred settings

# Key configuration parameters (see config/training-default.yaml):
# - learning_rate: 1e-5 (conservative for 48k dataset)
# - num_epochs: 12 (adjusted for dataset size)
# - batch_size: 8 (fits most GPUs)
# - gradient_clip_val: 0.3 (strict for stability)
# - use_early_stopping: true (patience: 3 epochs)
# - use_scheduler: true (cosine annealing)
# - amp_dtype: bfloat16 (best for A100)
```

## 🖥️ Training on vast.ai

### One-Command Setup

```bash
# On vast.ai instance (after SSH)
curl -LSsf https://raw.githubusercontent.com/AsherJingkongChen/vlm-bridge-for-image-captioning/main/scripts/setup_vastai_remote.sh | bash
```

This script will:

1. Install all dependencies
2. Clone the repository
3. Process the dataset
4. Start training automatically

### Monitor from Local Machine

```bash
# On your local machine
wget https://raw.githubusercontent.com/AsherJingkongChen/vlm-bridge-for-image-captioning/main/scripts/control_vastai_local.sh

# Monitor training progress with SSH key and port
bash control_vastai_local.sh root@ssh2.vast.ai monitor -p 12345 -i ~/.ssh/id_ed25519
```

### Download Checkpoints to Local Machine

```bash
# Download checkpoints with SSH key and port
bash control_vastai_local.sh root@ssh2.vast.ai download -p 12345 -i ~/.ssh/id_ed25519
```

Then open http://localhost:6006 for TensorBoard.

## 📊 Model Architecture

### Vision Encoder (DINOv2-large)

-   Input: RGB images [batch, 3, 224, 224]
-   Output: Visual features [batch, 257, 1024]
-   304.4M parameters (frozen)

### Bridge Module (Cross-Attention)

-   Architecture: Transformer decoder block
-   Function: Enhances text embeddings with visual context
-   66.1M parameters (trainable)
-   Key components:
    -   Vision projection: 1024 → 2304 dims
    -   Multi-head cross-attention (8 heads)
    -   Feed-forward network
    -   Layer normalization

### Language Model (Gemma-2-2B)

-   Input: Enhanced text embeddings
-   Output: Next token predictions
-   2.61B parameters (frozen)

### Data Flow

(Note: This is a simplified diagram)

```
Image → DINOv2 → Visual Features   ↘
                                    Cross-Attention → Enhanced Embeddings → Gemma → Caption
Text → Tokenizer → Text Embeddings ↗
```

## 💻 Hardware Requirements

### Training (A100 40GB recommended)

-   VRAM: ~21-31 GB (depending on batch size)
-   Batch size 8: ~21 GB
-   Batch size 16: ~31 GB
-   Training time: ~2-3 hours per epoch on A100

### Inference (RTX 4080 16GB minimum)

-   VRAM: ~8-10 GB
-   Inference speed: ~5-10 captions/second

## 📈 Training Tips

### Hyperparameters

-   Learning rate: 1e-5 to 2e-5 (default: 1e-5) - Conservative for 48k dataset
-   Batch size: 8-16 (depending on GPU memory)
-   Gradient clipping: 0.3 (strict for stability)
-   Mixed precision: BF16 (A100) or FP16 (other GPUs)
-   LR Scheduler: Cosine annealing with min_lr=1e-6

### Monitoring (Multi-Stage Expectations)

**🔴 Initial Stage (Epoch 1-3):**

-   Validation Loss: 6.0 → 3.5
-   Perplexity: 400+ → 30-50

**🟡 Mid Stage (Epoch 4-8):**

-   Validation Loss: 3.5 → 2.2
-   Perplexity: 30-50 → 9-15

**🟢 Final Stage (Epoch 9-12):**

-   Validation Loss: 2.2 → 1.8
-   Perplexity: 9-15 → 6-12

Best model saved automatically based on validation loss. Early stopping after 3 epochs without improvement.

### Common Issues

**Out of Memory**

-   Reduce batch size
-   Enable gradient checkpointing
-   Use FP16 instead of BF16

**Slow Training**

-   Increase number of data loader workers
-   Ensure GPU utilization > 90%
-   Check for data loading bottlenecks

**Poor Performance**

-   Train for more epochs (15-20)
-   Try different learning rates
-   Ensure data quality

## 📝 Project Structure

```
vlm-bridge-for-image-captioning/
├── src/vlm_bridge/
│   ├── data_pipeline/          # Data processing modules
│   ├── model_architecture/     # Model components
│   └── training_strategy/      # Training logic
├── config/                     # Training configurations
│   └── training-default.yaml  # Default training parameters
├── scripts/
│   ├── setup_vastai_remote.sh  # vast.ai setup script
│   └── control_vastai_local.sh # Local controlling script
├── data/groundcap/             # Processed dataset
│   ├── train/                  # 41,880 samples
│   ├── val/                    # 2,618 samples (5% for fast validation)
│   └── test/                   # 10,470 samples
├── checkpoints/                # Saved model weights
└── logs/                       # TensorBoard logs
```

## 🎯 Expected Results

After 12 epochs of training on 48k dataset:

**Success Thresholds:**

-   **Basic Performance**: Perplexity < 15 (considering dataset size)
-   **Good Performance**: Perplexity < 10
-   **Excellent Performance**: Perplexity < 8

**Outcomes:**

-   Meaningful captions for unseen images
-   Good generalization within domain
-   Training time: ~30 min/epoch on A100

## 🤝 Acknowledgments

-   DINOv2: Self-supervised vision transformer by Meta
-   Gemma-2-2B: Efficient language model by Google
-   GroundCap: High-quality image-caption dataset

## 📄 License

This project is licensed under the MIT License. Model weights are subject to their respective licenses (DINOv2: Apache 2.0, Gemma: Gemma Terms of Use).
