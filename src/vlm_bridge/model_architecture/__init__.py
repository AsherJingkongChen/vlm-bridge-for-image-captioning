"""
Vision-Language Bridge Model Architecture v2.0: Bridge-Lite

This module implements the core model architecture for the Vision-Language Bridge,
following the RED-F framework design principles.

Core Components:
- VisionEncoder: Frozen DINOv2 for visual feature extraction
- LanguageModel: Frozen Gemma-2-2B for text generation
- BridgeLite: Trainable stacked cross-attention bridge between vision and language
- BridgeBlock: Individual attention block (Cross-Attention + Self-Attention + FFN)
- FullModel: Complete assembled model for training and inference

Architecture Overview:
    Image → VisionEncoder → [N, 1024] → BridgeLite (2x BridgeBlock) → [N, 2304] → LanguageModel → Text

Only the BridgeLite is trainable (~130M parameters), keeping the vision and language models frozen.
"""

from .vision_encoder import VisionEncoder
from .language_model import LanguageModel
from .bridge_module import (
    BridgeLite,
    BridgeBlock,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from .full_model import FullModel

__all__ = [
    "VisionEncoder",
    "LanguageModel",
    "BridgeLite",
    "BridgeBlock",
    "MultiHeadCrossAttention",
    "MultiHeadSelfAttention",
    "FullModel",
]
