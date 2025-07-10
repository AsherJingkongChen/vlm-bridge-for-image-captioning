"""
Vision-Language Bridge Model Architecture

This module implements the core model architecture for the Vision-Language Bridge,
following the RED-F framework design principles.

Core Components:
- VisionEncoder: Frozen DINOv2 for visual feature extraction
- LanguageModel: Frozen Gemma-2-2B for text generation
- BridgeModule: Trainable cross-attention bridge between vision and language
- FullModel: Complete assembled model for training and inference

Architecture Overview:
    Image → VisionEncoder → [N, 768] → BridgeModule → [N, 2304] → LanguageModel → Text

Only the BridgeModule is trainable, keeping the vision and language models frozen.
"""

from .vision_encoder import VisionEncoder
from .language_model import LanguageModel
from .bridge_module import BridgeModule
from .full_model import FullModel

__all__ = [
    "VisionEncoder",
    "LanguageModel", 
    "BridgeModule",
    "FullModel"
]