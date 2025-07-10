"""
Vision Encoder using DINOv2

This module implements the VisionEncoder component that wraps the DINOv2 model
for visual feature extraction. The encoder is frozen during training.

Key Features:
- Uses facebook/dinov2-vitb14 (86M parameters)
- Outputs visual features of shape [batch_size, seq_len, 768]
- All weights are frozen to preserve pre-trained representations
- Handles image preprocessing automatically
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from typing import Dict, Any, Optional
from PIL import Image


class VisionEncoder(nn.Module):
    """
    DINOv2-based Vision Encoder for extracting visual features from images.
    
    This encoder wraps the DINOv2 model and provides a clean interface for
    feature extraction while keeping all weights frozen.
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-base", device: Optional[str] = None):
        """
        Initialize the Vision Encoder.
        
        Args:
            model_name: HuggingFace model identifier for DINOv2
            device: Device to place the model on (auto-detect if None)
        """
        super().__init__()
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Load DINOv2 model and processor (use fast processor to avoid warnings)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Freeze all parameters
        self._freeze_parameters()
        
        # Set to evaluation mode
        self.model.eval()
        
        # Cache model dimensions
        self.hidden_size = self.model.config.hidden_size  # 768 for ViT-B/14
        
    def _freeze_parameters(self):
        """Freeze all model parameters to prevent training."""
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            images: Preprocessed images of shape [batch_size, 3, 224, 224]
            
        Returns:
            Visual features of shape [batch_size, seq_len, 768]
            where seq_len = (224/14)^2 + 1 = 257 (256 patches + 1 CLS token)
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Move input to device
        images = images.to(self.device)
        
        # Extract features without gradients
        with torch.no_grad():
            outputs = self.model(pixel_values=images)
            
        # Return the last hidden states
        # Shape: [batch_size, seq_len, hidden_size]
        return outputs.last_hidden_state
        
    def get_cls_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract only the CLS token features (global image representation).
        
        Args:
            images: Preprocessed images of shape [batch_size, 3, 224, 224]
            
        Returns:
            CLS features of shape [batch_size, 768]
        """
        features = self.forward(images)
        # CLS token is the first token
        return features[:, 0, :]
        
    def get_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract only the patch features (spatial representations).
        
        Args:
            images: Preprocessed images of shape [batch_size, 3, 224, 224]
            
        Returns:
            Patch features of shape [batch_size, 256, 768]
        """
        features = self.forward(images)
        # Patch tokens are from index 1 onwards
        return features[:, 1:, :]
        
    def preprocess_images(self, images) -> torch.Tensor:
        """
        Preprocess raw images for the model.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            Preprocessed tensor of shape [batch_size, 3, 224, 224]
        """
        # Use the processor to handle preprocessing
        inputs = self.processor(images, return_tensors="pt")
        return inputs['pixel_values']
        
    @property
    def output_dim(self) -> int:
        """Get the output dimension of the vision encoder."""
        return self.hidden_size
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model.config.name_or_path,
            "hidden_size": self.hidden_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "frozen_parameters": sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
            "device": self.device
        }