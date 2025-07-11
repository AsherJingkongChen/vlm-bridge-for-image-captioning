"""
Full Vision-Language Model

This module implements the complete Vision-Language Bridge model that assembles
all components together. It provides unified interfaces for training and inference.

Key Features:
- Combines VisionEncoder, LanguageModel, and BridgeModule
- Supports both training and inference modes
- Handles end-to-end forward pass from images to text
- Provides utilities for model saving/loading
- Supports autoregressive text generation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union
from PIL import Image

from .vision_encoder import VisionEncoder
from .language_model import LanguageModel
from .bridge_module import BridgeModule


class FullModel(nn.Module):
    """
    Complete Vision-Language Bridge model.

    This model combines all components and provides a unified interface for
    training and inference. Only the BridgeModule is trainable.
    """

    def __init__(
        self,
        vision_model_name: str = "facebook/dinov2-large",
        language_model_name: str = "google/gemma-2-2b",
        bridge_num_heads: int = 8,
        bridge_dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        Initialize the full vision-language model.

        Args:
            vision_model_name: HuggingFace model identifier for vision encoder
            language_model_name: HuggingFace model identifier for language model
            bridge_num_heads: Number of attention heads in bridge module
            bridge_dropout: Dropout probability in bridge module
            device: Device to place models on (auto-detect if None)
        """
        super().__init__()

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Initialize components
        self.vision_encoder = VisionEncoder(vision_model_name, device)
        self.language_model = LanguageModel(language_model_name, device)

        # Initialize bridge module with correct dimensions and data type
        self.bridge_module = BridgeModule(
            vision_dim=self.vision_encoder.output_dim,
            language_dim=self.language_model.output_dim,
            num_heads=bridge_num_heads,
            dropout=bridge_dropout,
        ).to(device)
        
        # Set bridge module to same data type as language model
        if device == "cuda":
            self.bridge_module = self.bridge_module.half()  # Convert to float16

        # Cache model info
        self.vision_dim = self.vision_encoder.output_dim
        self.language_dim = self.language_model.output_dim

        # Ensure only bridge module is trainable
        self._freeze_pretrained_components()

    def _freeze_pretrained_components(self):
        """Ensure only the bridge module is trainable."""
        # Vision encoder and language model should already be frozen
        # But let's double-check
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

        # Bridge module should be trainable
        for param in self.bridge_module.parameters():
            param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete vision-language model.

        Data flow:
        1. Images → VisionEncoder (frozen) → vision_features
        2. input_ids → LanguageModel.embeddings (frozen) → text_embeddings
        3. (vision_features, text_embeddings) → BridgeModule (trainable) → enhanced_embeddings
        4. enhanced_embeddings → LanguageModel.forward_from_embeddings (frozen) → logits

        Args:
            images: Input images [batch_size, 3, 224, 224]
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]

        Returns:
            Dictionary containing:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - loss: Cross-entropy loss (if labels provided)
            - vision_features: Visual features [batch_size, vision_seq_len, vision_dim]
            - text_embeddings: Text embeddings [batch_size, seq_len, language_dim]
            - enhanced_embeddings: Vision-enhanced embeddings [batch_size, seq_len, language_dim]
        """
        # Step 1: Extract visual features (frozen)
        vision_features = self.vision_encoder(images)
        # Shape: [batch_size, vision_seq_len, vision_dim]

        # Step 2: Get standard text embeddings (frozen)
        text_embeddings = self.language_model.get_embeddings(input_ids)
        # Shape: [batch_size, seq_len, language_dim]

        # Step 3: Create vision-enhanced embeddings (trainable)
        enhanced_embeddings = self.bridge_module(
            vision_features=vision_features, text_embeddings=text_embeddings
        )
        # Shape: [batch_size, seq_len, language_dim]

        # Step 4: Generate logits through language model (frozen)
        logits = self.language_model.forward_from_embeddings(
            inputs_embeds=enhanced_embeddings, attention_mask=attention_mask
        )
        # Shape: [batch_size, seq_len, vocab_size]

        # Prepare output
        output = {
            "logits": logits,
            "vision_features": vision_features,
            "text_embeddings": text_embeddings,
            "enhanced_embeddings": enhanced_embeddings,
        }

        # Compute loss if labels are provided
        if labels is not None:
            loss = self._compute_loss(logits, labels, attention_mask)
            output["loss"] = loss

        return output

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Cross-entropy loss
        """
        # Shift logits and labels for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(shift_logits, shift_labels)

        return loss

    def generate_caption(
        self,
        image: Union[torch.Tensor, Image.Image],
        max_length: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a caption for a single image using auto-regressive generation.

        This implements the exact inference process described in the SOP:
        1. Extract visual features once
        2. Auto-regressively generate tokens, where each step:
           - Gets current text embeddings
           - Enhances them with vision through BridgeModule
           - Generates next token probability distribution

        Args:
            image: Input image (tensor or PIL Image)
            max_length: Maximum caption length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter

        Returns:
            Generated caption string
        """
        self.eval()

        with torch.no_grad():
            # Preprocess image if needed
            if isinstance(image, Image.Image):
                images = self.vision_encoder.preprocess_images([image])
            else:
                images = image.unsqueeze(0) if image.dim() == 3 else image

            images = images.to(self.device)

            # Extract visual features once (they don't change during generation)
            vision_features = self.vision_encoder(images)

            # Start with BOS token
            input_ids = torch.tensor([[self.language_model.tokenizer.bos_token_id]]).to(
                self.device
            )

            # Auto-regressive generation loop
            for _ in range(max_length):
                # Get current text embeddings
                text_embeddings = self.language_model.get_embeddings(input_ids)

                # Enhance with vision through BridgeModule
                enhanced_embeddings = self.bridge_module(
                    vision_features, text_embeddings
                )

                # Generate logits through language model
                logits = self.language_model.forward_from_embeddings(
                    enhanced_embeddings
                )

                # Get next token logits (last position)
                next_token_logits = logits[:, -1, :] / temperature

                # Sample next token
                if do_sample:
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")

                    # Sample from the filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append token to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Check for EOS token
                if next_token.item() == self.language_model.tokenizer.eos_token_id:
                    break

            # Decode the generated sequence
            caption = self.language_model.decode_text(input_ids)[0]

            # Clean up the caption
            caption = caption.replace(
                self.language_model.tokenizer.bos_token, ""
            ).strip()
            caption = caption.replace(
                self.language_model.tokenizer.eos_token, ""
            ).strip()

            return caption

    def save_model(self, path: str):
        """
        Save the model (only the trainable bridge module).

        Args:
            path: Path to save the model
        """
        # Only save the bridge module since other components are frozen
        torch.save(
            {
                "bridge_module_state_dict": self.bridge_module.state_dict(),
                "model_config": {
                    "vision_dim": self.vision_dim,
                    "language_dim": self.language_dim,
                    "vision_model_name": self.vision_encoder.model.config.name_or_path,
                    "language_model_name": self.language_model.model.config.name_or_path,
                },
            },
            path,
        )

    def load_model(self, path: str):
        """
        Load the model (only the trainable bridge module).

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.bridge_module.load_state_dict(checkpoint["bridge_module_state_dict"])

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        vision_info = self.vision_encoder.get_model_info()
        language_info = self.language_model.get_model_info()
        bridge_info = self.bridge_module.get_model_info()

        total_params = (
            vision_info["num_parameters"]
            + language_info["num_parameters"]
            + bridge_info["total_parameters"]
        )

        trainable_params = bridge_info["trainable_parameters"]

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "trainable_ratio": f"{trainable_params / total_params:.6f}",
            "components": {
                "vision_encoder": vision_info,
                "language_model": language_info,
                "bridge_module": bridge_info,
            },
            "device": self.device,
        }
