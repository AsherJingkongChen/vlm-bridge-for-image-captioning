"""
Language Model using Gemma-2-2B

This module implements the LanguageModel component that wraps the Gemma-2-2B model
for text generation. The model is frozen during training.

Key Features:
- Uses google/gemma-2-2b (base model, not instruction-tuned)
- Model dimension: 2304
- All weights are frozen to preserve pre-trained representations
- Handles text tokenization automatically
- Supports autoregressive generation
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, List, Union


class LanguageModel(nn.Module):
    """
    Gemma-2-2B based Language Model for text generation.

    This model wraps the Gemma-2-2B model and provides a clean interface for
    text generation while keeping all weights frozen.
    """

    def __init__(
        self, model_name: str = "google/gemma-2-2b", device: Optional[str] = None
    ):
        """
        Initialize the Language Model.

        Args:
            model_name: HuggingFace model identifier for Gemma-2-2B
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

        # Load Gemma-2-2B model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        # Move model to device if not using device_map
        if device != "cuda":
            self.model = self.model.to(self.device)

        # Freeze all parameters
        self._freeze_parameters()

        # Set to evaluation mode
        self.model.eval()

        # Cache model dimensions
        self.hidden_size = self.model.config.hidden_size  # 2304 for Gemma-2-2B
        self.vocab_size = self.model.config.vocab_size

    def _freeze_parameters(self):
        """Freeze all model parameters to prevent training."""
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the language model.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass without gradients
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

        return outputs.logits

    def forward_from_embeddings(
        self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass starting from embeddings (skipping the embedding layer).

        This method allows us to use vision-enhanced embeddings from the BridgeModule
        instead of standard text embeddings.

        Args:
            inputs_embeds: Pre-computed embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Move inputs to device
        inputs_embeds = inputs_embeds.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass - allow gradients to flow through frozen model
        # Note: Model weights are frozen via requires_grad=False, but we need
        # gradient flow for backpropagation to trainable BridgeModule
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return outputs.logits

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get word embeddings from input tokens.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]

        Returns:
            Embeddings of shape [batch_size, seq_len, hidden_size]
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)

        # Get embeddings from the model - allow gradients to flow through
        # Note: Embedding weights are frozen via requires_grad=False, but we need
        # gradient flow for backpropagation to trainable BridgeModule
        embeddings = self.model.get_input_embeddings()(input_ids)

        return embeddings

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using the language model.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs of shape [batch_size, generated_len]
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Set pad token ID
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        # Generate with specified parameters
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

        return outputs.sequences

    def encode_text(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text strings into token IDs.

        Args:
            texts: Text string or list of strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Dictionary containing 'input_ids' and 'attention_mask'
        """
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def decode_text(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode token IDs back to text strings.

        Args:
            token_ids: Token IDs of shape [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the language model."""
        return self.hidden_size

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model.config.name_or_path,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "frozen_parameters": sum(
                p.numel() for p in self.model.parameters() if not p.requires_grad
            ),
            "device": self.device,
        }
