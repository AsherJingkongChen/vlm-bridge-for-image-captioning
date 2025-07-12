"""
Bridge Module for Vision-Language Connection v2.0: Bridge-Lite

This module implements the Bridge-Lite architecture inspired by BLIP-2 Q-Former.
It uses stacked BridgeBlocks to create a more powerful vision-language fusion system.

Key Features:
- Stack of 2 BridgeBlocks (configurable)
- Each BridgeBlock: Cross-Attention + Self-Attention + FFN
- Optimized for smaller datasets (50K samples)
- Reduces overfitting while improving interaction capability
- Only trainable component in the full model (~130M parameters)

Architecture:
Text Embeddings + Vision Features → BridgeBlock_1 → BridgeBlock_2 → Enhanced Text Embeddings → Gemma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for vision-language fusion.

    This module implements cross-attention where:
    - Query comes from text embeddings (language_dim)
    - Key and Value come from visual features (vision_dim)

    The cross-attention layer handles dimension alignment dynamically,
    projecting both query and key/value to a common dimension.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        """
        Initialize multi-head cross-attention.

        Args:
            query_dim: Query dimension (2304 for text embeddings)
            kv_dim: Key/Value dimension (1024 for vision features)
            d_model: Internal model dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability (0.2 as per SOP)
        """
        super().__init__()

        assert d_model % num_heads == 0

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V with different input dimensions
        self.w_q = nn.Linear(query_dim, d_model)
        self.w_k = nn.Linear(kv_dim, d_model)
        self.w_v = nn.Linear(kv_dim, d_model)

        # Output projection back to query dimension
        self.w_o = nn.Linear(d_model, query_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention.

        Args:
            query: Query tensor [batch_size, seq_len_q, query_dim]
            key: Key tensor [batch_size, seq_len_k, kv_dim]
            value: Value tensor [batch_size, seq_len_v, kv_dim]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k]

        Returns:
            Attended features [batch_size, seq_len_q, query_dim]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # Linear projections - handling dimension mismatch here
        Q = self.w_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.w_k(key)  # [batch_size, seq_len_k, d_model]
        V = self.w_v(value)  # [batch_size, seq_len_v, d_model]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )

        # Final linear projection back to query dimension
        output = self.w_o(attention_output)

        return output

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention using PyTorch's efficient implementation.
        """
        return F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for within-text interaction.

    This module implements self-attention where Query, Key, and Value all
    come from the same input sequence (text embeddings).
    """

    def __init__(self, d_model: int, num_heads: int = 18, dropout: float = 0.2):
        """
        Initialize multi-head self-attention.

        Args:
            d_model: Model dimension (should be 2304 for Gemma-2-2B)
            num_heads: Number of attention heads (18 as specified in SOP)
            dropout: Dropout probability (0.2 as specified in SOP)
        """
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]

        Returns:
            Self-attended features [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.w_q(x)  # [batch_size, seq_len, d_model]
        K = self.w_k(x)  # [batch_size, seq_len, d_model]
        V = self.w_v(x)  # [batch_size, seq_len, d_model]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.w_o(attention_output)

        return output

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention using PyTorch's efficient implementation.
        """
        return F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )


class BridgeBlock(nn.Module):
    """
    A single Bridge Block containing:
    1. Cross-Attention (text queries, vision keys/values)
    2. Self-Attention (within text)
    3. Feed-Forward Network

    This is the core building block of the Bridge-Lite architecture.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        language_dim: int = 2304,
        num_heads_cross: int = 8,
        num_heads_self: int = 18,
        dropout: float = 0.2,
    ):
        """
        Initialize a single Bridge Block.

        Args:
            vision_dim: Vision encoder output dimension (1024 for DINOv2-Large)
            language_dim: Language model dimension (2304 for Gemma-2-2B)
            num_heads_cross: Number of heads for cross-attention
            num_heads_self: Number of heads for self-attention (18 as per SOP)
            dropout: Dropout probability (0.2 as per SOP)
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim

        # Layer 1: Cross-Attention (text queries attend to vision keys/values)
        # Cross-attention handles dimension mismatch between vision (1024) and language (2304)
        self.cross_attention = MultiHeadCrossAttention(
            query_dim=language_dim,
            kv_dim=vision_dim,
            d_model=language_dim,  # Internal dimension for attention computation
            num_heads=num_heads_cross,
            dropout=dropout,
        )
        self.ln_cross = nn.LayerNorm(language_dim)

        # Layer 2: Self-Attention (within text)
        self.self_attention = MultiHeadSelfAttention(
            d_model=language_dim, num_heads=num_heads_self, dropout=dropout
        )
        self.ln_self = nn.LayerNorm(language_dim)

        # Layer 3: Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(language_dim, language_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim * 4, language_dim),
            nn.Dropout(dropout),
        )
        self.ln_ffn = nn.LayerNorm(language_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through a single Bridge Block.

        Args:
            text_embeddings: Text embeddings [batch_size, text_seq_len, language_dim]
            vision_features: Vision features [batch_size, vision_seq_len, vision_dim] (raw, unprojected)

        Returns:
            Enhanced text embeddings [batch_size, text_seq_len, language_dim]
        """
        # Layer 1: Cross-Attention with residual connection
        normed_text = self.ln_cross(text_embeddings)
        cross_attended = self.cross_attention(
            query=normed_text,
            key=vision_features,
            value=vision_features,
            mask=None,  # Vision is global context
        )
        text_embeddings = text_embeddings + cross_attended

        # Layer 2: Self-Attention with residual connection
        normed_text = self.ln_self(text_embeddings)
        self_attended = self.self_attention(normed_text)
        text_embeddings = text_embeddings + self_attended

        # Layer 3: Feed-Forward Network with residual connection
        normed_text = self.ln_ffn(text_embeddings)
        ffn_output = self.ffn(normed_text)
        text_embeddings = text_embeddings + ffn_output

        return text_embeddings


class BridgeLite(nn.Module):
    """
    Bridge-Lite: Stack of Bridge Blocks for enhanced vision-language fusion.

    This module implements the Bridge-Lite architecture with configurable number
    of stacked BridgeBlocks. Designed for better performance on smaller datasets.

    Key architectural decision (per SOP): No explicit vision projection layer.
    The Cross-Attention mechanism in each BridgeBlock handles the dimension
    alignment between vision (1024d) and language (2304d) features dynamically.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        language_dim: int = 2304,
        num_blocks: int = 2,
        num_heads_cross: int = 8,
        num_heads_self: int = 18,
        dropout: float = 0.2,
    ):
        """
        Initialize the Bridge-Lite module.

        Args:
            vision_dim: Vision encoder output dimension (1024 for DINOv2-Large)
            language_dim: Language model dimension (2304 for Gemma-2-2B)
            num_blocks: Number of BridgeBlocks to stack (default: 2)
            num_heads_cross: Number of heads for cross-attention
            num_heads_self: Number of heads for self-attention (18 as per SOP)
            dropout: Dropout probability (0.2 as per SOP)
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.num_blocks = num_blocks

        # Stack of Bridge Blocks
        # Note: Vision projection is removed per SOP - Cross-Attention handles dimension alignment
        self.bridge_blocks = nn.ModuleList(
            [
                BridgeBlock(
                    vision_dim=vision_dim,
                    language_dim=language_dim,
                    num_heads_cross=num_heads_cross,
                    num_heads_self=num_heads_self,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize module weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better convergence
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, vision_features: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the Bridge-Lite module.

        Args:
            vision_features: Visual features [batch_size, vision_seq_len, vision_dim]
            text_embeddings: Text embeddings [batch_size, text_seq_len, language_dim]

        Returns:
            Enhanced text embeddings [batch_size, text_seq_len, language_dim]
        """
        # Process through stack of Bridge Blocks
        # Each block's Cross-Attention handles the dimension mismatch internally
        enhanced_embeddings = text_embeddings
        for bridge_block in self.bridge_blocks:
            enhanced_embeddings = bridge_block(enhanced_embeddings, vision_features)

        return enhanced_embeddings

    def get_model_info(self) -> dict:
        """Get information about the Bridge-Lite module."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "architecture": "Bridge-Lite",
            "num_blocks": self.num_blocks,
            "vision_dim": self.vision_dim,
            "language_dim": self.language_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_ratio": f"{trainable_params / total_params:.4f}",
        }
