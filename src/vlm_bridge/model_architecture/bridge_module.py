"""
Bridge Module for Vision-Language Connection

This module implements the BridgeModule as a modified input layer for the language model.
It acts as a "vision-aware text embedding layer" that enhances text embeddings with visual context.

Key Features:
- Transformer Decoder Block with cross-attention
- Replaces standard text embedding layer in auto-regressive generation
- Maintains causal properties for language modeling
- Only trainable component in the full model

Architecture:
Text Embeddings + Vision Features → Cross-Attention → Enhanced Text Embeddings → Gemma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for vision-language fusion.
    
    This module implements cross-attention where:
    - Query comes from text embeddings
    - Key and Value come from visual features
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multi-head cross-attention.
        
        Args:
            d_model: Model dimension (should be 2304 for Gemma-2-2B)
            num_heads: Number of attention heads
            dropout: Dropout probability
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
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Attended features [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Linear projections
        Q = self.w_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.w_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.w_v(value)  # [batch_size, seq_len_v, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output
        
    def _scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Scaled dot-product attention mechanism.
        """
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output


class BridgeModule(nn.Module):
    """
    Vision-aware text embedding layer for auto-regressive language generation.
    
    This module acts as a replacement for the standard text embedding layer,
    creating vision-enhanced text embeddings that maintain auto-regressive properties.
    """
    
    def __init__(self, 
                 vision_dim: int = 1024,
                 language_dim: int = 2304,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize the bridge module.
        
        Args:
            vision_dim: Vision encoder output dimension (1024 for DINOv2-Large)
            language_dim: Language model dimension (2304 for Gemma-2-2B)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        
        # Linear projection to map vision features to language dimension
        self.vision_projection = nn.Linear(vision_dim, language_dim)
        
        # Layer normalization (pre-norm style like modern transformers)
        self.ln1 = nn.LayerNorm(language_dim)
        self.ln2 = nn.LayerNorm(language_dim)
        
        # Cross-attention mechanism (text queries, vision keys/values)
        self.cross_attention = MultiHeadCrossAttention(
            d_model=language_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network (standard transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(language_dim, language_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim * 4, language_dim),
            nn.Dropout(dropout)
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
                
    def forward(self, 
                vision_features: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass creating vision-enhanced text embeddings.
        
        Args:
            vision_features: Visual features [batch_size, vision_seq_len, vision_dim]
            text_embeddings: Text embeddings [batch_size, text_seq_len, language_dim]
            
        Returns:
            Enhanced text embeddings [batch_size, text_seq_len, language_dim]
        """
        # Project vision features to language dimension
        projected_vision = self.vision_projection(vision_features)
        # Shape: [batch_size, vision_seq_len, language_dim]
        
        # Pre-layer norm for text embeddings
        normed_text = self.ln1(text_embeddings)
        
        # Cross-attention: text queries attend to vision keys/values
        # Note: No causal mask needed here - vision is global context
        attended_features = self.cross_attention(
            query=normed_text,           # [batch_size, text_seq_len, language_dim]
            key=projected_vision,        # [batch_size, vision_seq_len, language_dim]
            value=projected_vision,      # [batch_size, vision_seq_len, language_dim]
            mask=None                    # Vision is accessible to all text positions
        )
        
        # Residual connection after cross-attention
        x = text_embeddings + attended_features
        
        # Pre-layer norm for FFN
        normed_x = self.ln2(x)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(normed_x)
        enhanced_embeddings = x + ffn_output
        
        return enhanced_embeddings
        
    def get_attention_weights(self, 
                            vision_features: torch.Tensor,
                            text_embeddings: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            vision_features: Visual features [batch_size, vision_seq_len, vision_dim]
            text_embeddings: Text embeddings [batch_size, text_seq_len, language_dim]
            attention_mask: Attention mask [batch_size, text_seq_len, vision_seq_len]
            
        Returns:
            Attention weights [batch_size, num_heads, text_seq_len, vision_seq_len]
        """
        # Project vision features
        projected_vision = self.vision_projection(vision_features)
        normed_text = self.ln1(text_embeddings)
        
        # Get attention weights (modify cross_attention to return weights)
        # This is a simplified version - in practice, you'd modify the attention module
        batch_size, text_seq_len, _ = text_embeddings.shape
        vision_seq_len = vision_features.shape[1]
        
        # For now, return dummy weights (implement actual attention weight extraction)
        return torch.ones(batch_size, 8, text_seq_len, vision_seq_len)
        
    def get_model_info(self) -> dict:
        """Get information about the bridge module."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "vision_dim": self.vision_dim,
            "language_dim": self.language_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_ratio": f"{trainable_params / total_params:.4f}"
        }