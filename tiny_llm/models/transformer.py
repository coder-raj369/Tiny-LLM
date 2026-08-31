"""
Transformer model for Tiny LLM.
4-layer, 6-head, 192-dim transformer for domain-specific Q&A.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for attention.
    More efficient than absolute positional embeddings.
    """
    
    def __init__(self, dim: int, max_seq_length: int = 256, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            seq_len: Optional sequence length override
        
        Returns:
            Rope embeddings of shape (1, seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # Repeat for both sin and cos
        return emb.unsqueeze(0)  # (1, seq_len, dim)


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Attention mask of shape (seq_len, seq_len) or (batch, seq_len, seq_len)
            cache: Optional KV cache for inference (key, value) tuples
        
        Returns:
            Output tensor of shape (batch, seq_len, dim) and updated cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # KV cache for inference (optional)
        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_len, kv_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embedding_dim)
        
        # Final projection
        output = self.proj(attn_output)
        output = self.proj_dropout(output)
        
        # Return updated cache for inference
        new_cache = (k, v) if cache is not None else None
        
        return output, new_cache


class FeedForward(nn.Module):
    """Feed-forward network (MLP)."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block (attention + FFN with residuals)."""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ffn = FeedForward(embedding_dim, 4 * embedding_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with pre-norm and residuals.
        
        Args:
            x: Input tensor
            mask: Attention mask
            cache: KV cache for inference
        
        Returns:
            Output tensor and updated cache
        """
        # Attention block
        x_norm = self.norm1(x)
        attn_output, new_cache = self.attn(x_norm, mask, cache)
        x = x + self.dropout(attn_output)
        
        # Feed-forward block
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        
        return x, new_cache


class Transformer(nn.Module):
    """
    Tiny transformer model for domain-specific Q&A.
    
    Architecture:
    - 4 transformer layers
    - 6 attention heads
    - 192 embedding dimensions
    - ~3-5M parameters
    """
    
    def __init__(
        self,
        vocab_size: int = 2000,
        embedding_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        context_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.dropout = dropout
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask (prevents attending to future tokens).
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            Attention mask of shape (seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training and evaluation.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask of shape (batch, seq_len)
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token + positional embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        pos_emb = self.pos_embedding(pos_ids)
        x = token_emb + pos_emb
        x = self.embed_dropout(x)
        
        # Causal attention mask
        causal_mask = self.create_attention_mask(seq_len, input_ids.device)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x, _ = block(x, mask=causal_mask, cache=None)
        
        # Final processing
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: int = 3,
    ) -> torch.Tensor:
        """
        Generate text tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs of shape (batch, seq_len) or (1, seq_len)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (>1 = more diverse, <1 = more conservative)
            top_k: Keep only top-k most likely tokens
            top_p: Keep only tokens with cumulative probability up to top_p
            eos_token_id: End-of-sequence token ID (stops generation)
        
        Returns:
            Generated token IDs of shape (batch, seq_len + max_tokens)
        """
        device = input_ids.device
        
        for _ in range(max_tokens):
            # Get predictions
            with torch.no_grad():
                logits = self.forward(input_ids[:, -self.context_length:])
            
            # Take last token's logits
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[:, indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if EOS token generated
            if next_token.item() == eos_token_id:
                break
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model configuration and parameter count."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "context_length": self.context_length,
            "dropout": self.dropout,
            "total_parameters": self.count_parameters(),
            "total_parameters_millions": self.count_parameters() / 1e6,
        }
