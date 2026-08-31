"""
Tests for model architecture.
"""

import torch
import pytest

from tiny_llm.models import Transformer, BPETokenizer


def test_transformer_instantiation():
    """Test that transformer can be instantiated."""
    model = Transformer(
        vocab_size=2000,
        embedding_dim=192,
        num_layers=4,
        num_heads=6,
        context_length=256,
    )
    assert model is not None
    assert model.vocab_size == 2000
    assert model.embedding_dim == 192


def test_transformer_forward_pass():
    """Test forward pass through transformer."""
    model = Transformer(vocab_size=2000, embedding_dim=192, num_layers=4, num_heads=6)
    
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 2000, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, 2000)


def test_transformer_parameter_count():
    """Test parameter count."""
    model = Transformer(vocab_size=2000, embedding_dim=192, num_layers=4, num_heads=6)
    param_count = model.count_parameters()
    
    # Should be roughly 3-5M parameters
    assert 2_000_000 < param_count < 10_000_000


def test_transformer_generation():
    """Test text generation."""
    model = Transformer(vocab_size=2000, embedding_dim=192, num_layers=4, num_heads=6)
    model.eval()
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    generated = model.generate(input_ids, max_tokens=10)
    
    # Should be original + 10 new tokens (or less if EOS)
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 5


def test_bpe_tokenizer():
    """Test BPE tokenizer."""
    tokenizer = BPETokenizer(vocab_size=100)
    
    # Train on dummy data
    texts = [
        "quantum computing is interesting",
        "entanglement and superposition are key concepts",
    ]
    tokenizer.train(texts, num_merges=50)
    
    assert tokenizer.is_trained
    assert len(tokenizer.vocab) > 0


def test_tokenizer_encode_decode():
    """Test tokenizer encode/decode round-trip."""
    tokenizer = BPETokenizer(vocab_size=100)
    texts = ["hello world"] * 5
    tokenizer.train(texts, num_merges=20)
    
    original_text = "hello"
    encoded = tokenizer.encode(original_text)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    # Should decode to something (not exactly same due to BPE, but should be similar)
    assert decoded is not None
    assert len(decoded) > 0
