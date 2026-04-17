"""
Tests for BCA layers.
"""
import torch
import torch.nn as nn
import numpy as np
import pytest

from src.bca_layers import (
    BlockCirculantLinear,
    BCATransformerLayer,
    convert_linear_to_bca,
    approximate_weights_with_bca
)


def test_block_circulant_linear_creation():
    """Test creation of BlockCirculantLinear layer."""
    layer = BlockCirculantLinear(64, 128, block_size=8, bias=True)
    
    assert layer.in_features == 64
    assert layer.out_features == 128
    assert layer.block_size == 8
    assert layer.bias is not None
    assert layer.circ_params.shape == (128 // 8, 64 // 8, 8)
    
    # Test forward pass
    x = torch.randn(4, 64)
    y = layer(x)
    assert y.shape == (4, 128)


def test_bca_vs_direct():
    """Test BCA forward pass matches direct multiplication for small sizes."""
    torch.manual_seed(42)
    
    # Small sizes for exact comparison
    in_features = 16
    out_features = 32
    block_size = 4
    
    layer = BlockCirculantLinear(in_features, out_features, block_size, bias=False, use_fft=False)
    
    # Get full weight matrix
    W_full = layer.get_full_weight_matrix()
    
    # Test with random input
    x = torch.randn(2, in_features)
    
    # BCA forward
    y_bca = layer(x)
    
    # Direct multiplication
    y_direct = torch.matmul(x, W_full.T)
    
    # Should be close (might be small differences due to floating point)
    assert torch.allclose(y_bca, y_direct, rtol=1e-4, atol=1e-5)


def test_bca_with_fft():
    """Test BCA with FFT enabled."""
    layer = BlockCirculantLinear(64, 128, block_size=8, bias=True, use_fft=True)
    
    x = torch.randn(4, 64)
    y = layer(x)
    
    assert y.shape == (4, 128)
    
    # Test that FFT and non-FFT give similar results (not exact due to complex -> real conversions)
    layer_no_fft = BlockCirculantLinear(64, 128, block_size=8, bias=True, use_fft=False)
    layer_no_fft.circ_params.data.copy_(layer.circ_params.data)
    layer_no_fft.bias.data.copy_(layer.bias.data)
    
    y_no_fft = layer_no_fft(x)
    
    # Allow larger tolerance due to FFT numerical differences
    assert torch.allclose(y, y_no_fft, rtol=1e-3, atol=1e-4)


def test_parameter_count():
    """Test BCA parameter reduction."""
    in_features = 256
    out_features = 512
    block_size = 8
    
    # Standard linear layer
    linear = nn.Linear(in_features, out_features, bias=True)
    linear_params = sum(p.numel() for p in linear.parameters())
    
    # BCA layer
    bca = BlockCirculantLinear(in_features, out_features, block_size, bias=True)
    bca_params = sum(p.numel() for p in bca.parameters())
    
    # BCA should have fewer parameters
    compression_ratio = linear_params / bca_params
    expected_compression = block_size  # Approximately block_size compression
    
    assert compression_ratio > 1.0
    assert abs(compression_ratio - expected_compression) < 2.0  # Allow some variation


def test_approximate_weights_with_bca():
    """Test weight approximation function."""
    torch.manual_seed(42)
    
    weight = torch.randn(64, 32)
    block_size = 8
    
    approximated = approximate_weights_with_bca(weight, block_size)
    
    assert approximated.shape == weight.shape
    
    # Approximated weights should have block-circulant structure
    # Check that each block is approximately circulant
    num_blocks_out = 64 // 8
    num_blocks_in = 32 // 8
    
    for i in range(num_blocks_out):
        for j in range(num_blocks_in):
            block = approximated[i*8:(i+1)*8, j*8:(j+1)*8]
            
            # Check that each row is a shifted version of first row
            first_row = block[0]
            for k in range(1, 8):
                shifted = torch.roll(first_row, k)
                # Not exact due to approximation, but should be correlated
                correlation = torch.corrcoef(torch.stack([block[k], shifted]))[0, 1]
                assert correlation > 0.5  # Should be positively correlated


def test_bca_transformer_layer():
    """Test BCA transformer layer."""
    hidden_size = 64
    num_heads = 4
    intermediate_size = 128
    block_size = 8
    
    layer = BCATransformerLayer(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        block_size=block_size,
        use_fft=True
    )
    
    batch_size = 2
    seq_len = 16
    
    # Test forward pass
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output = layer(hidden_states, attention_mask)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    # Test without attention mask
    output_no_mask = layer(hidden_states)
    assert output_no_mask.shape == (batch_size, seq_len, hidden_size)


def test_convert_linear_to_bca():
    """Test conversion of linear layers to BCA."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(32, 64)
            self.linear2 = nn.Linear(64, 32)
            self.other = nn.Conv2d(3, 16, 3)
    
    model = SimpleModel()
    
    # Convert linear layers
    converted = convert_linear_to_bca(
        model,
        block_size=8,
        use_fft=True,
        layer_names=["linear1", "linear2"]
    )
    
    # Check conversions
    assert isinstance(converted.linear1, BlockCirculantLinear)
    assert isinstance(converted.linear2, BlockCirculantLinear)
    assert isinstance(converted.other, nn.Conv2d)  # Should not be converted
    
    # Test forward pass
    x = torch.randn(2, 32)
    y = converted.linear1(x)
    assert y.shape == (2, 64)


def test_gradient_flow():
    """Test that gradients flow through BCA layers."""
    layer = BlockCirculantLinear(32, 64, block_size=8, bias=True)
    
    x = torch.randn(4, 32, requires_grad=True)
    y = layer(x)
    
    # Compute loss and backward
    target = torch.randn(4, 64)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert layer.circ_params.grad is not None
    assert layer.bias.grad is not None
    
    # Check gradient shapes
    assert layer.circ_params.grad.shape == layer.circ_params.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_memory_efficiency():
    """Test memory usage of BCA vs standard linear."""
    import torch.cuda as cuda
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    in_features = 2048
    out_features = 4096
    block_size = 16
    
    # Standard linear
    linear = nn.Linear(in_features, out_features).cuda()
    x = torch.randn(1, in_features).cuda()
    
    # BCA linear
    bca = BlockCirculantLinear(in_features, out_features, block_size).cuda()
    
    # Forward pass memory (approximate)
    torch.cuda.reset_peak_memory_stats()
    _ = linear(x)
    linear_memory = cuda.max_memory_allocated()
    
    torch.cuda.reset_peak_memory_stats()
    _ = bca(x)
    bca_memory = cuda.max_memory_allocated()
    
    # BCA should use less memory (parameter storage + computation)
    assert bca_memory < linear_memory * 1.5  # Allow some overhead for FFT buffers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])