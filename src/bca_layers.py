"""
Block Circulant Analysis (BCA) layers for efficient CPU-based FFT operations.
Based on circulant matrix approximation: W = F^H diag(c) F
where F is DFT matrix, c is learned vector, and ^H is Hermitian transpose.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import fft, ifft
from typing import Optional, Tuple, Union, List
import math

from .logger_utils import logger


class BlockCirculantLinear(nn.Module):
    """
    Linear layer with block-circulant weight matrix approximation.
    
    Weight matrix W of shape (out_features, in_features) is approximated as
    block-circulant matrix parameterized by a smaller set of parameters.
    For a block size B, the weight matrix is divided into (out_features/B) x (in_features/B)
    blocks, each block is a circulant matrix of size B x B.
    
    Matrix-vector multiplication can be performed efficiently using FFT:
    y = W x = F^H diag(c) F x
    where F is DFT matrix, c is learned complex diagonal.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 8,
        bias: bool = True,
        use_fft: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.use_fft = use_fft
        
        # Validate dimensions are divisible by block_size
        assert in_features % block_size == 0, f"in_features ({in_features}) must be divisible by block_size ({block_size})"
        assert out_features % block_size == 0, f"out_features ({out_features}) must be divisible by block_size ({block_size})"
        
        self.num_blocks_in = in_features // block_size
        self.num_blocks_out = out_features // block_size
        
        # Each block is a circulant matrix parameterized by its first row
        # We store real parameters for the first row of each block
        # Shape: (num_blocks_out, num_blocks_in, block_size)
        self.circ_params = nn.Parameter(
            torch.randn(self.num_blocks_out, self.num_blocks_in, block_size, device=device, dtype=dtype)
        )
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self._init_parameters()
        
        # Precompute FFT plans if using scipy (for CPU)
        if use_fft and not torch.cuda.is_available():
            self._setup_fft_plans()
    
    def _init_parameters(self):
        """Initialize circulant parameters with Xavier/Glorot initialization."""
        # For each block, initialize first row with small random values
        # Scale by sqrt(block_size) to maintain variance
        nn.init.xavier_uniform_(self.circ_params.view(
            self.num_blocks_out * self.num_blocks_in, self.block_size
        ))
        self.circ_params.data /= math.sqrt(self.block_size)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _setup_fft_plans(self):
        """Setup FFT plans for efficient repeated transforms (optional)."""
        # For scipy/numpy FFT, we can precompute twiddle factors
        # In practice, we'll rely on scipy's FFT which uses efficient plans
        pass
    
    def _circ_multiply_fft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply input x by block-circulant weight matrix using FFT.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Reshape x to separate blocks: (..., num_blocks_in, block_size)
        x_shape = x.shape
        x = x.view(*x_shape[:-1], self.num_blocks_in, self.block_size)
        
        # Real FFT along block dimension
        # x_rfft is automatically a complex tensor with shape (..., num_blocks_in, block_size // 2 + 1)
        x_rfft = torch.fft.rfft(x, dim=-1)
        
        # Compute real FFT of circulant parameters
        circ_rfft = torch.fft.rfft(self.circ_params, dim=-1)
        
        # Multiply in frequency domain: for each output block, sum over input blocks
        if len(x_shape) > 2:
            # Batch case
            output_rfft = torch.einsum('...ib,nib->...nb', x_rfft, circ_rfft)
        else:
            # No batch dimension
            output_rfft = torch.einsum('ib,nib->nb', x_rfft, circ_rfft)
        
        # Inverse Real FFT
        output = torch.fft.irfft(output_rfft, n=self.block_size, dim=-1)
        
        # Reshape to (..., out_features)
        output = output.view(*x_shape[:-1], self.out_features)
        
        return output
    
    def _circ_multiply_direct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply input x by block-circulant weight matrix using direct matrix multiplication.
        Used for verification and small block sizes.
        """
        # Construct full weight matrix (inefficient, for verification only)
        W = self.get_full_weight_matrix()
        return F.linear(x, W, self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fft and self.block_size > 1:
            output = self._circ_multiply_fft(x)
        else:
            # Fallback to direct multiplication for small block size or disabled FFT
            output = self._circ_multiply_direct(x)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def get_full_weight_matrix(self) -> torch.Tensor:
        """
        Construct the full weight matrix from circulant parameters.
        This is memory-inefficient and should only be used for verification.
        
        Returns:
            Weight matrix of shape (out_features, in_features)
        """
        W = torch.zeros(self.out_features, self.in_features,
                       device=self.circ_params.device,
                       dtype=self.circ_params.dtype)
        
        for i in range(self.num_blocks_out):
            for j in range(self.num_blocks_in):
                # Get first row of circulant block
                first_row = self.circ_params[i, j]
                # Build circulant matrix
                block = self._build_circulant_matrix(first_row)
                # Place block in weight matrix
                row_start = i * self.block_size
                col_start = j * self.block_size
                W[row_start:row_start + self.block_size,
                  col_start:col_start + self.block_size] = block
        
        return W
    
    def _build_circulant_matrix(self, first_row: torch.Tensor) -> torch.Tensor:
        """Build circulant matrix from its first row."""
        n = self.block_size
        circ = torch.zeros(n, n, device=first_row.device, dtype=first_row.dtype)
        
        for i in range(n):
            circ[i] = torch.roll(first_row, i)
        
        return circ
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'block_size={self.block_size}, bias={self.bias is not None}, ' \
               f'use_fft={self.use_fft}'


class BCATransformerLayer(nn.Module):
    """
    Transformer layer with BCA-approximated linear layers.
    Replaces all linear projections with BlockCirculantLinear layers.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        block_size: int = 8,
        use_fft: bool = True,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.block_size = block_size
        
        # Self-attention projections with BCA
        self.q_proj = BlockCirculantLinear(
            hidden_size, hidden_size, block_size, bias=True, use_fft=use_fft,
            device=device, dtype=dtype
        )
        self.k_proj = BlockCirculantLinear(
            hidden_size, hidden_size, block_size, bias=True, use_fft=use_fft,
            device=device, dtype=dtype
        )
        self.v_proj = BlockCirculantLinear(
            hidden_size, hidden_size, block_size, bias=True, use_fft=use_fft,
            device=device, dtype=dtype
        )
        self.o_proj = BlockCirculantLinear(
            hidden_size, hidden_size, block_size, bias=True, use_fft=use_fft,
            device=device, dtype=dtype
        )
        
        # Feed-forward projections with BCA
        self.gate_proj = BlockCirculantLinear(
            hidden_size, intermediate_size, block_size, bias=True, use_fft=use_fft,
            device=device, dtype=dtype
        )
        self.up_proj = BlockCirculantLinear(
            hidden_size, intermediate_size, block_size, bias=True, use_fft=use_fft,
            device=device, dtype=dtype
        )
        self.down_proj = BlockCirculantLinear(
            intermediate_size, hidden_size, block_size, bias=True, use_fft=use_fft,
            device=device, dtype=dtype
        )
        
        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps,
                                           device=device, dtype=dtype)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps,
                                                    device=device, dtype=dtype)
        
        # Activation
        self.act_fn = nn.SiLU()  # For Qwen's SwiGLU
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Compute Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = hidden_states.shape
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        hidden_states = residual + attn_output
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        ff_output = self.down_proj(self.act_fn(gate) * up)
        
        hidden_states = residual + ff_output
        
        return hidden_states


def convert_linear_to_bca(
    model: nn.Module,
    block_size: int = 8,
    use_fft: bool = True,
    layer_names: List[str] = None
) -> nn.Module:
    """
    Convert linear layers in a model to BlockCirculantLinear layers.
    
    Args:
        model: PyTorch model
        block_size: Size of circulant blocks
        use_fft: Whether to use FFT for multiplication
        layer_names: List of layer names to convert (if None, convert all linear layers)
    
    Returns:
        Model with converted layers
    """
    if layer_names is None:
        # Default list for transformer models
        layer_names = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head", "embed_tokens"
        ]
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer should be converted
            convert = False
            for pattern in layer_names:
                if pattern in name:
                    convert = True
                    break
            
            if convert:
                logger.info(f"Converting layer {name} to BlockCirculantLinear")
                # Get parent module and attribute name
                parent = model
                parts = name.split('.')
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                attr_name = parts[-1]
                
                # Replace with BlockCirculantLinear
                bca_layer = BlockCirculantLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    block_size=block_size,
                    bias=module.bias is not None,
                    use_fft=use_fft,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )
                
                # Copy weights (approximate)
                # Since we can't directly copy, we initialize and then maybe fine-tune
                # Optionally, we could use SVD to approximate weights with circulant structure
                setattr(parent, attr_name, bca_layer)
    
    logger.info("Finished BCA conversion")
    return model


def approximate_weights_with_bca(
    weight: torch.Tensor,
    block_size: int = 8
) -> torch.Tensor:
    """
    Approximate a weight matrix with block-circulant structure.
    
    Args:
        weight: Weight matrix of shape (out_features, in_features)
        block_size: Size of circulant blocks
    
    Returns:
        Approximated weight matrix
    """
    out_features, in_features = weight.shape
    assert out_features % block_size == 0
    assert in_features % block_size == 0
    
    num_blocks_out = out_features // block_size
    num_blocks_in = in_features // block_size
    
    # Reshape weight into blocks
    weight_blocks = weight.view(num_blocks_out, block_size, num_blocks_in, block_size)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3)  # (num_blocks_out, num_blocks_in, block_size, block_size)
    
    # For each block, approximate with circulant matrix
    # Use average of all circulant shifts as approximation
    approximated_blocks = torch.zeros_like(weight_blocks)
    
    for i in range(num_blocks_out):
        for j in range(num_blocks_in):
            block = weight_blocks[i, j]
            # Compute first row as average of all circulant shifts
            first_row = torch.zeros(block_size, device=weight.device, dtype=weight.dtype)
            for k in range(block_size):
                first_row += torch.roll(block.diag(), -k)
            first_row /= block_size
            
            # Build circulant matrix from first row
            circ_block = torch.zeros(block_size, block_size, device=weight.device, dtype=weight.dtype)
            for k in range(block_size):
                circ_block[k] = torch.roll(first_row, k)
            
            approximated_blocks[i, j] = circ_block
    
    # Reshape back to original shape
    approximated_blocks = approximated_blocks.permute(0, 2, 1, 3)
    approximated_weight = approximated_blocks.reshape(out_features, in_features)
    
    return approximated_weight