"""
Utilities for loading and converting Qwen models with BCA layers.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig
)
from typing import Optional, Dict, Any, List, Union
import os
import json

from .bca_layers import (
    BlockCirculantLinear,
    convert_linear_to_bca,
    approximate_weights_with_bca
)


def load_qwen_model(
    model_name: str = "Qwen/Qwen3.5-0.8B",
    use_bca: bool = True,
    block_size: int = 8,
    use_fft: bool = True,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,  # kept for backwards compat
    dtype: torch.dtype = None
) -> PreTrainedModel:
    """
    Load Qwen model and optionally convert to BCA layers.
    
    Args:
        model_name: Hugging Face model name or path
        use_bca: Whether to convert linear layers to BCA
        block_size: Block size for circulant matrices
        use_fft: Whether to use FFT for BCA layers
        device: Device to load model on
        torch_dtype: Data type for model
    
    Returns:
        Loaded model (with BCA conversion if requested)
    """
    from .logger_utils import logger
    logger.info(f"Loading model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use `dtype` (new API); fall back to torch_dtype for compat
    _dtype = dtype if dtype is not None else torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_dtype,          # new non-deprecated kwarg
        low_cpu_mem_usage=True,
        device_map=device if device != "cpu" else None
    )
    
    if device == "cpu":
        model = model.to(torch.float32)  # Ensure CPU uses float32
    
    # Convert to BCA if requested
    if use_bca:
        print(f"Converting model to BCA with block_size={block_size}, use_fft={use_fft}...")
        model = convert_qwen_to_bca(
            model,
            block_size=block_size,
            use_fft=use_fft,
            layer_types=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"]
        )
    
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully. Parameters: {count_parameters(model):,}")
    logger.info(f"Memory footprint: {get_model_size_mb(model):.2f} MB")

    return model, tokenizer


def convert_qwen_to_bca(
    model: PreTrainedModel,
    block_size: int = 8,
    use_fft: bool = True,
    layer_types: List[str] = None
) -> PreTrainedModel:
    """
    Convert Qwen model's linear layers to BCA layers.
    
    Args:
        model: Qwen model
        block_size: Block size for circulant matrices
        use_fft: Whether to use FFT for BCA layers
        layer_types: List of layer type names to convert
    
    Returns:
        Model with BCA layers
    """
    if layer_types is None:
        layer_types = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    
    # Get the transformer model (model.model for Qwen)
    transformer = model.model if hasattr(model, 'model') else model
    
    # Count original parameters
    original_params = count_parameters(transformer)
    
    # Convert linear layers in each transformer block
    for layer_idx, layer in enumerate(transformer.layers):
        for layer_type in layer_types:
            if hasattr(layer, layer_type):
                linear_layer = getattr(layer, layer_type)
                
                if isinstance(linear_layer, nn.Linear):
                    # Create BCA layer
                    bca_layer = BlockCirculantLinear(
                        in_features=linear_layer.in_features,
                        out_features=linear_layer.out_features,
                        block_size=block_size,
                        bias=linear_layer.bias is not None,
                        use_fft=use_fft,
                        device=linear_layer.weight.device,
                        dtype=linear_layer.weight.dtype
                    )
                    
                    # Approximate original weights with BCA structure
                    with torch.no_grad():
                        # Simple initialization: copy first row of each block
                        # from the original weights
                        weight = linear_layer.weight.data
                        approximated_weight = approximate_weights_with_bca(
                            weight, block_size=block_size)
                    
                    # For BCA layers, we need to set circ_params
                    # We'll initialize circ_params from the approximated weight
                    # by extracting first row of each block
                    out_features, in_features = weight.shape
                    num_blocks_out = out_features // block_size
                    num_blocks_in = in_features // block_size
                    
                    # Reshape approximated weight into blocks
                    weight_blocks = approximated_weight.view(
                        num_blocks_out, block_size, num_blocks_in, block_size
                    )
                    weight_blocks = weight_blocks.permute(0, 2, 1, 3)
                    
                    # Extract first row of each block
                    first_rows = weight_blocks[:, :, 0, :]
                    
                    # Set BCA parameters
                    bca_layer.circ_params.data.copy_(first_rows)
                    
                    # Copy bias if exists
                    if linear_layer.bias is not None:
                        bca_layer.bias.data.copy_(linear_layer.bias.data)
                    
                    # Replace layer
                    setattr(layer, layer_type, bca_layer)
                    
                    print(f"  Converted {layer_type} in layer {layer_idx}")
    
    # Count new parameters
    new_params = count_parameters(transformer)
    compression_ratio = new_params / original_params
    
    print(f"BCA conversion complete. Parameter count: {original_params:,} -> {new_params:,}")
    print(f"Compression ratio: {compression_ratio:.3f}")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)


def prepare_model_for_mezo(
    model: PreTrainedModel,
    freeze_embeddings: bool = True,
    freeze_layers: Optional[List[int]] = None
) -> PreTrainedModel:
    """
    Prepare model for MeZO training by freezing certain layers.
    
    Args:
        model: Model to prepare
        freeze_embeddings: Whether to freeze embedding layers
        freeze_layers: List of layer indices to freeze (None to freeze all but last few)
    
    Returns:
        Prepared model
    """
    # Get transformer model
    transformer = model.model if hasattr(model, 'model') else model
    
    if freeze_layers is None:
        # Freeze all but last 2 layers by default for MeZO
        freeze_layers = list(range(len(transformer.layers) - 2))
    
    # Freeze specified layers
    for layer_idx in freeze_layers:
        if layer_idx < len(transformer.layers):
            layer = transformer.layers[layer_idx]
            for param in layer.parameters():
                param.requires_grad = False
    
    # Freeze embeddings if requested
    if freeze_embeddings:
        if hasattr(transformer, 'embed_tokens'):
            for param in transformer.embed_tokens.parameters():
                param.requires_grad = False
        if hasattr(transformer, 'embed_positions'):
            for param in transformer.embed_positions.parameters():
                param.requires_grad = False
    
    # Freeze output layer if exists
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model prepared for MeZO. Trainable params: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.1f}%)")
    
    return model


def save_bca_model(
    model: PreTrainedModel,
    save_path: str,
    tokenizer = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Save BCA-converted model with custom configuration.
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    # Save configuration
    if config is None:
        config = {}
    
    # Add BCA-specific config
    bca_config = {
        "use_bca": True,
        "bca_block_size": getattr(model, 'bca_block_size', 8),
        "bca_use_fft": getattr(model, 'bca_use_fft', True)
    }
    config.update(bca_config)
    
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    
    print(f"Model saved to {save_path}")


def load_bca_model(
    load_path: str,
    base_model_name: str = None,
    device: str = "cpu"
) -> PreTrainedModel:
    """
    Load BCA-converted model.
    """
    # Load configuration
    config_path = os.path.join(load_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    # Load base model
    if base_model_name is None:
        base_model_name = config.get("_name_or_path", "Qwen/Qwen3.5-0.8B")
    
    model, tokenizer = load_qwen_model(
        model_name=base_model_name,
        use_bca=config.get("use_bca", True),
        block_size=config.get("bca_block_size", 8),
        use_fft=config.get("bca_use_fft", True),
        device=device,
        torch_dtype=torch.float32
    )
    
    # Load weights
    weights_path = os.path.join(load_path, "pytorch_model.bin")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    
    return model, tokenizer