"""
MeZO + BCA Framework for CPU-efficient LLM Fine-tuning.
"""

from .bca_layers import (
    BlockCirculantLinear,
    BCATransformerLayer,
    convert_linear_to_bca,
    approximate_weights_with_bca
)

from .mezo_optimizer import (
    MeZOOptimizer
)


from .model_utils import (
    load_qwen_model,
    convert_qwen_to_bca,
    prepare_model_for_mezo,
    save_bca_model,
    load_bca_model,
    count_parameters,
    get_model_size_mb
)

from .trainer import (
    MeZOBCATrainer,
    load_config,
    create_dataloader
)

from .logger_utils import (
    setup_logger,
    logger
)

__version__ = "0.1.0"
__author__ = "Skripsi Research"
__all__ = [
    # BCA layers
    "BlockCirculantLinear",
    "BCATransformerLayer",
    "convert_linear_to_bca",
    "approximate_weights_with_bca",
    
    # MeZO optimizer
    "MeZOOptimizer",

    
    # Model utilities
    "load_qwen_model",
    "convert_qwen_to_bca",
    "prepare_model_for_mezo",
    "save_bca_model",
    "load_bca_model",
    "count_parameters",
    "get_model_size_mb",
    
    # Trainer
    "MeZOBCATrainer",
    "load_config",
    "create_dataloader",
    
    # Utilities
    "setup_logger",
    "logger",
]