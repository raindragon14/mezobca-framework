#!/usr/bin/env python3
"""
Script to run training with MeZO + BCA on Qwen 0.8B.
"""
import torch
import os
import sys
import argparse
import yaml

from src.trainer import MeZOBCATrainer, load_config, create_dataloader
from src.model_utils import load_qwen_model
from src.logger_utils import logger
from datasets import load_dataset


def _format_tool_call_to_chatml(messages: list, tokenizer) -> str:
    """
    Convert a list of messages (multi-turn tool-calling format) into a 
    single ChatML string that Qwen understands natively.
    
    Example output:
        <|im_start|>system\n...<|im_end|>
        <|im_start|>user\n...<|im_end|>
        <|im_start|>assistant\n...<tool_call>...</tool_call><|im_end|>
        <|im_start|>tool\n...<|im_end|>
        <|im_start|>assistant\nFinal answer<|im_end|>
    """
    import json
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        
        # Handle tool_calls embedded in assistant messages
        if role == "assistant" and "tool_calls" in msg:
            tool_str = ""
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                tool_str += f"<tool_call>\n{json.dumps({'name': fn['name'], 'arguments': json.loads(fn['arguments'])}, ensure_ascii=False, indent=2)}\n</tool_call>"
            content = (content + "\n" + tool_str).strip()
        
        # Tool results use <tool_response> tag
        if role == "tool":
            role = "tool"
            content = f"<tool_response>\n{content}\n</tool_response>"
        
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def prepare_dataset(tokenizer, config):
    """Prepare dataset for training. Supports both plain text and multi-turn tool-calling format."""
    dataset_config = config.get("dataset", {})
    dataset_path = dataset_config.get("path", "data/sample_dataset.json")
    dataset_name = dataset_config.get("name", None)
    dataset_format = dataset_config.get("format", "text")  # "text" or "tool_calling"
    max_length = config["training"]["max_length"]
    
    logger.info(f"Loading dataset from {dataset_path} [format={dataset_format}]")
    
    if dataset_name:
        dataset = load_dataset(dataset_name)
    elif dataset_path.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_path)
    elif dataset_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=dataset_path)
    else:
        try:
            dataset = load_dataset(dataset_path)
        except Exception:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    if dataset_format == "tool_calling":
        # Multi-turn ChatML conversion: messages list → single tokenized string
        def preprocess_tool_calling(examples):
            texts = []
            for msgs in examples["messages"]:
                texts.append(_format_tool_call_to_chatml(msgs, tokenizer))
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        tokenized_dataset = dataset.map(
            preprocess_tool_calling,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
    else:
        # Standard plain-text format (original behavior)
        text_column = dataset_config.get("text_column", "text")
        
        def preprocess_function(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
    
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Qwen with MeZO + BCA")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (overrides config)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset path (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu or cuda)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--block_size", type=int, default=None,
                       help="BCA block size (overrides config)")
    parser.add_argument("--no_bca", action="store_true",
                       help="Disable BCA layers")
    parser.add_argument("--no_mezo", action="store_true",
                       help="Disable MeZO (use Adam instead)")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode (smaller dataset, fewer epochs)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Dry run mode (runs only 1 batch, limits dataset strictly)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config["model"]["name"] = args.model
    if args.dataset:
        config["dataset"] = {"path": args.dataset, "text_column": "text"}
    if args.output_dir:
        config["checkpoint"]["output_dir"] = args.output_dir
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.block_size:
        config["model"]["block_size"] = args.block_size
    if args.no_bca:
        config["model"]["use_bca"] = False
    if args.no_mezo:
        config["training"]["optimizer"] = "adam"
    
    # Debug mode
    if args.debug or args.dry_run:
        logger.info(f"{'DRY RUN' if args.dry_run else 'DEBUG'} MODE: Using minimal dataset and 1 epoch")
        config["training"]["num_epochs"] = 1
        config["training"]["batch_size"] = 1
        config["dataset"] = {
            "path": "data/tool_calling_dataset.json",
            "format": "tool_calling"
        }
        config["model"]["use_bca"] = True
        config["model"]["block_size"] = 8
    
    # Set device
    config["hardware"]["device"] = args.device
    
    # Hardware optimization for CPU
    if args.device == 'cpu':
        num_threads = config.get("hardware", {}).get("num_threads", None)
        if num_threads:
            torch.set_num_threads(num_threads)
            logger.info(f"PyTorch CPU threads manually locked to {num_threads} for performance.")
        
        use_mkl = config.get("hardware", {}).get("use_mkl", True)
        if use_mkl and torch.backends.mkl.is_available():
            logger.info("Intel MKL is explicitly active for CPU matrix accelerations.")
    # Load model and tokenizer
    logger.info("Loading model...")
    hw_config = config.get("hardware", {})
    use_bfloat16 = hw_config.get("use_bfloat16", False)
    compile_model = hw_config.get("compile_model", False)

    model, tokenizer = load_qwen_model(
        model_name=config["model"]["name"],
        use_bca=config["model"].get("use_bca", True),
        block_size=config["model"].get("block_size", 8),
        use_fft=config["model"].get("use_fft", True),
        device=args.device,
        dtype=torch.bfloat16 if use_bfloat16 else torch.float32
    )

    if compile_model:
        logger.info("Attempting to compile model with torch.compile() (Inductor backend)...")
        try:
            # Note: Compilation on Windows CPU can occasionally fail depending on the C++ installation.
            model = torch.compile(model)
            logger.info("Model successfully queued for compilation.")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}. Falling back to eager mode.")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    tokenized_dataset = prepare_dataset(tokenizer, config)
    
    if args.dry_run:
        logger.info("Dry run: Truncating dataset to 2 examples")
        tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(2, len(tokenized_dataset["train"]))))
        if "validation" in tokenized_dataset:
            tokenized_dataset["validation"] = tokenized_dataset["validation"].select(range(min(2, len(tokenized_dataset["validation"]))))
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        tokenized_dataset["train"],
        tokenizer,
        max_length=config["training"]["max_length"],
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    eval_dataloader = None
    if "validation" in tokenized_dataset:
        eval_dataloader = create_dataloader(
            tokenized_dataset["validation"],
            tokenizer,
            max_length=config["training"]["max_length"],
            batch_size=config["training"]["batch_size"],
            shuffle=False
        )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = MeZOBCATrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    trainer.train(num_epochs=config["training"]["num_epochs"])
    
    logger.info("\nTraining completed!")
    logger.info(f"Best loss: {trainer.best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    
    # Save final model
    final_path = os.path.join(trainer.checkpoint_dir, "final_model")
    trainer.save_checkpoint("final_model")
    
    logger.info(f"\nFinal model saved to: {final_path}")
    
    # Test inference
    logger.info("\nTesting inference with trained model...")
    test_prompt = "The future of artificial intelligence is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(args.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Generated: {generated}")


if __name__ == "__main__":
    main()