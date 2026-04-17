"""
Main trainer for MeZO + BCA fine-tuning on CPU.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
import json
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from .mezo_optimizer import MeZOOptimizer, MeZOTrainer
from .model_utils import load_qwen_model, prepare_model_for_mezo, save_bca_model
from .bca_layers import BlockCirculantLinear
from .logger_utils import logger


class MeZOBCATrainer:
    """
    Trainer for MeZO + BCA fine-tuning.
    
    Features:
    - MeZO optimization for memory efficiency
    - BCA layers for FFT-accelerated CPU computation
    - Gradient checkpointing
    - Mixed precision (optional)
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[nn.Module] = None,
        tokenizer = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        device: str = "cpu"
    ):
        self.config = config
        self.device = device
        
        # Setup device
        if device == "cpu":
            torch.set_num_threads(config.get("hardware", {}).get("num_threads", 8))
            if config.get("hardware", {}).get("use_mkl", True):
                torch.backends.mkl.enabled = True
        
        # Load model if not provided
        if model is None:
            self.model, self.tokenizer = load_qwen_model(
                model_name=config["model"]["name"],
                use_bca=config["model"].get("use_bca", True),
                block_size=config["model"].get("block_size", 8),
                use_fft=config["model"].get("use_fft", True),
                device=device,
                torch_dtype=torch.float32
            )
        else:
            self.model = model
            self.tokenizer = tokenizer
        
        # Prepare model for MeZO
        self.model = prepare_model_for_mezo(
            self.model,
            freeze_embeddings=True,
            freeze_layers=config.get("training", {}).get("freeze_layers", None)
        )
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Optimizer
        self.optimizer = MeZOOptimizer(
            self.model,
            lr=config["training"]["learning_rate"],
            perturbation_epsilon=config["mezo"]["perturbation_epsilon"],
            sampling_type=config["mezo"]["sampling_type"],
            normalize_grad=config["mezo"]["normalize_grad"],
            num_grad_estimates=config["mezo"]["num_grad_estimates"],
            weight_decay=config.get("training", {}).get("weight_decay", 0.01),
            device=device,
            grad_clip=config["mezo"].get("grad_clip", 1.0),
            warmup_steps=config["mezo"].get("warmup_steps", 0),
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Logging
        self.use_wandb = config.get("logging", {}).get("use_wandb", False)
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb is not installed but use_wandb is True. Disabling wandb logging.")
                self.use_wandb = False
            else:
                wandb.init(project="mezo-bca-qwen", config=config)
        
        # Checkpointing
        self.checkpoint_dir = config.get("checkpoint", {}).get("output_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Statistics
        self.train_losses = []
        self.eval_losses = []
        
        logger.info("MeZO-BCA Trainer initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Trainable parameters: {self.count_trainable_parameters():,}")
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step with MeZO.
        
        Args:
            batch: Batch of data
        
        Returns:
            Dictionary with training statistics
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Define loss function for MeZO
        def loss_fn():
            outputs = self.model(**batch)
            return outputs.loss
        
        # Estimate gradient and update
        stats = self.optimizer.train_step(loss_fn)
        
        # Removed redundant forward pass! `stats` already contains the zeroth-order approximated origin loss.
        
        self.global_step += 1
        
        return stats
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Training step
            stats = self.train_step(batch)
            batch_loss = stats["loss"]
            epoch_loss += batch_loss
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
            
            # Logging
            if self.use_wandb:
                wandb.log({
                    "train/loss": batch_loss,
                    "train/learning_rate": self.optimizer.lr,
                    "train/step": self.global_step,
                    "train/epoch": self.epoch
                })
            
            # Checkpointing
            if self.global_step % self.config.get("logging", {}).get("save_steps", 100) == 0:
                self.save_checkpoint(f"step_{self.global_step}")
        
        avg_epoch_loss = epoch_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        return {"train_loss": avg_epoch_loss}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.eval_dataloader)
        
        with torch.no_grad():
            pbar = tqdm(self.eval_dataloader, desc="Evaluation")
            for batch in pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                outputs = self.model(**batch)
                batch_loss = outputs.loss.item()
                total_loss += batch_loss
                
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
        
        avg_loss = total_loss / num_batches
        self.eval_losses.append(avg_loss)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "eval/loss": avg_loss,
                "eval/epoch": self.epoch
            })
        
        return {"eval_loss": avg_loss}
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config["training"]["num_epochs"]
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            logger.info(f"{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_stats = self.train_epoch()
            
            # Evaluate
            if self.eval_dataloader is not None:
                eval_stats = self.evaluate()
                all_stats = {**train_stats, **eval_stats}
                
                # Save best model
                if eval_stats["eval_loss"] < self.best_loss:
                    self.best_loss = eval_stats["eval_loss"]
                    self.save_checkpoint("best_model")
            else:
                all_stats = train_stats
            
            # Print epoch summary
            logger.info(f"Epoch {epoch} summary:")
            for key, value in all_stats.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Save checkpoint
            if epoch % self.config.get("logging", {}).get("save_epochs", 1) == 0:
                self.save_checkpoint(f"epoch_{epoch}")
        
        # Final save
        self.save_checkpoint("final_model")
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}")
        
        # Save model
        save_bca_model(
            model=self.model,
            save_path=checkpoint_path,
            tokenizer=self.tokenizer,
            config={
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                **self.config
            }
        )
        
        # Save optimizer state (ZO-SGD only needs step_count, no Adam momentum state)
        optimizer_state = {
            "step_count": self.optimizer.step_count,
            "lr": self.optimizer.lr,
            "epsilon": self.optimizer.epsilon,
        }
        torch.save(optimizer_state, os.path.join(checkpoint_path, "optimizer.pt"))
        
        # Save training statistics
        stats = {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss
        }
        with open(os.path.join(checkpoint_path, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # Load model
        self.model, self.tokenizer = load_qwen_model(
            model_name=self.config["model"]["name"],
            use_bca=self.config["model"].get("use_bca", True),
            block_size=self.config["model"].get("block_size", 8),
            use_fft=self.config["model"].get("use_fft", True),
            device=self.device,
            torch_dtype=torch.float32
        )
        
        # Load weights
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.state = optimizer_state["optimizer_state"]
            self.optimizer.step_count = optimizer_state["step_count"]
        
        # Load statistics
        stats_path = os.path.join(checkpoint_path, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            self.train_losses = stats["train_losses"]
            self.eval_losses = stats["eval_losses"]
            self.epoch = stats["epoch"]
            self.global_step = stats["global_step"]
            self.best_loss = stats["best_loss"]
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloader(
    dataset,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 1,
    shuffle: bool = True
) -> DataLoader:
    """
    Create dataloader for training.
    
    Args:
        dataset: Hugging Face dataset
        tokenizer: Tokenizer for preprocessing
        max_length: Maximum sequence length
        batch_size: Batch size (MeZO typically uses 1)
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader
    """
    def collate_fn(batch):
        import torch
        if "input_ids" in batch[0]:
            # Already tokenized
            collated = {
                key: torch.stack([torch.tensor(example[key]) if not isinstance(example[key], torch.Tensor) else example[key] for example in batch])
                for key in batch[0].keys()
            }
            if "labels" not in collated:
                collated["labels"] = collated["input_ids"].clone()
            return collated
        
        # Tokenize batch if "text" is present
        texts = [example["text"] for example in batch]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MeZO + BCA Fine-tuning")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu or cuda)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to dataset (optional)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device
    config["hardware"]["device"] = args.device
    
    # Load dataset
    from datasets import load_dataset
    if args.dataset:
        dataset = load_dataset("json", data_files=args.dataset)
    else:
        # Example dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Split dataset
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
    
    # Load model and tokenizer
    model, tokenizer = load_qwen_model(
        model_name=config["model"]["name"],
        use_bca=config["model"].get("use_bca", True),
        block_size=config["model"].get("block_size", 8),
        use_fft=config["model"].get("use_fft", True),
        device=args.device,
        torch_dtype=torch.float32
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        tokenizer,
        max_length=config["training"]["max_length"],
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    eval_dataloader = create_dataloader(
        eval_dataset,
        tokenizer,
        max_length=config["training"]["max_length"],
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )
    
    # Create trainer
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
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()