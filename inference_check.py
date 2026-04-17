import torch
from transformers import AutoTokenizer
from src.model_utils import load_bca_model
import sys
import os
import logging

# Fix Windows terminal encoding for output that might contain non-ASCII characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InferenceCheck")

def run_inference(model_path, prompt, device="cpu"):
    logger.info(f"Loading model and tokenizer from {model_path}...")
    
    # Load model and tokenizer using our utility
    model, tokenizer = load_bca_model(model_path, device=device)
    model.eval()

    logger.info(f"Generating for prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set generation parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=True, 
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Check if checkpoint exists
    checkpoint_path = os.path.join("checkpoints", "epoch_0")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        exit(1)
        
    prompt = "The quick brown fox"
    result = run_inference(checkpoint_path, prompt)
    
    print("\n" + "="*50)
    print("INFERENCE RESULT")
    print("="*50)
    print(f"Prompt: {prompt}")
    print(f"Result: {result}")
    print("="*50)
