#!/usr/bin/env python3
"""
Merge LoRA adapter with base model.

This script loads a trained LoRA adapter and merges it with the base model,
creating a merged model ready for inference without needing to load the adapter separately.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import shutil
from pathlib import Path


def merge_lora_adapter(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    device: str = "auto",
    max_memory: dict = None
):
    """
    Merge LoRA adapter with base model.
    
    Args:
        base_model_path: Path to base model directory
        lora_adapter_path: Path to LoRA adapter directory (contains adapter_config.json and adapter_model.safetensors)
        output_path: Path to save merged model
        device: Device to use ("auto", "cpu", "cuda")
        max_memory: Maximum memory per GPU (for device_map)
    """
    print(f"Loading base model from: {base_model_path}")
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    print(f"Output path: {output_path}")
    
    # Check if paths exist
    if not os.path.exists(base_model_path):
        raise ValueError(f"Base model path does not exist: {base_model_path}")
    
    if not os.path.exists(lora_adapter_path):
        raise ValueError(f"LoRA adapter path does not exist: {lora_adapter_path}")
    
    adapter_config_path = os.path.join(lora_adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise ValueError(f"adapter_config.json not found in {lora_adapter_path}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load base model
    print("\n[1/4] Loading base model...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    
    if device == "cpu":
        model_kwargs["device_map"] = "cpu"
    elif max_memory:
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = max_memory
    else:
        model_kwargs["device_map"] = "auto"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs
    )
    print("✓ Base model loaded")
    
    # Load LoRA adapter
    print("\n[2/4] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    print("✓ LoRA adapter loaded")
    
    # Merge adapter
    print("\n[3/4] Merging LoRA adapter into base model...")
    model = model.merge_and_unload()
    print("✓ LoRA adapter merged")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model
    print(f"\n[4/4] Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    print("✓ Merged model saved")
    
    # Copy tokenizer and config files
    print("\nCopying tokenizer and config files...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    config = AutoConfig.from_pretrained(base_model_path)
    config.save_pretrained(output_path)
    
    # Copy other files if they exist
    files_to_copy = [
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    
    for file in files_to_copy:
        src = os.path.join(base_model_path, file)
        dst = os.path.join(output_path, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    print("✓ Tokenizer and config files copied")
    
    print(f"\n{'='*60}")
    print("✓ Merge completed successfully!")
    print(f"{'='*60}")
    print(f"\nMerged model saved to: {output_path}")
    print(f"You can now use this model for inference without loading the adapter separately.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model"
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='/media/fmodels/TheHouseOfTheDude/gpt-oss-20b_uncanned/dequanted/',
        help='Path to base model directory'
    )
    parser.add_argument(
        '--lora_adapter',
        type=str,
        default='./outputs/gpt-oss-20b-creative-writing',
        help='Path to LoRA adapter directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs/gpt-oss-20b-creative-writing-merged',
        help='Path to save merged model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for merging (default: auto)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (useful if GPU memory is insufficient)'
    )
    
    args = parser.parse_args()
    
    if args.cpu:
        args.device = 'cpu'
    
    try:
        merge_lora_adapter(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_adapter,
            output_path=args.output,
            device=args.device
        )
        return 0
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

