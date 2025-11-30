#!/usr/bin/env python3
"""
Pre-dequantize MXFP4 quantized model to bf16 to avoid GPU OOM during training.

This script loads the quantized model, dequantizes it to bf16, and saves it
so that Axolotl training doesn't need to dequantize during loading.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path


def dequantize_model(input_path: str, output_path: str, device: str = "cpu"):
    """
    Load MXFP4 quantized model and save as bf16.
    
    Args:
        input_path: Path to quantized model
        output_path: Path to save dequantized model
        device: Device to use for dequantization (cpu recommended to avoid OOM)
    """
    print(f"Loading quantized model from {input_path}...")
    print(f"Using device: {device}")
    
    # Load model on CPU to avoid GPU memory issues during dequantization
    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    
    print(f"Model loaded. Saving dequantized model to {output_path}...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Dequantized model saved to {output_path}")
    print(f"  You can now use this path in your Axolotl config instead of the original model")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-dequantize MXFP4 quantized model to bf16"
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to quantized model directory'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path to save dequantized model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for dequantization (default: cpu to avoid OOM)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input path '{args.input_path}' does not exist!")
        return 1
    
    try:
        dequantize_model(args.input_path, args.output_path, args.device)
        return 0
    except Exception as e:
        print(f"Error during dequantization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

