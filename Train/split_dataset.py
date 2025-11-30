#!/usr/bin/env python3
"""
Split JSONL dataset into training and validation sets for fine-tuning.

Usage:
    python split_dataset.py <input_file> [--val-ratio VAL_RATIO] [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import random
from pathlib import Path


def split_jsonl(input_file: str, val_ratio: float = 0.1, output_dir: str = "Datasets", seed: int = 42):
    """
    Split a JSONL file into training and validation sets.
    
    Args:
        input_file: Path to input JSONL file
        val_ratio: Ratio of validation set (default: 0.1 = 10%)
        output_dir: Directory to save output files (default: "Datasets")
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all lines from input file
    print(f"Reading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"Total examples: {total_lines}")
    
    # Shuffle lines
    random.shuffle(lines)
    
    # Calculate split point
    val_size = int(total_lines * val_ratio)
    train_size = total_lines - val_size
    
    train_lines = lines[:train_size]
    val_lines = lines[train_size:]
    
    print(f"Training examples: {train_size} ({train_size/total_lines*100:.1f}%)")
    print(f"Validation examples: {val_size} ({val_size/total_lines*100:.1f}%)")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base filename
    input_path = Path(input_file)
    base_name = input_path.stem
    
    # Write training set
    train_file = os.path.join(output_dir, f"{base_name}_train.jsonl")
    print(f"\nWriting training set to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    # Write validation set
    val_file = os.path.join(output_dir, f"{base_name}_val.jsonl")
    print(f"Writing validation set to {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line + '\n')
    
    print(f"\n✓ Dataset split complete!")
    print(f"  Training: {train_file}")
    print(f"  Validation: {val_file}")
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(
        description="Split JSONL dataset into training and validation sets"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio of validation set (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='Datasets',
        help='Directory to save output files (default: Datasets)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return 1
    
    if not (0 < args.val_ratio < 1):
        print(f"Error: val-ratio must be between 0 and 1, got {args.val_ratio}")
        return 1
    
    split_jsonl(args.input_file, args.val_ratio, args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    exit(main())

