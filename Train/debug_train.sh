#!/bin/bash
# Debug script to run Axolotl training with verbose output
# This will help identify the actual error

set -e

# Enable PyTorch error reporting
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_LOGS=+all

# Set PyTorch CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run with verbose output
cd "$(dirname "$0")"

echo "Current directory: $(pwd)"
echo "Config file: finetune_gpt_oss_20b.yaml"
echo "Checking dataset paths..."

# Check if datasets exist
if [ ! -f "../Datasets/dataset_20251128_154142_train.jsonl" ]; then
    echo "ERROR: Training dataset not found at ../Datasets/dataset_20251128_154142_train.jsonl"
    exit 1
fi

if [ ! -f "../Datasets/dataset_20251128_154142_val.jsonl" ]; then
    echo "ERROR: Validation dataset not found at ../Datasets/dataset_20251128_154142_val.jsonl"
    exit 1
fi

echo "Datasets found. Starting training with verbose output..."
echo ""

# Run training with Python's -u flag for unbuffered output
axolotl train finetune_gpt_oss_20b.yaml 2>&1 | tee training.log

