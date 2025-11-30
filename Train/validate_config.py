#!/usr/bin/env python3
"""
Quick script to validate Axolotl config before training.
"""

import yaml
import sys

def validate_config(config_path):
    """Validate Axolotl config file."""
    print(f"Validating {config_path}...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    errors = []
    warnings = []
    
    # Check for mutually exclusive warmup parameters
    if 'warmup_steps' in config and 'warmup_ratio' in config:
        if config['warmup_steps'] is not None and config['warmup_ratio'] is not None:
            errors.append("Both warmup_steps and warmup_ratio are set - they are mutually exclusive!")
    
    # Check batch size parameters
    batch_params = []
    if 'micro_batch_size' in config:
        batch_params.append('micro_batch_size')
    if 'gradient_accumulation_steps' in config:
        batch_params.append('gradient_accumulation_steps')
    if 'batch_size' in config:
        batch_params.append('batch_size')
    
    if len(batch_params) < 2:
        errors.append(f"Need at least 2 of (micro_batch_size, gradient_accumulation_steps, batch_size). Found: {batch_params}")
    
    if 'gradient_accumulation_steps' in config and 'batch_size' in config:
        if config['gradient_accumulation_steps'] is not None and config['batch_size'] is not None:
            errors.append("Both gradient_accumulation_steps and batch_size are set - use only one!")
    
    # Check dataset paths
    if 'datasets' in config:
        for i, dataset in enumerate(config['datasets']):
            if 'path' in dataset:
                import os
                if not os.path.exists(dataset['path']):
                    warnings.append(f"Dataset {i} path not found: {dataset['path']}")
    
    # Print results
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("\n✓ Config validation passed!")
    return True

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "finetune_gpt_oss_20b.yaml"
    success = validate_config(config_file)
    sys.exit(0 if success else 1)

