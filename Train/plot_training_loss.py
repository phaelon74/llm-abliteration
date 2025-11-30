#!/usr/bin/env python3
"""
Plot training loss from Axolotl training log file.

This script parses the training.log file and extracts loss values,
then creates visualization plots showing loss progression over training.
"""

import argparse
import re
import json
import matplotlib.pyplot as plt
from pathlib import Path


def parse_training_log(log_file: str):
    """
    Parse training log file and extract loss, epoch, and learning rate values.
    
    Args:
        log_file: Path to training.log file
        
    Returns:
        List of dictionaries with step, loss, epoch, learning_rate
    """
    losses = []
    step = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for lines with loss dictionaries
            if "'loss':" in line or "'loss':" in line:
                # Try to extract the dictionary
                match = re.search(r"\{[^}]+\}", line)
                if match:
                    try:
                        # Replace single quotes with double quotes for JSON parsing
                        dict_str = match.group(0).replace("'", '"')
                        # Fix Python-specific values (True, False, None)
                        dict_str = dict_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                        data = json.loads(dict_str)
                        
                        if 'loss' in data:
                            losses.append({
                                'step': step,
                                'loss': data['loss'],
                                'epoch': data.get('epoch', 0),
                                'learning_rate': data.get('learning_rate', 0),
                                'grad_norm': data.get('grad_norm', 0)
                            })
                            step += 1
                    except (json.JSONDecodeError, ValueError):
                        continue
    
    return losses


def plot_training_metrics(losses: list, output_dir: str = "."):
    """
    Create plots from training loss data.
    
    Args:
        losses: List of loss dictionaries
        output_dir: Directory to save plots
    """
    if not losses:
        print("No loss data found in log file!")
        return
    
    steps = [d['step'] for d in losses]
    loss_values = [d['loss'] for d in losses]
    epochs = [d['epoch'] for d in losses]
    learning_rates = [d['learning_rate'] for d in losses]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss vs Steps
    ax1 = axes[0, 0]
    ax1.plot(steps, loss_values, 'b-', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs Steps')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=loss_values[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {loss_values[-1]:.4f}')
    ax1.legend()
    
    # Plot 2: Loss vs Epoch
    ax2 = axes[0, 1]
    ax2.plot(epochs, loss_values, 'g-', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss vs Epoch')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate vs Steps
    ax3 = axes[1, 0]
    ax3.plot(steps, learning_rates, 'orange', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Loss distribution (histogram)
    ax4 = axes[1, 1]
    ax4.hist(loss_values, bins=20, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Loss Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Loss Distribution')
    ax4.axvline(x=loss_values[-1], color='r', linestyle='--', linewidth=2, label=f'Final: {loss_values[-1]:.4f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'training_loss_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {output_file}")
    
    # Also create a simple loss-only plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, loss_values, 'b-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Loss Progression\n(Started: {loss_values[0]:.4f}, Final: {loss_values[-1]:.4f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=loss_values[-1], color='r', linestyle='--', alpha=0.5, label=f'Final Loss: {loss_values[-1]:.4f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file2 = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✓ Loss plot saved to: {output_file2}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"{'='*60}")
    print(f"Total steps: {len(losses)}")
    print(f"Initial loss: {loss_values[0]:.4f}")
    print(f"Final loss: {loss_values[-1]:.4f}")
    print(f"Loss reduction: {loss_values[0] - loss_values[-1]:.4f} ({((loss_values[0] - loss_values[-1]) / loss_values[0] * 100):.1f}%)")
    print(f"Min loss: {min(loss_values):.4f} (at step {steps[loss_values.index(min(loss_values))]})")
    print(f"Max loss: {max(loss_values):.4f} (at step {steps[loss_values.index(max(loss_values))]})")
    print(f"Final epoch: {epochs[-1]:.2f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss from Axolotl training log"
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='training.log',
        help='Path to training.log file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save plots (default: current directory)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    print(f"Parsing training log: {args.log_file}")
    losses = parse_training_log(args.log_file)
    
    if not losses:
        print("No loss data found in log file!")
        print("Make sure the log file contains loss dictionaries in the format:")
        print("  {'loss': 5.3, 'epoch': 0.22, ...}")
        return 1
    
    print(f"Found {len(losses)} loss entries")
    print(f"Creating plots...")
    
    plot_training_metrics(losses, args.output_dir)
    
    return 0


if __name__ == "__main__":
    import os
    exit(main())

