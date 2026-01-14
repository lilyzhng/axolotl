#!/usr/bin/env python3
"""
Simple script to extract and plot training loss from axolotl debug.log

Usage:
    python scripts/custom/plot_training_loss.py --log-file ./outputs/qwen3-vl-verifier/debug.log
"""

import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path


def parse_metrics(log_file):
    """Extract loss metrics from debug.log"""
    train_losses = []
    eval_losses = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find training loss entries: {'loss': X.XXX, ...}
    train_pattern = r"\{'loss': ([\d.]+).*?'epoch': ([\d.]+)\}"
    for match in re.finditer(train_pattern, content):
        loss = float(match.group(1))
        epoch = float(match.group(2))
        train_losses.append((epoch, loss))
    
    # Find eval loss entries: {'eval_loss': X.XXX, ...}
    eval_pattern = r"\{'eval_loss': ([\d.]+).*?'epoch': ([\d.]+)\}"
    for match in re.finditer(eval_pattern, content):
        loss = float(match.group(1))
        epoch = float(match.group(2))
        eval_losses.append((epoch, loss))
    
    return train_losses, eval_losses


def plot_losses(train_losses, eval_losses, output_file):
    """Create a plot of training and eval losses"""
    plt.figure(figsize=(10, 6))
    
    if train_losses:
        epochs, losses = zip(*train_losses)
        plt.plot(epochs, losses, 'b-o', label='Training Loss', markersize=8)
    
    if eval_losses:
        epochs, losses = zip(*eval_losses)
        plt.plot(epochs, losses, 'r-s', label='Eval Loss', markersize=8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    if train_losses:
        for epoch, loss in train_losses:
            plt.annotate(f'{loss:.3f}', (epoch, loss), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")
    
    # Also show summary
    print("\n=== Training Summary ===")
    if train_losses:
        print(f"Training Loss: {train_losses[0][1]:.4f} → {train_losses[-1][1]:.4f}")
    if eval_losses:
        print(f"Eval Loss: {eval_losses[0][1]:.4f} → {eval_losses[-1][1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from axolotl debug.log")
    parser.add_argument(
        "--log-file", 
        type=str, 
        default="./outputs/qwen3-vl-verifier/debug.log",
        help="Path to the debug.log file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output path for the plot (default: same dir as log file)"
    )
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    output_file = args.output or log_file.parent / "training_loss.png"
    
    train_losses, eval_losses = parse_metrics(log_file)
    
    print(f"Found {len(train_losses)} training loss entries")
    print(f"Found {len(eval_losses)} eval loss entries")
    
    if not train_losses and not eval_losses:
        print("No loss data found in log file!")
        return
    
    plot_losses(train_losses, eval_losses, output_file)


if __name__ == "__main__":
    main()
