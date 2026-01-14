#!/usr/bin/env python3
"""
Visualize verifier dataset samples in a comparison figure.

Creates a side-by-side comparison of ACCEPT vs REJECT examples to verify
the dataset quality before training.

Usage:
    python scripts/custom/visualize_verifier_dataset.py \
        --dataset-dir ./data/verifier-dataset \
        --output ./data/verifier-dataset/comparison_figure.png \
        --num-samples 2
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def analyze_overlay_coverage(image_path: str) -> float:
    """
    Estimate the percentage of pixels covered by green overlay.
    
    Args:
        image_path: Path to the overlay image
        
    Returns:
        Coverage percentage (0-100)
    """
    img = np.array(Image.open(image_path))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    # Detect green overlay (green channel significantly higher than red and blue)
    green_mask = (g.astype(int) - r.astype(int) > 20) & \
                 (g.astype(int) - b.astype(int) > 20)
    
    coverage = 100 * green_mask.sum() / green_mask.size
    return coverage


def create_comparison_figure(
    dataset_dir: str,
    output_path: str,
    num_samples: int = 2,
    show_coverage: bool = True,
):
    """
    Create a comparison figure showing ACCEPT vs REJECT samples.
    
    Args:
        dataset_dir: Directory containing the verifier dataset
        output_path: Where to save the output figure
        num_samples: Number of sample rows to show
        show_coverage: Whether to show overlay coverage percentage
    """
    images_dir = os.path.join(dataset_dir, "images")
    train_json = os.path.join(dataset_dir, "train.json")
    
    if not os.path.exists(train_json):
        print(f"Error: {train_json} not found")
        sys.exit(1)
    
    # Load dataset to get queries
    with open(train_json) as f:
        data = json.load(f)
    
    # Group samples by original index
    samples_by_idx = {}
    for sample in data:
        # Parse sample ID like "0_pos", "0_neg_hard_0", etc.
        parts = sample['id'].split('_')
        idx = int(parts[0])
        if idx not in samples_by_idx:
            samples_by_idx[idx] = {'query': sample['query'], 'files': {}}
        samples_by_idx[idx]['files'][sample['id']] = sample
    
    # Limit to requested number of samples
    sample_indices = sorted(samples_by_idx.keys())[:num_samples]
    
    # Create figure: 5 columns (1 accept + 2 hard reject + 2 easy reject)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = [
        'ACCEPT\n(correct mask)',
        'REJECT hard\n(other semantic obj)',
        'REJECT hard\n(other semantic obj)',
        'REJECT easy\n(mask from diff img)',
        'REJECT easy\n(mask from diff img)',
    ]
    
    for row, idx in enumerate(sample_indices):
        query = samples_by_idx[idx]['query']
        
        # Expected file patterns
        file_patterns = [
            f"{idx}_pos_accept.jpg",
            f"{idx}_neg_hard_0_reject.jpg",
            f"{idx}_neg_hard_1_reject.jpg",
            f"{idx}_neg_easy_0_reject.jpg",
            f"{idx}_neg_easy_1_reject.jpg",
        ]
        
        for col, (filename, col_title) in enumerate(zip(file_patterns, column_titles)):
            img_path = os.path.join(images_dir, filename)
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[row, col].imshow(img)
                
                if show_coverage:
                    coverage = analyze_overlay_coverage(img_path)
                    title = f"{col_title}\n({coverage:.1f}% coverage)"
                else:
                    title = col_title
                    
                axes[row, col].set_title(title, fontsize=10)
            else:
                axes[row, col].text(0.5, 0.5, 'Not found', ha='center', va='center')
                axes[row, col].set_title(col_title, fontsize=10)
            
            axes[row, col].axis('off')
        
        # Add query label on the left
        fig.text(0.02, (num_samples - row - 0.5) / num_samples, 
                 f'Query:\n"{query}"', 
                 fontsize=11, fontweight='bold', 
                 rotation=90, va='center', ha='center')
    
    plt.suptitle('Verifier Dataset: ACCEPT vs REJECT Examples', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.06)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure to: {output_path}")
    
    # Also print statistics
    print("\n=== Coverage Statistics ===")
    for idx in sample_indices:
        query = samples_by_idx[idx]['query']
        print(f'\nSample {idx}: Query = "{query}"')
        
        file_patterns = [
            (f"{idx}_pos_accept.jpg", "ACCEPT"),
            (f"{idx}_neg_hard_0_reject.jpg", "REJECT hard 0"),
            (f"{idx}_neg_hard_1_reject.jpg", "REJECT hard 1"),
            (f"{idx}_neg_easy_0_reject.jpg", "REJECT easy 0"),
            (f"{idx}_neg_easy_1_reject.jpg", "REJECT easy 1"),
        ]
        
        for filename, label in file_patterns:
            img_path = os.path.join(images_dir, filename)
            if os.path.exists(img_path):
                coverage = analyze_overlay_coverage(img_path)
                print(f"  {coverage:5.1f}% | {label}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize verifier dataset samples"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/verifier-dataset",
        help="Directory containing the verifier dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/verifier-dataset/comparison_figure.png",
        help="Output path for the comparison figure"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of sample rows to visualize"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Don't show coverage percentages in titles"
    )
    args = parser.parse_args()
    
    create_comparison_figure(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        num_samples=args.num_samples,
        show_coverage=not args.no_coverage,
    )


if __name__ == "__main__":
    main()

