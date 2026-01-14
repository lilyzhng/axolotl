#!/usr/bin/env python3
"""
Analyze overlay coverage in verifier dataset images.

This script analyzes the generated verifier dataset to show statistics about
mask coverage for ACCEPT vs REJECT samples, helping verify dataset quality.

Usage:
    python scripts/custom/analyze_coverage.py \
        --dataset-dir ./data/verifier-dataset \
        --num-samples 10
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image


def analyze_overlay_coverage(image_path: str) -> float:
    """
    Estimate the percentage of pixels covered by green overlay.
    
    The overlay is detected by finding pixels where the green channel
    is significantly higher than red and blue channels.
    
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


def analyze_dataset(dataset_dir: str, num_samples: int = None, verbose: bool = True):
    """
    Analyze coverage statistics for the verifier dataset.
    
    Args:
        dataset_dir: Directory containing the verifier dataset
        num_samples: Maximum number of samples to analyze (None = all)
        verbose: Whether to print detailed per-sample statistics
        
    Returns:
        Dictionary with coverage statistics
    """
    images_dir = os.path.join(dataset_dir, "images")
    train_json = os.path.join(dataset_dir, "train.json")
    
    if not os.path.exists(train_json):
        print(f"Error: {train_json} not found")
        sys.exit(1)
    
    # Load dataset
    with open(train_json) as f:
        data = json.load(f)
    
    # Group by sample index and type
    samples_by_idx = defaultdict(dict)
    for sample in data:
        parts = sample['id'].split('_')
        idx = int(parts[0])
        sample_type = '_'.join(parts[1:])  # e.g., "pos", "neg_hard_0", "neg_easy_1"
        samples_by_idx[idx][sample_type] = sample
    
    # Limit samples if requested
    sample_indices = sorted(samples_by_idx.keys())
    if num_samples:
        sample_indices = sample_indices[:num_samples]
    
    # Collect statistics
    stats = {
        'accept': [],
        'reject_hard': [],
        'reject_easy': [],
    }
    
    for idx in sample_indices:
        samples = samples_by_idx[idx]
        query = samples.get('pos', {}).get('query', 'Unknown')
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Sample {idx}: Query = \"{query}\"")
            print('='*70)
        
        for sample_type, sample in sorted(samples.items()):
            # Get image path
            image_path = sample['images'][0] if sample['images'] else None
            if not image_path or not os.path.exists(image_path):
                continue
            
            coverage = analyze_overlay_coverage(image_path)
            label = sample['label']
            
            # Categorize
            if 'pos' in sample_type:
                stats['accept'].append(coverage)
                category = 'ACCEPT'
            elif 'hard' in sample_type:
                stats['reject_hard'].append(coverage)
                category = 'REJECT hard'
            elif 'easy' in sample_type:
                stats['reject_easy'].append(coverage)
                category = 'REJECT easy'
            else:
                category = label
            
            if verbose:
                print(f"  {coverage:5.1f}% | {category} ({sample_type})")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("COVERAGE SUMMARY STATISTICS")
    print('='*70)
    
    for category, coverages in stats.items():
        if coverages:
            arr = np.array(coverages)
            print(f"\n{category.upper().replace('_', ' ')} ({len(coverages)} samples):")
            print(f"  Mean:   {arr.mean():5.1f}%")
            print(f"  Std:    {arr.std():5.1f}%")
            print(f"  Min:    {arr.min():5.1f}%")
            print(f"  Max:    {arr.max():5.1f}%")
            print(f"  Median: {np.median(arr):5.1f}%")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze overlay coverage in verifier dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/verifier-dataset",
        help="Directory containing the verifier dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary statistics, not per-sample details"
    )
    args = parser.parse_args()
    
    analyze_dataset(
        dataset_dir=args.dataset_dir,
        num_samples=args.num_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

