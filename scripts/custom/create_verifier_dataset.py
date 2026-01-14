#!/usr/bin/env python3
"""
Create a Verifier Dataset from MaskGroups-HQ.

This script creates ACCEPT/REJECT training data for a segmentation verifier:
- ACCEPT: correct mask group for the query
- REJECT: wrong mask group (hard negatives from same image, easy negatives from other images)

Based on plans.md:
- For each (image, query, correct mask group):
  - 1 Positive example (ACCEPT)
  - 3-6 Negative examples (REJECT):
    a) Wrong mask group from the same image (hard negative)
    b) Wrong mask group from a different image (easy negative)

The output images have mask overlays rendered on them.

Usage:
    python scripts/custom/create_verifier_dataset.py \
        --images-dir ./data/maskgroups-hq/images_resized \
        --output-dir ./data/verifier-dataset \
        --num-negatives 4
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# For RLE decoding
try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not installed. Install with: pip install pycocotools")


DATASET_NAME = "Shengcao1006/MaskGroups-HQ"

# Prompt template from plans.md
SYSTEM_PROMPT = "You are a segmentation QA verifier."
USER_PROMPT_TEMPLATE = 'Query: {query}. Does the highlighted mask group match the query? Answer exactly: ACCEPT or REJECT.'


def decode_rle_mask(rle_mask: Dict) -> np.ndarray:
    """Decode RLE mask to binary numpy array."""
    if HAS_PYCOCOTOOLS:
        # pycocotools expects {'counts': str, 'size': [h, w]}
        return mask_utils.decode(rle_mask).astype(bool)
    else:
        # Simple fallback - just return empty mask
        h, w = rle_mask['size']
        return np.zeros((h, w), dtype=bool)


def combine_masks(mask_pool: List[Dict], indices: List[int]) -> np.ndarray:
    """Combine multiple masks from pool into a single mask."""
    if not indices or not mask_pool:
        return None
    
    combined = None
    for idx in indices:
        if idx < len(mask_pool):
            mask = decode_rle_mask(mask_pool[idx])
            if combined is None:
                combined = mask
            else:
                combined = combined | mask
    
    return combined


def render_overlay(image: Image.Image, mask: np.ndarray, 
                   color: Tuple[int, int, int] = (0, 255, 0),
                   alpha: float = 0.4,
                   outline: bool = True) -> Image.Image:
    """
    Render a semi-transparent mask overlay on the image with optional boundary outline.
    
    Args:
        image: PIL Image (RGB)
        mask: Binary mask (H, W)
        color: Overlay color (R, G, B)
        alpha: Transparency (0-1)
        outline: Whether to draw boundary outline
    
    Returns:
        Image with overlay
    """
    img_array = np.array(image).copy()
    
    if mask is None or mask.sum() == 0:
        return image
    
    # Ensure mask matches image size
    if mask.shape[:2] != img_array.shape[:2]:
        # Resize mask to match image
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_img) > 127
    
    # Apply semi-transparent overlay
    overlay = img_array.copy()
    overlay[mask] = [int(c * alpha + overlay[mask][:, i].mean() * (1 - alpha)) 
                     for i, c in enumerate(color)]
    
    # Blend
    img_array[mask] = (img_array[mask] * (1 - alpha) + 
                       np.array(color) * alpha).astype(np.uint8)
    
    # Draw boundary outline
    if outline:
        from scipy import ndimage
        # Find boundary by dilating and XORing
        dilated = ndimage.binary_dilation(mask, iterations=2)
        boundary = dilated ^ mask
        img_array[boundary] = color  # Solid color for boundary
    
    return Image.fromarray(img_array)


def extract_query(conversations: List[Dict]) -> str:
    """Extract the text query from conversations."""
    for conv in conversations:
        if conv.get('from') == 'human':
            # Extract query text after <mask_pool>\n
            value = conv.get('value', '')
            if '<mask_pool>' in value:
                parts = value.split('<mask_pool>')
                if len(parts) > 1:
                    query = parts[1].strip()
                    return query
            # Fallback: return everything after <image>
            if '<image>' in value:
                return value.split('<image>')[-1].strip()
            return value
    return "unknown query"


def create_verifier_sample(
    image_path: str,
    query: str,
    mask: np.ndarray,
    label: str,  # "ACCEPT" or "REJECT"
    sample_id: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Create a single verifier training sample with overlay image."""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Render overlay
    color = (0, 255, 0) if label == "ACCEPT" else (255, 100, 0)  # Green for correct, orange for wrong
    overlay_image = render_overlay(image, mask, color=color, alpha=0.4)
    
    # Save overlay image
    overlay_filename = f"{sample_id}_{label.lower()}.jpg"
    overlay_path = os.path.join(output_dir, "images", overlay_filename)
    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    overlay_image.save(overlay_path, quality=90)
    
    # Create training sample in chat format
    return {
        "id": sample_id,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT_TEMPLATE.format(query=query)}
                ]
            },
            {
                "role": "assistant",
                "content": label
            }
        ],
        "images": [os.path.abspath(overlay_path)],
        "label": label,
        "query": query,
    }


def main():
    parser = argparse.ArgumentParser(description="Create verifier dataset from MaskGroups-HQ")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./data/maskgroups-hq/images_resized",
        help="Directory containing the original images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/verifier-dataset",
        help="Output directory for the verifier dataset"
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=4,
        help="Number of negative examples per positive (default: 4)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of source samples to process (for testing)"
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.5,
        help="Ratio of hard negatives (same image) vs easy negatives (different image)"
    )
    args = parser.parse_args()
    
    if not HAS_PYCOCOTOOLS:
        print("ERROR: pycocotools is required for mask decoding.")
        print("Install with: pip install pycocotools")
        return
    
    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="validation")
    
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    
    print(f"Processing {len(ds)} samples...")
    print(f"Creating {args.num_negatives} negatives per positive")
    print(f"Hard negative ratio: {args.hard_negative_ratio}")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    all_samples = []
    stats = {"accept": 0, "reject": 0, "skipped": 0}
    
    # Pre-index samples by image for easy negative sampling
    samples_by_image = {}
    for i, sample in enumerate(ds):
        img_name = sample['image']
        if img_name not in samples_by_image:
            samples_by_image[img_name] = []
        samples_by_image[img_name].append(i)
    
    for idx, sample in enumerate(tqdm(ds, desc="Creating verifier samples")):
        try:
            image_filename = sample['image']
            image_path = os.path.join(images_dir, image_filename)
            
            if not os.path.exists(image_path):
                stats["skipped"] += 1
                continue
            
            query = extract_query(sample['conversations'])
            # IMPORTANT: gt_mask_pool contains the ground truth masks
            # gt_mask_group contains indices into gt_mask_pool (NOT mask_pool!)
            # mask_pool is a different set of masks (e.g., SAM proposals)
            gt_mask_pool = sample['gt_mask_pool']
            mask_pool = sample['mask_pool']  # Used for negative sampling
            gt_mask_group = sample['gt_mask_group']
            
            if not gt_mask_group or not gt_mask_pool:
                stats["skipped"] += 1
                continue
            
            # Flatten mask group indices (handle nested lists)
            correct_indices = gt_mask_group[0] if gt_mask_group else []
            if not correct_indices:
                stats["skipped"] += 1
                continue
            
            # === POSITIVE EXAMPLE (ACCEPT) ===
            # Use gt_mask_pool for correct masks (indices refer to this pool)
            correct_mask = combine_masks(gt_mask_pool, correct_indices)
            if correct_mask is not None and correct_mask.sum() > 0:
                pos_sample = create_verifier_sample(
                    image_path=image_path,
                    query=query,
                    mask=correct_mask,
                    label="ACCEPT",
                    sample_id=f"{idx}_pos",
                    output_dir=output_dir,
                )
                all_samples.append(pos_sample)
                stats["accept"] += 1
            
            # === NEGATIVE EXAMPLES (REJECT) ===
            num_hard = int(args.num_negatives * args.hard_negative_ratio)
            num_easy = args.num_negatives - num_hard
            
            # Hard negatives: OTHER semantic objects from same image (using gt_mask_pool)
            # This makes them "hard" because they're meaningful objects, not random SAM segments
            all_gt_indices = list(range(len(gt_mask_pool)))
            available_for_hard = [i for i in all_gt_indices if i not in correct_indices]
            
            for neg_idx in range(num_hard):
                if len(available_for_hard) >= 1:
                    # Sample other semantic masks from gt_mask_pool
                    wrong_size = random.randint(1, min(3, len(available_for_hard)))
                    wrong_indices = random.sample(available_for_hard, wrong_size)
                    wrong_mask = combine_masks(gt_mask_pool, wrong_indices)  # Use gt_mask_pool!
                    
                    if wrong_mask is not None and wrong_mask.sum() > 0:
                        neg_sample = create_verifier_sample(
                            image_path=image_path,
                            query=query,
                            mask=wrong_mask,
                            label="REJECT",
                            sample_id=f"{idx}_neg_hard_{neg_idx}",
                            output_dir=output_dir,
                        )
                        all_samples.append(neg_sample)
                        stats["reject"] += 1
            
            # Easy negatives: correct mask group from a DIFFERENT image
            # (these are easy because the mask doesn't even belong to this image)
            other_images = [img for img in samples_by_image.keys() if img != image_filename]
            
            for neg_idx in range(num_easy):
                if other_images:
                    other_img = random.choice(other_images)
                    other_sample_idx = random.choice(samples_by_image[other_img])
                    other_sample = ds[other_sample_idx]
                    
                    other_gt_mask_pool = other_sample['gt_mask_pool']  # Use gt_mask_pool!
                    other_gt_group = other_sample['gt_mask_group']
                    
                    if other_gt_group and other_gt_mask_pool:
                        other_indices = other_gt_group[0] if other_gt_group else []
                        if other_indices:
                            wrong_mask = combine_masks(other_gt_mask_pool, other_indices)  # Use gt_mask_pool!
                            
                            if wrong_mask is not None and wrong_mask.sum() > 0:
                                neg_sample = create_verifier_sample(
                                    image_path=image_path,  # Original image
                                    query=query,  # Original query
                                    mask=wrong_mask,  # Mask from different image
                                    label="REJECT",
                                    sample_id=f"{idx}_neg_easy_{neg_idx}",
                                    output_dir=output_dir,
                                )
                                all_samples.append(neg_sample)
                                stats["reject"] += 1
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            stats["skipped"] += 1
            continue
    
    # Save dataset
    output_path = os.path.join(output_dir, "train.json")
    print(f"\nSaving {len(all_samples)} samples to: {output_path}")
    
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    
    # Print stats
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {len(all_samples)}")
    print(f"  ACCEPT: {stats['accept']}")
    print(f"  REJECT: {stats['reject']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Ratio: 1:{stats['reject']/max(stats['accept'],1):.1f}")
    
    # Show sample
    if all_samples:
        print(f"\nSample entry:")
        sample = all_samples[0]
        print(json.dumps({
            "id": sample["id"],
            "label": sample["label"],
            "query": sample["query"],
            "images": sample["images"],
            "messages": [
                {"role": m["role"], "content": str(m["content"])[:100] + "..."} 
                for m in sample["messages"]
            ]
        }, indent=2))


if __name__ == "__main__":
    main()

