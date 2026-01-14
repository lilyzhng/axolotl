#!/usr/bin/env python3
"""
Preprocess MaskGroups-HQ dataset to work with axolotl.

This script:
1. Downloads images from HuggingFace (images_resized.zip)
2. Converts image filenames to full local paths
3. Converts 'conversations' format to 'messages' format (OpenAI style)
4. Wraps image path in a list (axolotl expects list of images)
5. Saves as JSON for axolotl training

Usage:
    python scripts/custom/preprocess_maskgroups.py --images-dir ./data/maskgroups-hq/images_resized --output-dir ./data/maskgroups-hq
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "Shengcao1006/MaskGroups-HQ"


def convert_conversations_to_messages(conversations):
    """Convert ShareGPT/Vicuna format to OpenAI messages format."""
    messages = []
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
    }
    for conv in conversations:
        role = role_map.get(conv["from"], conv["from"])
        messages.append({
            "role": role,
            "content": conv["value"]
        })
    return messages


def preprocess_sample(sample, images_dir):
    """Preprocess a single sample."""
    # Get full image path (use absolute path for reliability)
    image_filename = sample["image"]
    image_path = os.path.abspath(os.path.join(images_dir, image_filename))
    
    # Convert conversations to messages format
    messages = convert_conversations_to_messages(sample["conversations"])
    
    return {
        "id": sample["id"],
        "messages": messages,
        "images": [image_path],  # Wrap in list as axolotl expects
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess MaskGroups-HQ dataset for axolotl")
    parser.add_argument(
        "--images-dir", 
        type=str, 
        default="./data/maskgroups-hq/images_resized",
        help="Directory containing the extracted images"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./data/maskgroups-hq",
        help="Directory to save the preprocessed JSON"
    )
    args = parser.parse_args()
    
    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="validation")
    
    print(f"Processing {len(ds)} samples...")
    print(f"Images directory: {images_dir}")
    
    processed_samples = []
    missing_images = []
    
    for sample in tqdm(ds):
        processed = preprocess_sample(sample, images_dir)
        
        # Check if image exists
        if not os.path.exists(processed["images"][0]):
            missing_images.append(processed["images"][0])
        
        processed_samples.append(processed)
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images not found!")
        print("First 5 missing:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        print(f"\nMake sure you've extracted images_resized.zip to: {images_dir}")
        print("\nDownload from: https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ/blob/main/images_resized.zip")
    
    # Save to JSON
    output_path = os.path.join(output_dir, "train.json")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(processed_samples, f, indent=2)
    
    print(f"Done! Saved {len(processed_samples)} samples.")
    print(f"\nSample entry:")
    print(json.dumps(processed_samples[0], indent=2))


if __name__ == "__main__":
    main()
