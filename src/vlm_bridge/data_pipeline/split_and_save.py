"""
Dataset Split and Save Module

This module implements the data_pipeline::split_and_save functionality
as defined in the RED-F project tree. It splits the transformed dataset
into train/val/test splits and organizes files into the final directory structure.
"""

import json
from pathlib import Path
from typing import Dict
from datasets import Dataset


def split_and_organize_files(
    transformed_dataset: Dataset, base_output_dir: str
) -> None:
    """
    Generate captions.jsonl files for the already-split dataset.

    Assumes images are already in the correct directory structure:
    base_output_dir/
    ├── train/
    │   ├── images/
    │   └── captions.jsonl (created by this function)
    ├── val/
    │   ├── images/
    │   └── captions.jsonl (created by this function)
    └── test/
        ├── images/
        └── captions.jsonl (created by this function)

    Args:
        transformed_dataset: Dataset with image_path, caption, and split_assignment fields
        base_output_dir: Base directory for output (e.g., "data/groundcap/")
    """
    print(f"Generating captions.jsonl files for {len(transformed_dataset)} samples...")

    # Create base output directory
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Group samples by split
    split_data = {"train": [], "val": [], "test": []}

    for sample in transformed_dataset:
        try:
            split_assignments = sample.get("split_assignment", [])
            if not split_assignments:
                print(
                    f"Warning: Sample {sample.get('original_id', 'unknown')} has no split assignment"
                )
                continue

            for split_name in split_assignments:
                if split_name in split_data:
                    split_data[split_name].append(sample)
                else:
                    print(
                        f"Warning: Unknown split name '{split_name}' for sample {sample.get('original_id', 'unknown')}"
                    )

        except Exception as e:
            print(f"Error processing sample in split organization: {e}")
            continue

    print(
        f"Split sizes: train={len(split_data['train'])}, val={len(split_data['val'])}, test={len(split_data['test'])}"
    )

    # Generate jsonl files for each split
    for split_name, samples in split_data.items():
        if samples:
            print(f"Generating {split_name} captions.jsonl...")
            _generate_captions_jsonl(samples, base_path, split_name)

    print("✅ All captions.jsonl files generated successfully!")
    print(f"Output directory: {base_output_dir}")


def _generate_captions_jsonl(samples: list, base_path: Path, split_name: str) -> None:
    """
    Generate captions.jsonl file for a single split.

    Args:
        samples: List of samples for this split
        base_path: Base output directory
        split_name: Name of the split (train/val/test)
    """
    split_dir = base_path / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Prepare captions data
    captions_data = []

    for sample in samples:
        # Get image path and make it relative to split directory
        image_path = Path(sample["image_path"])
        relative_path = f"images/{image_path.name}"

        # Prepare caption data
        caption_entry = {
            "image_path": relative_path,
            "caption": sample["caption"],
            "original_id": sample["original_id"],
        }
        captions_data.append(caption_entry)

    # Save captions as JSONL
    captions_file = split_dir / "captions.jsonl"
    with open(captions_file, "w", encoding="utf-8") as f:
        for entry in captions_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  ✅ {split_name} captions.jsonl generated: {len(captions_data)} samples")


def get_split_stats(base_output_dir: str) -> Dict[str, any]:
    """
    Get statistics about the organized splits.

    Args:
        base_output_dir: Base directory containing the splits

    Returns:
        Dictionary with split statistics
    """
    base_path = Path(base_output_dir)
    stats = {}

    for split_name in ["train", "val", "test"]:
        split_dir = base_path / split_name
        images_dir = split_dir / "images"
        captions_file = split_dir / "captions.jsonl"

        if images_dir.exists() and captions_file.exists():
            # Count images
            image_count = len(list(images_dir.glob("*.jpg")))

            # Count captions
            caption_count = 0
            with open(captions_file, "r", encoding="utf-8") as f:
                caption_count = sum(1 for _ in f)

            stats[split_name] = {
                "image_count": image_count,
                "caption_count": caption_count,
                "directory": str(split_dir),
                "images_dir": str(images_dir),
                "captions_file": str(captions_file),
            }
        else:
            stats[split_name] = {"error": "Split not found"}

    return stats
