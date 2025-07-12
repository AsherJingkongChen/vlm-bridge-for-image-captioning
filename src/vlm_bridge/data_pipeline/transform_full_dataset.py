"""
Full Dataset Transformation Module

This module implements the data_pipeline::transform_full_dataset functionality
as defined in the RED-F project tree. It converts the entire GroundCap dataset
into a standardized (image_path, caption) format for training.
"""

import re
from pathlib import Path
from typing import Dict, Any
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed


def transform_and_save_images(dataset: Dataset, final_base_dir: str) -> Dataset:
    """
    Transform the entire GroundCap dataset into (image_path, caption) format.

    This function:
    1. Implements data split logic (train 80%, val 2% non-overlapping, test 18%)
    2. Saves images directly to final directory structure
    3. Extracts clean captions (removes HTML grounding tags)
    4. Returns a new Dataset with image_path and caption fields

    Args:
        dataset: Input GroundCap dataset (combined train+test)
        final_base_dir: Base directory for final structure (e.g., "data/groundcap/")

    Returns:
        Dataset: Transformed dataset with image_path and caption fields
    """
    print(f"Transforming and splitting {len(dataset)} samples...")

    # Create final directory structure
    final_base_path = Path(final_base_dir)
    train_images_dir = final_base_path / "train" / "images"
    val_images_dir = final_base_path / "val" / "images"
    test_images_dir = final_base_path / "test" / "images"

    # Create directories
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)

    # Calculate split indices for non-overlapping splits
    total_size = len(dataset)
    train_end = int(0.8 * total_size)  # 80% for training
    val_start = train_end  # Validation starts where training ends
    val_end = int(0.82 * total_size)  # 2% for validation
    test_start = val_end  # Test starts where validation ends

    print(
        f"Split strategy: Train 0-{train_end}, Val {val_start}-{val_end}, Test {test_start}-{total_size}"
    )

    # Process samples in parallel with threading
    transformed_data = [None] * len(dataset)
    max_workers = 4
    print(f"  Using {max_workers} threads for parallel JPEG processing...")

    def process_sample(i, sample):
        try:
            # 1. Determine which split this sample belongs to (non-overlapping)
            split_dirs = []
            if i < train_end:
                split_dirs.append(("train", train_images_dir))
            elif val_start <= i < val_end:
                split_dirs.append(("val", val_images_dir))
            elif i >= test_start:
                split_dirs.append(("test", test_images_dir))

            if not split_dirs:
                raise ValueError(f"Sample {i} doesn't belong to any split")

            # 2. Save image to appropriate directory(ies)
            original_id = sample["id"]
            image_filename = f"{original_id}.jpg"

            # Save to the appropriate split directory (non-overlapping)
            split_name, split_dir = split_dirs[0]  # Only one split per sample now
            # Ensure directory exists
            split_dir.mkdir(parents=True, exist_ok=True)
            split_image_path = split_dir / image_filename
            if not split_image_path.exists():
                # Save directly to the split directory
                sample["image"].save(str(split_image_path), "JPEG", quality=95)

            # Set the image path
            primary_image_path = split_image_path

            # 3. Extract clean caption
            raw_caption = sample["caption"]
            clean_caption = _extract_clean_caption(raw_caption)

            # 4. Create transformed sample with primary path
            transformed_sample = {
                "image_path": str(primary_image_path),
                "caption": clean_caption,
                "original_id": sample["id"],
                "split_assignment": [split_name],  # Single split assignment (non-overlapping)
            }

            transformed_data[i] = transformed_sample

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Don't set transformed_data[i] to None, let it fail properly
            raise

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_sample, i, sample)
            for i, sample in enumerate(dataset)
        ]

        # Wait for completion
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions

    print(
        f"✅ Transformation and splitting complete! {len(transformed_data)} samples processed."
    )
    print(f"Images saved directly to final directory structure: {final_base_dir}")

    # Create new dataset from transformed data
    return Dataset.from_list(transformed_data)


def _extract_clean_caption(raw_caption: str) -> str:
    """
    Extract clean caption text by removing HTML grounding tags.

    Args:
        raw_caption: Original caption with HTML tags

    Returns:
        str: Clean caption text
    """
    # Remove all HTML tags (including grounding tags)
    clean_caption = re.sub(r"<[^>]+>", "", raw_caption)

    # Clean up extra whitespace
    clean_caption = re.sub(r"\s+", " ", clean_caption).strip()

    return clean_caption


def get_transform_stats(
    original_dataset: Dataset, transformed_dataset: Dataset
) -> Dict[str, Any]:
    """
    Generate statistics comparing original and transformed datasets.

    Args:
        original_dataset: Original GroundCap dataset
        transformed_dataset: Transformed dataset

    Returns:
        Dict containing transformation statistics
    """
    # Sample captions for comparison
    original_sample = original_dataset[0]["caption"]
    transformed_sample = transformed_dataset[0]["caption"]

    # Calculate average caption lengths
    original_lengths = [len(sample["caption"]) for sample in original_dataset]
    transformed_lengths = [len(sample["caption"]) for sample in transformed_dataset]

    stats = {
        "original_count": len(original_dataset),
        "transformed_count": len(transformed_dataset),
        "avg_original_caption_length": sum(original_lengths) / len(original_lengths),
        "avg_transformed_caption_length": sum(transformed_lengths)
        / len(transformed_lengths),
        "sample_original_caption": original_sample[:100] + "..."
        if len(original_sample) > 100
        else original_sample,
        "sample_transformed_caption": transformed_sample[:100] + "..."
        if len(transformed_sample) > 100
        else transformed_sample,
        "transformation_success": len(original_dataset) == len(transformed_dataset),
    }

    return stats
