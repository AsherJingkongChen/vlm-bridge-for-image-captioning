"""
GroundCap Dataset Loading and Exploration Module

This module implements the data_pipeline::load_and_explore functionality
as defined in the RED-F project tree. It handles loading the GroundCap dataset
and provides focused dataset structure analysis for transform_full_dataset preparation.
"""

import re
from datasets import load_dataset, DatasetDict


def load_ground_cap() -> DatasetDict:
    """
    Load the GroundCap dataset from HuggingFace.

    Returns:
        DatasetDict: The loaded GroundCap dataset with train/test splits

    Raises:
        RuntimeError: If dataset loading fails
    """
    try:
        print("Loading GroundCap dataset from HuggingFace...")
        dataset = load_dataset("daniel3303/groundcap")

        print("Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")

        # Basic validation
        if not dataset:
            raise RuntimeError("Dataset is empty")

        for split_name, split_data in dataset.items():
            if len(split_data) == 0:
                raise RuntimeError(f"Split '{split_name}' is empty")

        return dataset

    except Exception as e:
        raise RuntimeError(f"Failed to load GroundCap dataset: {str(e)}")


def inspect_dataset_structure(dataset: DatasetDict) -> None:
    """
    Inspect GroundCap dataset structure for transform_full_dataset preparation.

    Focus: Understand data format to enable (image_path, caption) transformation.

    Args:
        dataset: The loaded GroundCap dataset
    """
    print("🔍 Loading and exploring GroundCap dataset...")
    
    print("\n" + "=" * 60)
    print("GROUNDCAP DATASET STRUCTURE ANALYSIS")
    print("=" * 60)

    # 1. Basic Dataset Information
    print("\n1. DATASET SPLITS")
    print("-" * 30)

    total_samples = 0
    for split_name, split_data in dataset.items():
        split_size = len(split_data)
        total_samples += split_size
        print(f"  {split_name}: {split_size:,} samples")

    print(f"  TOTAL: {total_samples:,} samples")

    # 2. Data Structure for Transform Preparation
    print("\n2. DATA STRUCTURE FOR TRANSFORM")
    print("-" * 30)

    sample = dataset["train"][0]

    # Image analysis
    print(f"  Image: {sample['image'].size} {sample['image'].mode}")

    # Caption analysis for text extraction
    caption = sample["caption"]
    print(f"  Caption length: {len(caption)} chars")

    # Check if caption contains grounding tags
    has_tags = any(tag in caption for tag in ["<gdo", "<gda", "<gdl"])
    print(f"  Contains grounding tags: {has_tags}")

    if has_tags:
        # Show how to extract clean text
        clean_caption = re.sub(r"<[^>]+>", "", caption)
        print(f"  Clean caption length: {len(clean_caption)} chars")
        print(f"  Clean text: {clean_caption[:240]}...")

    # 3. Transform Requirements Summary
    print("\n3. TRANSFORM REQUIREMENTS")
    print("-" * 30)
    print("  ✓ Images: PIL format, ready for saving")
    print("  ✓ Captions: Text extraction needed (remove HTML tags)")
    print(f"  ✓ Target format: (image_path, caption) for {total_samples:,} samples")

    print("\n" + "=" * 60)
    print("READY FOR TRANSFORM_FULL_DATASET")
    print("=" * 60)
    
    print("\n✅ Dataset exploration complete!")
