"""
Data Loader for Vision-Language Bridge

This module provides PyTorch Dataset and DataLoader implementations for loading
image-caption pairs from our standardized GroundCap data structure.

Core Components:
- VLDataset: PyTorch Dataset for loading images and captions
- Preprocessing pipelines for DINOv2 (images) and Gemma 2 2B (text)
- DataLoader factory functions with proper batching and padding

Usage:
    from vlm_bridge.data_pipeline.data_loader import VLDataset, create_data_loader

    dataset = VLDataset("data/groundcap/train/")
    loader = create_data_loader(dataset, batch_size=8)

    for batch in loader:
        images = batch['images']      # [batch_size, 3, 224, 224]
        input_ids = batch['input_ids']  # [batch_size, seq_len]
        attention_mask = batch['attention_mask']  # [batch_size, seq_len]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer


class VLDataset(Dataset):
    """
    PyTorch Dataset for Vision-Language pairs from GroundCap structure.

    Expected directory structure:
    data_dir/
    ├── images/
    │   ├── image_0001.jpg
    │   ├── image_0002.jpg
    │   └── ...
    └── captions.jsonl

    Each line in captions.jsonl:
    {"image_path": "images/image_0001.jpg", "caption": "A description of the image"}
    """

    def __init__(
        self,
        data_dir: str,
        image_processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to data directory (e.g., "data/groundcap/train/")
            image_processor: DINOv2 image processor (auto-loaded if None)
            tokenizer: Gemma 2 2B tokenizer (auto-loaded if None)
            max_length: Maximum sequence length for text tokenization
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length

        # Load preprocessors
        self.image_processor = image_processor or AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("google/gemma-2-2b")

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load captions data
        self.captions_file = self.data_dir / "captions.jsonl"
        self.data_samples = self._load_captions()

    def _load_captions(self) -> List[Dict[str, str]]:
        """Load captions from JSONL file."""
        if not self.captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {self.captions_file}")

        samples = []
        with open(self.captions_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    # Validate required fields
                    if "image_path" not in sample or "caption" not in sample:
                        continue  # Skip silently
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue  # Skip silently

        return samples

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict containing:
            - 'image': PIL Image
            - 'caption': str
            - 'image_path': str (for debugging)
        """
        sample = self.data_samples[idx]

        # Load image
        image_path = self.data_dir / sample["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        return {
            "image": image,
            "caption": sample["caption"],
            "image_path": str(image_path),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Dict containing batched tensors:
            - 'images': [batch_size, 3, 224, 224]
            - 'input_ids': [batch_size, max_seq_len]
            - 'attention_mask': [batch_size, max_seq_len]
        """
        # Extract images and captions
        images = [sample["image"] for sample in batch]
        captions = [sample["caption"] for sample in batch]

        # Process images for DINOv2
        image_inputs = self.image_processor(images, return_tensors="pt")

        # Process captions for Gemma 2 2B
        # Note: Gemma tokenizer automatically adds BOS token when add_special_tokens=True (default)
        text_inputs = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": image_inputs["pixel_values"],  # [batch_size, 3, 224, 224]
            "input_ids": text_inputs["input_ids"],  # [batch_size, seq_len]
            "attention_mask": text_inputs["attention_mask"],  # [batch_size, seq_len]
        }


def create_data_loader(
    dataset: VLDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with proper configuration for vision-language training.

    Args:
        dataset: VLDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        PyTorch DataLoader
    """
    # Disable pin_memory on MPS (Apple Silicon) as it's not supported
    if torch.backends.mps.is_available() and pin_memory:
        pin_memory = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
    )


def get_data_loaders(
    base_data_dir: str = "data/groundcap/",
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train/val/test data loaders.

    Args:
        base_data_dir: Base directory containing train/val/test subdirs
        batch_size: Batch size for all loaders
        max_length: Maximum sequence length for tokenization
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    base_path = Path(base_data_dir)

    # Load shared preprocessors once
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    # Create datasets
    train_dataset = VLDataset(
        str(base_path / "train"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = VLDataset(
        str(base_path / "val"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = VLDataset(
        str(base_path / "test"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = create_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def inspect_data_loader(base_output_dir: str = "data/groundcap/") -> None:
    """
    Inspect the data loader functionality and visualize batch format.

    Args:
        base_output_dir: Base directory containing train/val/test subdirs
    """
    print("\n🔍 Inspecting Vision-Language Data Loader...")
    print("=" * 70)

    base_path = Path(base_output_dir)

    # Test each split
    for split_name in ["train", "val", "test"]:
        split_dir = base_path / split_name

        if not split_dir.exists():
            print(f"\n❌ {split_name} directory not found at {split_dir}")
            continue

        print(f"\n📂 {split_name.upper()} SPLIT")
        print("-" * 70)

        try:
            # Create dataset
            dataset = VLDataset(str(split_dir))
            print(f"  Dataset size: {len(dataset):,} samples")

            # Test single sample and batch
            if len(dataset) > 0:
                # Single sample inspection
                sample = dataset[0]
                print("\n  📄 Single Sample:")
                print("    - Image: PIL.Image.Image")
                print(f'    - Caption: "{sample["caption"][:60]}..."')

                # Create batch
                loader = create_data_loader(dataset, batch_size=2, num_workers=0)
                batch = next(iter(loader))

                print("\n  📦 Batch Format (batch_size=2):")
                print("\n    🖼️  Images (for DINOv2 ViT input):")
                print(f"      - Shape: {batch['images'].shape}")
                print("      - Format: RGB normalized tensors")
                print("      - Ready for: facebook/dinov2-base (patch_size=14)")

                print("\n    💬 Text Tokens (for Gemma-2-2b LM input):")
                print(f"      - Input IDs shape: {batch['input_ids'].shape}")
                print(f"      - Attention mask shape: {batch['attention_mask'].shape}")
                print("      - Tokenizer: google/gemma-2-2b (vocab_size=256K)")
                print("      - Special tokens: BOS (2) auto-added, PAD (0) for padding")

                # Show example token decoding
                # Get non-padded tokens from first sample
                first_sample_tokens = batch["input_ids"][0]
                first_sample_mask = batch["attention_mask"][0]
                non_pad_length = first_sample_mask.sum().item()

                # Show first 10 non-padded tokens
                if non_pad_length > 0:
                    actual_tokens = first_sample_tokens[-non_pad_length:][:10].tolist()
                    print(f"      - First 10 tokens (non-padded): {actual_tokens}")
                else:
                    print(
                        f"      - First 10 tokens: {first_sample_tokens[:10].tolist()}"
                    )

        except Exception as e:
            print(f"  ❌ Error loading {split_name}: {e}")

    print("\n" + "=" * 70)
    print("✅ Data loader ready for Vision-Language Bridge training!")
