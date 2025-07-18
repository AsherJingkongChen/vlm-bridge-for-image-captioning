#!/usr/bin/env python3
"""
Data Pipeline CLI

Command line interface for processing the full GroundCap dataset.
Handles loading, transforming, and splitting the entire dataset.
"""

import argparse
import sys
import time

from .load_and_explore import load_ground_cap, inspect_dataset_structure
from .transform_full_dataset import transform_and_save_images
from .split_and_save import split_and_organize_files


def main():
    """Main CLI function for data pipeline operations."""
    parser = argparse.ArgumentParser(description="VLM Bridge Data Pipeline CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Explore command
    subparsers.add_parser("explore-dataset", help="Load and explore dataset")

    # Transform command
    transform_parser = subparsers.add_parser("transform", help="Transform full dataset")
    transform_parser.add_argument(
        "--output-dir",
        default="data/groundcap/",
        help="Output directory for final structure (default: data/groundcap/)",
    )

    # Inspect loader command
    inspect_parser = subparsers.add_parser("inspect-loader", help="Inspect data loader")
    inspect_parser.add_argument(
        "--data-dir",
        default="data/groundcap/",
        help="Base directory containing train/val/test splits (default: data/groundcap/)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "explore-dataset":
            # Load and explore dataset
            dataset = load_ground_cap()
            inspect_dataset_structure(dataset)

        elif args.command == "transform":
            # Transform dataset - now saves directly to final structure
            print("🔄 Transforming dataset directly to final structure...")

            # Load and combine datasets efficiently
            dataset = load_ground_cap()

            # Use HuggingFace's concatenate_datasets instead of Python lists
            from datasets import concatenate_datasets

            print(f"Including TRAIN split ({len(dataset['train'])} samples)...")
            print(f"Including TEST split ({len(dataset['test'])} samples)...")

            # Efficiently combine datasets
            combined_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

            # Transform and save directly to final structure
            start_time = time.time()
            transformed_dataset = transform_and_save_images(
                combined_dataset, args.output_dir
            )

            # Generate captions.jsonl files
            split_and_organize_files(transformed_dataset, args.output_dir)

            elapsed = time.time() - start_time
            print(f"✅ Complete pipeline finished in {elapsed:.2f} seconds")

        elif args.command == "inspect-loader":
            # Inspect data loader
            from .data_loader import inspect_data_loader

            inspect_data_loader(args.data_dir)

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
