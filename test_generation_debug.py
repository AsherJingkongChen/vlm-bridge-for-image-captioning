"""
Quick generation debugging script to test fixes immediately.

This script provides a fast way to test the generation fixes without
running the full training pipeline.
"""

import torch
import os
from pathlib import Path

# Set environment to reduce noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

from vlm_bridge.model_architecture import FullModel
from vlm_bridge.data_pipeline.data_loader import VLDataset


def test_generation_fixes():
    """Test all generation fixes quickly."""
    print("🚀 Testing Generation Fixes")
    print("=" * 50)

    # Load model
    print("📦 Loading model...")
    model = FullModel()

    # Try to load a trained checkpoint if available
    checkpoint_path = "checkpoints/experiment/best_model_weights_only.pth"
    if Path(checkpoint_path).exists():
        print(f"📁 Loading trained model: {checkpoint_path}")
        model.load_model(checkpoint_path)
    else:
        print("⚠️  Using untrained model")

    model.eval()

    # Load test data
    data_dir = Path("data/groundcap/train")
    if not data_dir.exists():
        print("❌ No test data found. Please run data pipeline first.")
        return

    dataset = VLDataset(str(data_dir))
    sample = dataset[0]
    image = sample["images"]

    print(f"🖼️  Testing with sample: {sample.get('image_path', 'unknown')}")

    # Test 1: Detailed debugging generation
    print("\n" + "=" * 50)
    print("Test 1: Detailed Debug Generation")
    print("=" * 50)

    with torch.no_grad():
        caption = model.generate_caption(
            image,
            max_length=15,
            do_sample=False,  # Greedy for consistency
            debug=True,  # Enable detailed debugging
        )

    print(f"\n✅ Generated caption: '{caption}'")
    print(f"📏 Length: {len(caption.split())} words")

    # Test 2: Multiple strategies
    print("\n" + "=" * 50)
    print("Test 2: Multiple Generation Strategies")
    print("=" * 50)

    strategies = [
        {"name": "greedy", "do_sample": False},
        {"name": "low_temp", "temperature": 0.1, "do_sample": True, "top_p": 1.0},
        {"name": "med_temp", "temperature": 0.7, "do_sample": True, "top_p": 0.9},
        {"name": "high_temp", "temperature": 1.0, "do_sample": True, "top_p": 0.8},
        {"name": "no_top_p", "temperature": 0.7, "do_sample": True, "top_p": 1.0},
    ]

    results = model.generate_caption_robust(image, max_length=20, strategies=strategies)

    print("Strategy Results:")
    for strategy_name, result_caption in results.items():
        length = (
            len(result_caption.split())
            if result_caption and not result_caption.startswith("ERROR")
            else 0
        )
        status = "✅" if length > 0 else "❌"
        print(f"  {status} {strategy_name:10s}: {length:2d} words - '{result_caption}'")

    # Test 3: Check if model learned anything
    print("\n" + "=" * 50)
    print("Test 3: Training vs Untrained Comparison")
    print("=" * 50)

    # Test multiple samples to see consistency
    print("Testing on 3 different images:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        test_image = sample["images"]

        with torch.no_grad():
            greedy_caption = model.generate_caption(
                test_image, max_length=15, do_sample=False
            )

        print(
            f"  Image {i + 1}: '{greedy_caption}' ({len(greedy_caption.split())} words)"
        )

    # Summary
    print("\n" + "=" * 50)
    print("🎯 SUMMARY")
    print("=" * 50)

    successful_strategies = [
        name
        for name, caption in results.items()
        if caption and not caption.startswith("ERROR") and len(caption.split()) > 0
    ]

    print(f"✅ Successful strategies: {len(successful_strategies)}/{len(strategies)}")
    print(f"   Working: {successful_strategies}")

    if not successful_strategies:
        print("❌ CRITICAL: No generation strategy works!")
        print("   Recommendations:")
        print("   1. Check if EOS token is triggered immediately")
        print("   2. Verify model weights loaded correctly")
        print("   3. Check for NaN/Inf in Bridge module")
        print("   4. Run full debug_generation.py for detailed analysis")
    else:
        print("✅ Some strategies work - generation pipeline is functional!")
        if len(successful_strategies) < len(strategies):
            print("⚠️  Some strategies still fail - may need fine-tuning")

    print("\n📝 To investigate further, run: uv run python debug_generation.py")


if __name__ == "__main__":
    test_generation_fixes()
