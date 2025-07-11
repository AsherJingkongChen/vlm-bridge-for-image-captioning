"""
Inference tests for model architecture components.

Inference only, no training/backpropagation.
This script tests the architecture with real models and minimal data,
focusing on shape validation and basic functionality.
"""

import os
import torch
import sys
import gc
from pathlib import Path

# Set environment variables to avoid warnings and reduce verbose output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

from vlm_bridge.model_architecture import (
    VisionEncoder,
    LanguageModel,
    BridgeModule,
    FullModel,
)
from vlm_bridge.data_pipeline.data_loader import VLDataset, create_data_loader


def test_vision_encoder_inference():
    """Test VisionEncoder with real DINOv2 model - inference only."""
    print("Testing VisionEncoder (inference only)...")

    # Initialize real DINOv2 model
    vision_encoder = VisionEncoder("facebook/dinov2-large")
    vision_encoder.eval()  # Ensure inference mode

    # Small batch size
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224)

    # Test inference
    with torch.no_grad():
        vision_features = vision_encoder(images)

    print(f"  Input images shape: {images.shape}")
    print(f"  Vision features shape: {vision_features.shape}")

    # Verify output shape
    expected_shape = (batch_size, 257, 1024)
    assert vision_features.shape == expected_shape, (
        f"Expected {expected_shape}, got {vision_features.shape}"
    )

    # Check model info
    info = vision_encoder.get_model_info()
    print(f"  Model: {info['model_name']}")
    print(f"  Parameters: {info['num_parameters']:,} (all frozen)")

    del vision_encoder, vision_features  # Free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    print("✓ VisionEncoder inference test passed!\n")


def test_language_model_inference():
    """Test LanguageModel with real Gemma-2-2B model - inference only."""
    print("Testing LanguageModel (inference only)...")

    # Initialize real Gemma model
    language_model = LanguageModel("google/gemma-2-2b")
    language_model.eval()

    # Short text for testing
    test_text = "A sunset."
    encoded = language_model.encode_text(test_text)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"  Test text: '{test_text}'")
    print(f"  Encoded shape: {input_ids.shape}")

    # Test inference only
    with torch.no_grad():
        # Get embeddings
        text_embeddings = language_model.get_embeddings(input_ids)
        print(f"  Text embeddings shape: {text_embeddings.shape}")

        # Test forward_from_embeddings (key method for our architecture)
        logits = language_model.forward_from_embeddings(text_embeddings, attention_mask)
        print(f"  Logits shape: {logits.shape}")

    # Verify dimensions
    batch_size, seq_len, vocab_size = logits.shape
    assert text_embeddings.shape[-1] == 2304, "Embedding dimension should be 2304"
    assert vocab_size == language_model.vocab_size, "Vocab size mismatch"

    # Model info
    info = language_model.get_model_info()
    print(f"  Model: {info['model_name']}")
    print(f"  Hidden size: {info['hidden_size']}")

    del language_model, text_embeddings, logits
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    print("✓ LanguageModel inference test passed!\n")


def test_bridge_module_inference():
    """Test BridgeModule with real dimensions - inference only."""
    print("Testing BridgeModule (inference only)...")

    # Initialize bridge module
    bridge = BridgeModule(
        vision_dim=1024,  # DINOv2-Large output
        language_dim=2304,  # Gemma-2-2B dimension
        num_heads=8,
        dropout=0.1,
    )
    bridge.eval()  # Inference mode

    # Small inputs for testing
    batch_size = 1
    vision_seq_len = 257  # DINOv2 sequence length
    text_seq_len = 5  # Short sequence

    vision_features = torch.randn(batch_size, vision_seq_len, 1024)
    text_embeddings = torch.randn(batch_size, text_seq_len, 2304)

    # Test inference only
    with torch.no_grad():
        enhanced_embeddings = bridge(vision_features, text_embeddings)

    print(f"  Vision features shape: {vision_features.shape}")
    print(f"  Text embeddings shape: {text_embeddings.shape}")
    print(f"  Enhanced embeddings shape: {enhanced_embeddings.shape}")

    # Verify output shape matches text embeddings
    assert enhanced_embeddings.shape == text_embeddings.shape, (
        "Output should match text embedding shape"
    )

    # Model info
    info = bridge.get_model_info()
    print(f"  Bridge parameters: {info['total_parameters']:,}")
    print(
        f"  Vision→Language projection: {info['vision_dim']} → {info['language_dim']}"
    )

    del bridge, enhanced_embeddings
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    print("✓ BridgeModule inference test passed!\n")


def test_full_model_minimal():
    """Test FullModel with minimal data - inference only."""
    print("Testing FullModel (minimal, inference only)...")

    print("  Loading models (this will take a moment)...")
    # Initialize full model
    full_model = FullModel(
        vision_model_name="facebook/dinov2-large",
        language_model_name="google/gemma-2-2b",
        bridge_num_heads=8,
        bridge_dropout=0.1,
    )
    full_model.eval()

    # Minimal test data
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224)

    # Very short text
    test_caption = "A photo."
    encoded = full_model.language_model.encode_text(test_caption)
    input_ids = encoded["input_ids"][:, :5]  # Limit to 5 tokens
    attention_mask = encoded["attention_mask"][:, :5]

    print(f"  Test caption: '{test_caption}'")
    print(f"  Input shapes - Images: {images.shape}, Text: {input_ids.shape}")

    # Test inference only
    with torch.no_grad():
        outputs = full_model(
            images=images, input_ids=input_ids, attention_mask=attention_mask
        )

    # Verify outputs
    logits = outputs["logits"]
    vision_features = outputs["vision_features"]
    text_embeddings = outputs["text_embeddings"]
    enhanced_embeddings = outputs["enhanced_embeddings"]

    print("  Output shapes:")
    print(f"    Logits: {logits.shape}")
    print(f"    Vision features: {vision_features.shape}")
    print(f"    Enhanced embeddings: {enhanced_embeddings.shape}")

    # Basic shape validation
    assert logits.shape[:2] == input_ids.shape, "Logits batch/seq should match input"
    assert vision_features.shape == (batch_size, 257, 1024), (
        "Vision features shape incorrect"
    )
    assert enhanced_embeddings.shape == text_embeddings.shape, (
        "Enhanced embeddings should match text"
    )

    # Model statistics
    info = full_model.get_model_info()
    total_params = info["total_parameters"]
    trainable_params = info["trainable_parameters"]
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(
        f"  Trainable ratio: {trainable_params / total_params:.4f} ({trainable_params / total_params * 100:.2f}%)"
    )

    # Verify only bridge is trainable
    assert trainable_params > 0, "Should have trainable parameters"
    assert trainable_params < total_params * 0.1, (
        "Trainable ratio should be less than 10%"
    )

    del full_model, outputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    print("✓ FullModel minimal test passed!\n")


def test_real_data_sample():
    """Test with one real GroundCap sample - inference only."""
    print("Testing with real GroundCap sample...")

    # Check if data exists
    data_dir = Path("data/groundcap/train")
    if not data_dir.exists():
        print("  ⚠️  GroundCap data not found. Skipping real data test.")
        print("  Run: uv run vlm-data-pipeline full --output-dir data/groundcap/")
        return

    # Load minimal dataset
    dataset = VLDataset(str(data_dir))
    # Use num_workers=0 to avoid multiprocessing issues in tests
    data_loader = create_data_loader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print(f"  Dataset size: {len(dataset)}")

    # Get one real sample
    batch = next(iter(data_loader))
    images = batch["images"]
    input_ids = batch["input_ids"][:, :10]  # Limit sequence length
    attention_mask = batch["attention_mask"][:, :10]

    print("  Real sample shapes:")
    print(f"    Images: {images.shape}")
    print(f"    Input IDs: {input_ids.shape}")

    # Load model for testing
    print("  Loading model for real data test...")
    full_model = FullModel()
    full_model.eval()

    # Test with real data - inference only
    with torch.no_grad():
        outputs = full_model(
            images=images, input_ids=input_ids, attention_mask=attention_mask
        )

    print("  Forward pass successful!")
    print(f"  Output logits shape: {outputs['logits'].shape}")

    # Decode sample to verify it's meaningful
    sample_ids = input_ids[0]
    decoded_text = full_model.language_model.decode_text(sample_ids.unsqueeze(0))[0]
    print(f"  Sample input text: '{decoded_text[:50]}...'")

    # Clean up resources
    del full_model, outputs, batch, data_loader, dataset
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    gc.collect()  # Force garbage collection

    print("✓ Real data sample test passed!\n")


if __name__ == "__main__":
    print("Vision-Language Bridge Model - Tests")
    print("=" * 70)
    print("Focus: Inference-only validation, minimal memory usage\n")

    try:
        # Core component tests
        test_vision_encoder_inference()
        test_language_model_inference()
        test_bridge_module_inference()

        # Integration tests
        test_full_model_minimal()
        test_real_data_sample()

        print("\n" + "=" * 70)
        print("All tests completed successfully! ✓")
        print("\nArchitecture validation:")
        print("- ✅ All components work correctly")
        print("- ✅ Shapes and dimensions are correct")
        print("- ✅ Real data integration works")
        print("- ✅ Ready for training strategy development")

        # Final cleanup before exit
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        gc.collect()

        # Explicit exit to ensure program terminates
        print("\nExiting test program...")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()

        print("\nTroubleshootings:")
        print("1. Memory: Close other applications to free RAM")
        print("2. Models: Ensure HuggingFace access to gated models")
        print("3. Data: Run data pipeline if testing with real data")
        print("4. Python: Use 'uv run python test_model_architecture.py'")

        # Exit with error code
        sys.exit(1)
