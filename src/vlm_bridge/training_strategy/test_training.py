"""
Real integration tests for training strategy module.

Note: These are real tests, no mock/fake/simulated/dummy data.
Tests use real models and data but at small scale for fast execution.
"""

import os
import tempfile
from pathlib import Path

import torch
import pytest

from .training_setup import TrainingConfig, prepare_environment
from .training_orchestrator import (
    execute_full_training,
    save_checkpoint,
    load_checkpoint,
)
from .core_training_loop import run_training_epoch, run_validation_epoch


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test directory structure
        data_dir = temp_path / "test_data"
        log_dir = temp_path / "test_logs"
        checkpoint_dir = temp_path / "test_checkpoints"

        yield {
            "data_dir": str(data_dir),
            "log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "temp_dir": str(temp_path),
        }


@pytest.fixture
def small_training_config(temp_dirs):
    """Create small training config for testing."""
    # Use real data directory but only train for minimal steps
    real_data_dir = "data/groundcap"

    # Skip test if data doesn't exist
    if not os.path.exists(real_data_dir):
        pytest.skip(f"Test requires real data: {real_data_dir}")

    return TrainingConfig(
        data_dir=real_data_dir,
        batch_size=2,  # Small batch size
        num_epochs=2,  # Only 2 epochs
        learning_rate=5e-5,
        log_dir=temp_dirs["log_dir"],
        checkpoint_dir=temp_dirs["checkpoint_dir"],
        log_every_n_steps=1,
        val_every_n_epochs=1,
        save_every_n_epochs=1,
        num_workers=0,  # Avoid multiprocessing issues
        device="cpu",  # Use CPU for test stability
    )


def test_prepare_environment(small_training_config):
    """Test training environment preparation."""
    context = prepare_environment(small_training_config)

    # Verify all context components
    assert context.config == small_training_config
    assert context.model is not None
    assert context.optimizer is not None
    assert context.train_loader is not None
    assert context.val_loader is not None
    assert context.device == torch.device("cpu")
    assert context.writer is not None
    assert context.checkpoint_dir.exists()
    assert context.start_epoch == 0
    assert context.best_val_loss == float("inf")

    # Verify model parameters
    trainable_params = sum(
        p.numel() for p in context.model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {trainable_params:,}")
    assert trainable_params > 0  # Should have trainable parameters

    # Cleanup
    context.writer.close()


def test_training_epoch(small_training_config):
    """Test single training epoch."""
    context = prepare_environment(small_training_config)

    # Only test a few batches
    max_batches = 3
    len(context.train_loader)

    # Create loader with limited batches
    limited_batches = []
    for i, batch in enumerate(context.train_loader):
        if i >= max_batches:
            break
        limited_batches.append(batch)

    # Temporarily replace train_loader
    class LimitedLoader:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    context.train_loader = LimitedLoader(limited_batches)

    # Execute training epoch
    train_loss = run_training_epoch(context, epoch=0)

    # Verify results
    assert isinstance(train_loss, float)
    assert train_loss > 0  # Loss should be positive
    assert train_loss < 20  # Loss shouldn't be too large

    print(f"Training loss: {train_loss:.4f}")

    # Cleanup
    context.writer.close()


def test_validation_epoch(small_training_config):
    """Test single validation epoch."""
    context = prepare_environment(small_training_config)

    # Only test a few batches
    max_batches = 3
    limited_batches = []
    for i, batch in enumerate(context.val_loader):
        if i >= max_batches:
            break
        limited_batches.append(batch)

    # Temporarily replace val_loader
    class LimitedLoader:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    context.val_loader = LimitedLoader(limited_batches)

    # Execute validation epoch
    val_loss, perplexity = run_validation_epoch(context, epoch=0)

    # Verify results
    assert isinstance(val_loss, float)
    assert isinstance(perplexity, float)
    assert val_loss > 0
    assert perplexity > 0

    print(f"Validation loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")

    # Cleanup
    context.writer.close()


def test_checkpoint_save_load(small_training_config):
    """Test checkpoint saving and loading."""
    context = prepare_environment(small_training_config)

    # Modify some state
    context.best_val_loss = 3.14
    epoch = 5

    # Save checkpoint
    save_checkpoint(context, epoch, is_best=True)

    # Verify files exist
    latest_path = context.checkpoint_dir / "latest_checkpoint.pth"
    best_path = context.checkpoint_dir / "best_model.pth"
    weights_only_path = context.checkpoint_dir / "best_model_weights_only.pth"

    assert latest_path.exists()
    assert best_path.exists()
    assert weights_only_path.exists()

    # Create new context and load checkpoint
    new_context = prepare_environment(small_training_config)
    load_checkpoint(new_context, str(latest_path))

    # Verify state restored
    assert new_context.start_epoch == epoch + 1
    assert new_context.best_val_loss == 3.14

    # Cleanup
    context.writer.close()
    new_context.writer.close()


def test_mini_training_run(small_training_config):
    """Test complete mini training run."""
    # Modify config for ultra-mini training
    small_training_config.num_epochs = 1

    # Execute training
    execute_full_training(small_training_config)

    # Verify output files
    checkpoint_dir = Path(small_training_config.checkpoint_dir)
    assert (checkpoint_dir / "latest_checkpoint.pth").exists()
    assert (checkpoint_dir / "best_model.pth").exists()

    print("Mini training run test passed!")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
