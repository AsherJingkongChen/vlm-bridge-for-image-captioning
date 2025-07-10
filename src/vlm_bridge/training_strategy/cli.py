"""
Command-line interface for VLM Bridge training.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch

from .training_setup import TrainingConfig
from .training_orchestrator import execute_full_training
from ..model_architecture import FullModel
from ..data_pipeline.data_loader import VLDataset, create_data_loader


def main():
    """Main function: parse command-line arguments and execute operations."""
    parser = argparse.ArgumentParser(
        description="Vision-Language Bridge Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new training
  uv run vlm-training train --data-dir data/groundcap --batch-size 8 --epochs 10
  
  # Resume training from checkpoint
  uv run vlm-training train --resume checkpoints/latest_checkpoint.pth
  
  # Validate model
  uv run vlm-training validate --checkpoint checkpoints/best_model.pth
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train Vision-Language Bridge model"
    )
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/groundcap",
        help="Dataset directory (default: data/groundcap)",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    train_parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    train_parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)",
    )
    train_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/experiment",
        help="TensorBoard log directory (default: logs/experiment)",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/experiment",
        help="Checkpoint save directory (default: checkpoints/experiment)",
    )
    train_parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )
    train_parser.add_argument(
        "--val-every-n-epochs",
        type=int,
        default=1,
        help="Validate every N epochs (default: 1)",
    )
    train_parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )
    train_parser.add_argument(
        "--no-amp", action="store_true", help="Disable mixed precision training"
    )
    train_parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Mixed precision type (default: bfloat16)",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device (default: auto-detect)",
    )
    train_parser.add_argument(
        "--resume", type=str, default=None, help="Resume training from checkpoint"
    )

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate trained model")
    val_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    val_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/groundcap",
        help="Dataset directory (default: data/groundcap)",
    )
    val_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    val_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (default: auto-detect)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        # Create training config
        config = TrainingConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            gradient_clip_val=args.gradient_clip,
            use_amp=not args.no_amp,
            amp_dtype=args.amp_dtype,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_every_n_steps=args.log_every_n_steps,
            val_every_n_epochs=args.val_every_n_epochs,
            save_every_n_epochs=args.save_every_n_epochs,
            num_workers=args.num_workers,
            device=args.device,
            resume_from_checkpoint=args.resume,
        )

        # Execute training
        execute_full_training(config)

    elif args.command == "validate":
        # Execute validation
        validate_model(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=args.device,
        )


def validate_model(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> None:
    """Validate trained model."""
    print("=" * 80)
    print("Starting Model Validation")
    print("=" * 80)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[Validation] Using device: {device}")
    print(f"[Validation] Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = FullModel(device=device)

    # Load BridgeModule weights
    model_state_dict = model.state_dict()
    for key, value in checkpoint["model_state_dict"].items():
        if key in model_state_dict:
            model_state_dict[key] = value
    model.load_state_dict(model_state_dict)
    model.eval()

    print("[Validation] Model loaded successfully")

    # Load validation data
    val_dataset = VLDataset(os.path.join(data_dir, "val"))
    val_loader = create_data_loader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"[Validation] Validation samples: {len(val_dataset)}")

    # Run validation
    from .core_training_loop import run_validation_epoch
    from .training_setup import TrainingContext, TrainingConfig
    from torch.utils.tensorboard import SummaryWriter

    # Create temporary context (just for reusing run_validation_epoch)
    temp_config = TrainingConfig(batch_size=batch_size, device=device)
    temp_writer = SummaryWriter(log_dir=None)  # No logging

    temp_context = TrainingContext(
        config=temp_config,
        model=model,
        optimizer=None,  # Not needed for validation
        train_loader=None,  # Not needed for validation
        val_loader=val_loader,
        device=torch.device(device),
        scaler=None,
        writer=temp_writer,
        checkpoint_dir=Path("."),
    )

    # Execute validation
    val_loss, perplexity = run_validation_epoch(temp_context, epoch=0)

    print("\n" + "=" * 80)
    print("Validation Complete")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("=" * 80)

    temp_writer.close()


if __name__ == "__main__":
    main()
