"""
Command-line interface for VLM Bridge training.
"""

import argparse

from .training_setup import TrainingConfig
from .training_orchestrator import execute_full_training


def main():
    """Main function: parse command-line arguments and execute training."""
    parser = argparse.ArgumentParser(
        description="Vision-Language Bridge Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new training with default config
  uv run vlm-training
  
  # Start training with custom config
  uv run vlm-training --config config/training-default.yaml
  
  # Resume training from checkpoint
  uv run vlm-training --resume checkpoints/latest_checkpoint.pth
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training-default.yaml",
        help="Training configuration file path (default: config/training-default.yaml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume training from checkpoint path"
    )

    args = parser.parse_args()

    # Load or create config file
    config_path = args.config
    config = TrainingConfig.from_yaml(config_path)

    # If config file didn't exist, save the default config for future reference
    from pathlib import Path

    if not Path(config_path).exists():
        print(f"[Config] Created default configuration at: {config_path}")
        config.to_yaml(config_path)

    # Override resume path if provided
    if args.resume:
        config.resume_from_checkpoint = args.resume

    # Execute training (validation is built-in)
    execute_full_training(config)


if __name__ == "__main__":
    main()
