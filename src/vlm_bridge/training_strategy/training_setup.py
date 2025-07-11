"""
Training setup module.

This module handles the preparation of training environment including
hardware configuration, optimizer setup, and logging initialization.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from ..model_architecture import FullModel
from ..data_pipeline.data_loader import VLDataset, create_data_loader


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # Data related
    data_dir: str = "data/groundcap"
    batch_size: int = 8
    num_workers: int = 4

    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    gradient_clip_val: float = 1.0

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"

    # Logging and checkpoints
    log_dir: str = "logs/experiment"
    checkpoint_dir: str = "checkpoints/experiment"
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 1

    # Hardware
    device: Optional[str] = None  # None for auto-detect

    # Resume training
    resume_from_checkpoint: Optional[str] = None


@dataclass
class TrainingContext:
    """Context containing all training components."""

    config: TrainingConfig
    model: FullModel
    optimizer: AdamW
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    device: torch.device
    scaler: Optional[torch.amp.GradScaler]
    writer: SummaryWriter
    checkpoint_dir: Path
    start_epoch: int = 0
    best_val_loss: float = float("inf")


def prepare_environment(config: TrainingConfig) -> TrainingContext:
    """
    Prepare all components needed for training.

    This is the main coordinator that calls other setup functions
    and assembles the TrainingContext.
    """
    print("[Training Setup] Preparing training environment...")

    # 1. Configure hardware and mixed precision
    device, scaler = configure_hardware_and_precision(config)

    # 2. Load data
    print(f"[Training Setup] Loading dataset from: {config.data_dir}")
    train_dataset = VLDataset(os.path.join(config.data_dir, "train"))
    val_dataset = VLDataset(os.path.join(config.data_dir, "val"))

    train_loader = create_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = create_data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    print(
        f"[Training Setup] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
    )

    # 3. Create model
    print("[Training Setup] Creating model architecture...")
    model = FullModel(device=str(device))

    # 4. Create optimizer
    optimizer = create_optimizer(model, config)

    # 5. Setup logging and checkpoints
    writer, checkpoint_dir = setup_logging_and_checkpoints(config)

    # 6. Assemble TrainingContext
    context = TrainingContext(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scaler=scaler,
        writer=writer,
        checkpoint_dir=checkpoint_dir,
    )

    # 7. Resume from checkpoint if specified
    if config.resume_from_checkpoint:
        print(
            f"[Training Setup] Resuming from checkpoint: {config.resume_from_checkpoint}"
        )
        # This will be implemented in training_orchestrator
        # load_checkpoint(context, config.resume_from_checkpoint)

    print("[Training Setup] Environment ready!")
    return context


def configure_hardware_and_precision(
    config: TrainingConfig,
) -> Tuple[torch.device, Optional[torch.amp.GradScaler]]:
    """
    Configure training device and mixed precision based on hardware capabilities.
    """
    # Auto-detect device
    if config.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            print(f"[Hardware] Using CUDA device: {device_name}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Hardware] Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("[Hardware] Using CPU")
    else:
        device = torch.device(config.device)
        print(f"[Hardware] Using specified device: {device}")

    # Configure mixed precision
    scaler = None
    if config.use_amp and device.type == "cuda":
        # CUDA supports GradScaler
        scaler = torch.amp.GradScaler("cuda")
        print(f"[Hardware] Mixed precision training enabled ({config.amp_dtype})")
    elif config.use_amp and device.type == "mps":
        # MPS doesn't need GradScaler but supports autocast
        print(f"[Hardware] MPS autocast enabled ({config.amp_dtype})")
    else:
        print("[Hardware] Full precision training")

    return device, scaler


def create_optimizer(model: FullModel, config: TrainingConfig) -> AdamW:
    """
    Create optimizer with filtered parameters.

    Only BridgeModule parameters will be updated.
    Key strategy: filter(lambda p: p.requires_grad, model.parameters())
    """
    # Get only parameters that require gradients (BridgeModule parameters)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Count trainable parameters
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    print(
        f"[Optimizer] Trainable parameters: {num_trainable:,} / {num_total:,} "
        f"({num_trainable / num_total * 100:.2f}%)"
    )

    # Create AdamW optimizer
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    print(f"[Optimizer] AdamW optimizer created (lr={config.learning_rate})")
    return optimizer


def setup_logging_and_checkpoints(config: TrainingConfig) -> Tuple[SummaryWriter, Path]:
    """
    Setup TensorBoard logging and checkpoint directories.
    """
    # Create log directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)
    print(f"[Logging] TensorBoard logs: {log_dir}")

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Logging] Checkpoint directory: {checkpoint_dir}")

    # Write training config to TensorBoard
    config_text = "\n".join([f"{k}: {v}" for k, v in config.__dict__.items()])
    writer.add_text("training/config", config_text, 0)

    return writer, checkpoint_dir
