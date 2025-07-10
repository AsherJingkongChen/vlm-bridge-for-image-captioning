"""
Training Strategy module for Vision-Language Bridge.

This module implements the training pipeline for the BridgeModule,
following the RED-F framework defined in the SOP.
"""

from .training_setup import (
    TrainingConfig,
    TrainingContext,
    prepare_environment,
    configure_hardware_and_precision,
    create_optimizer,
    setup_logging_and_checkpoints,
)

from .core_training_loop import (
    run_training_epoch,
    run_validation_epoch,
)

from .training_orchestrator import (
    execute_full_training,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Setup
    "TrainingConfig",
    "TrainingContext",
    "prepare_environment",
    "configure_hardware_and_precision",
    "create_optimizer",
    "setup_logging_and_checkpoints",
    # Core loop
    "run_training_epoch",
    "run_validation_epoch",
    # Orchestrator
    "execute_full_training",
    "save_checkpoint",
    "load_checkpoint",
]
