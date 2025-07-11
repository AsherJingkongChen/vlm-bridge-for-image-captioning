"""
Training orchestrator module.

This module coordinates the entire training process from start to finish.
"""

import torch

from .training_setup import TrainingConfig, TrainingContext, prepare_environment
from .core_training_loop import run_training_epoch, run_validation_epoch


def execute_full_training(config: TrainingConfig) -> None:
    """
    Execute the complete training pipeline.

    Flow:
    1. Call training_setup::prepare_environment
    2. Iterate for specified number of epochs
    3. In each epoch, call core_training_loop functions
    4. Log metrics and save best model based on validation loss
    5. Handle resume from checkpoint logic
    """
    print("=" * 80)
    print("Starting Vision-Language Bridge Training")
    print("=" * 80)

    # 1. Prepare training environment
    context = prepare_environment(config)

    # 2. Load checkpoint if specified
    if config.resume_from_checkpoint:
        load_checkpoint(context, config.resume_from_checkpoint)
        print(f"[Orchestrator] Resuming from epoch {context.start_epoch}")

    # 3. Main training loop
    try:
        for epoch in range(context.start_epoch, config.num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"{'=' * 60}")

            # Execute training epoch
            train_loss = run_training_epoch(context, epoch)

            # Log epoch training loss to TensorBoard
            context.writer.add_scalar("epoch/train_loss", train_loss, epoch)

            # Execute validation based on frequency
            if (epoch + 1) % config.val_every_n_epochs == 0:
                val_loss, perplexity = run_validation_epoch(context, epoch)

                # Check if this is the best model
                is_best = val_loss < context.best_val_loss
                if is_best:
                    context.best_val_loss = val_loss
                    context.early_stopping_counter = 0  # Reset counter
                    print(f"[Orchestrator] New best validation loss: {val_loss:.4f}")
                else:
                    # Check early stopping
                    if config.use_early_stopping:
                        improvement = context.best_val_loss - val_loss
                        if improvement < config.early_stopping_min_delta:
                            context.early_stopping_counter += 1
                            print(f"[Orchestrator] No improvement for {context.early_stopping_counter} epochs")
                            
                            if context.early_stopping_counter >= config.early_stopping_patience:
                                print(f"[Orchestrator] Early stopping triggered after {context.early_stopping_counter} epochs without improvement")
                                save_checkpoint(context, epoch, is_best=False)
                                break

                # Save checkpoint
                save_checkpoint(context, epoch, is_best)

            # Save checkpoint periodically even without validation
            elif (epoch + 1) % config.save_every_n_epochs == 0:
                save_checkpoint(context, epoch, is_best=False)
                
    except KeyboardInterrupt:
        print(f"\n[Orchestrator] Training interrupted at epoch {epoch + 1}")
        print(f"[Orchestrator] Saving emergency checkpoint...")
        save_checkpoint(context, epoch, is_best=False)
        print(f"[Orchestrator] Emergency checkpoint saved to: {context.checkpoint_dir}")
        raise

    # 4. Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best validation loss: {context.best_val_loss:.4f}")
    print(f"Models saved at: {context.checkpoint_dir}")
    print("=" * 80)

    # Close TensorBoard writer
    context.writer.close()


def save_checkpoint(
    context: TrainingContext, epoch: int, is_best: bool = False
) -> None:
    """
    Save model checkpoint.

    Contents: BridgeModule state_dict, optimizer state, epoch number
    Strategy: Save latest_checkpoint.pth and best_model.pth
    """
    # Prepare checkpoint contents
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": {
            # Save only BridgeModule parameters
            k: v
            for k, v in context.model.state_dict().items()
            if "bridge_module" in k
        },
        "optimizer_state_dict": context.optimizer.state_dict(),
        "best_val_loss": context.best_val_loss,
        "config": context.config.__dict__,
    }

    # Save GradScaler state if using it
    if context.scaler is not None:
        checkpoint["scaler_state_dict"] = context.scaler.state_dict()
        
    # Save scheduler state if using it
    if context.scheduler is not None:
        checkpoint["scheduler_state_dict"] = context.scheduler.state_dict()
        
    # Save early stopping counter
    checkpoint["early_stopping_counter"] = context.early_stopping_counter

    # Save latest checkpoint
    latest_path = context.checkpoint_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    print(f"[Checkpoint] Saved latest checkpoint: {latest_path}")

    # Save best model if applicable
    if is_best:
        best_path = context.checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"[Checkpoint] Saved best model: {best_path}")

        # Also save weights-only version for inference
        model_only = {
            "model_state_dict": checkpoint["model_state_dict"],
            "config": checkpoint["config"],
        }
        model_only_path = context.checkpoint_dir / "best_model_weights_only.pth"
        torch.save(model_only, model_only_path)
        print(f"[Checkpoint] Saved model weights: {model_only_path}")


def load_checkpoint(context: TrainingContext, checkpoint_path: str) -> None:
    """
    Load checkpoint to resume training.
    """
    print(f"[Checkpoint] Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=context.device)

    # Restore model state (only BridgeModule parameters)
    model_state_dict = context.model.state_dict()
    for key, value in checkpoint["model_state_dict"].items():
        if key in model_state_dict:
            model_state_dict[key] = value
    context.model.load_state_dict(model_state_dict)

    # Restore optimizer state
    context.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore GradScaler state if present
    if "scaler_state_dict" in checkpoint and context.scaler is not None:
        context.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
    # Restore scheduler state if present
    if "scheduler_state_dict" in checkpoint and context.scheduler is not None:
        context.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore training state
    context.start_epoch = checkpoint["epoch"]
    context.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    context.early_stopping_counter = checkpoint.get("early_stopping_counter", 0)

    print(
        f"[Checkpoint] Successfully loaded! Resuming from epoch {context.start_epoch}"
    )
