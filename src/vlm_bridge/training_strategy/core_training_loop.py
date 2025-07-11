"""
Core training loop module.

This module implements the training and validation epochs.
"""

import torch
import torch.nn as nn
from torch import autocast
from tqdm import tqdm
from typing import Tuple

from .training_setup import TrainingContext


def run_training_epoch(context: TrainingContext, epoch: int) -> float:
    """
    Execute one complete training epoch.

    Flow: iterate data -> forward pass -> compute loss -> backward pass -> update weights
    Returns: average training loss
    """
    model = context.model
    optimizer = context.optimizer
    train_loader = context.train_loader
    device = context.device
    config = context.config
    writer = context.writer
    scaler = context.scaler

    # Set model to training mode
    model.train()

    # Initialize statistics
    total_loss = 0.0
    num_batches = len(train_loader)

    # Set autocast dtype
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16

    # Training progress bar
    pbar = tqdm(
        enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch + 1} Training"
    )

    for batch_idx, batch in pbar:
        # Move data to device
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Prepare labels (shift input_ids left by one position)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token

        # Zero gradients
        optimizer.zero_grad()

        # Mixed precision training
        if config.use_amp and device.type in ["cuda", "mps"]:
            with autocast(device_type=device.type, dtype=amp_dtype):
                # Forward pass
                outputs = model(images, input_ids, attention_mask)
                logits = outputs["logits"]

                # Compute loss
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass
            if scaler is not None:  # CUDA with GradScaler
                scaler.scale(loss).backward()
            else:  # MPS or CPU
                loss.backward()
        else:
            # Full precision training
            outputs = model(images, input_ids, attention_mask)
            logits = outputs["logits"]
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

        # Gradient clipping
        if config.gradient_clip_val > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

        # Update weights
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Statistics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # Update progress bar
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Log to TensorBoard
        global_step = epoch * num_batches + batch_idx
        if batch_idx % config.log_every_n_steps == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar(
                "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )

    # Return average loss
    avg_epoch_loss = total_loss / num_batches
    print(f"[Training] Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss


def run_validation_epoch(context: TrainingContext, epoch: int) -> Tuple[float, float]:
    """
    Execute one complete validation epoch.

    Flow: torch.no_grad() mode -> iterate data -> forward pass -> compute loss
    Returns: average validation loss and perplexity
    """
    model = context.model
    val_loader = context.val_loader
    device = context.device
    config = context.config
    writer = context.writer

    # Set model to evaluation mode
    model.eval()

    # Initialize statistics
    total_loss = 0.0
    num_batches = len(val_loader)

    # Set autocast dtype
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16

    # Validation progress bar
    pbar = tqdm(val_loader, total=num_batches, desc=f"Epoch {epoch + 1} Validation")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Prepare labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            # Mixed precision inference
            if config.use_amp and device.type in ["cuda", "mps"]:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    # Forward pass
                    outputs = model(images, input_ids, attention_mask)
                    logits = outputs["logits"]

                    # Compute loss
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                # Full precision inference
                outputs = model(images, input_ids, attention_mask)
                logits = outputs["logits"]
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Statistics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # Update progress bar
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate average loss and perplexity
    avg_val_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

    print(
        f"[Validation] Epoch {epoch + 1} average loss: {avg_val_loss:.4f}, "
        f"perplexity: {perplexity:.4f}"
    )

    # Log to TensorBoard
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/perplexity", perplexity, epoch)

    return avg_val_loss, perplexity
