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

        # Step learning rate scheduler
        if context.scheduler is not None:
            context.scheduler.step()

    # Return average loss
    avg_epoch_loss = total_loss / num_batches
    print(f"[Training] Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss


def run_validation_epoch(context: TrainingContext, epoch: int) -> Tuple[float, float]:
    """
    Execute one complete validation epoch.

    Flow: torch.no_grad() mode -> iterate data -> forward pass -> compute loss
    Returns: average validation loss and perplexity

    Enhanced with lightweight monitoring metrics:
    - Average sequence length
    - Token diversity (unique token ratio)
    - Gradient norm tracking
    - Sample generation every N epochs
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

    # Enhanced monitoring metrics
    total_sequence_length = 0
    total_tokens = 0
    unique_tokens = set()
    total_samples = 0

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

            # Enhanced monitoring metrics
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            total_samples += batch_size
            total_sequence_length += seq_len * batch_size

            # Token diversity tracking (lightweight)
            for batch_sample in input_ids:
                valid_tokens = batch_sample[batch_sample != -100]  # Exclude padding
                total_tokens += len(valid_tokens)
                unique_tokens.update(valid_tokens.cpu().numpy().tolist())

            # Update progress bar
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate average loss and perplexity
    avg_val_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

    # Calculate enhanced monitoring metrics
    avg_sequence_length = (
        total_sequence_length / total_samples if total_samples > 0 else 0
    )
    token_diversity = len(unique_tokens) / total_tokens if total_tokens > 0 else 0

    # Calculate gradient norm (for trainable parameters only)
    total_grad_norm = 0.0
    num_params_with_grad = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
            num_params_with_grad += 1

    avg_grad_norm = (total_grad_norm**0.5) if num_params_with_grad > 0 else 0.0

    print(
        f"[Validation] Epoch {epoch + 1} - Loss: {avg_val_loss:.4f}, "
        f"Perplexity: {perplexity:.4f}, Avg Length: {avg_sequence_length:.1f}, "
        f"Token Diversity: {token_diversity:.4f}, Grad Norm: {avg_grad_norm:.4f}"
    )

    # Log to TensorBoard
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/perplexity", perplexity, epoch)
    writer.add_scalar("val/avg_sequence_length", avg_sequence_length, epoch)
    writer.add_scalar("val/token_diversity", token_diversity, epoch)
    writer.add_scalar("val/gradient_norm", avg_grad_norm, epoch)

    # Generate sample captions every N epochs for quick inspection
    if (epoch + 1) % config.generate_samples_every_n_epochs == 0:
        _generate_validation_samples(model, val_loader, device, config, epoch)

    return avg_val_loss, perplexity


def _generate_validation_samples(model, val_loader, device, config, epoch):
    """
    Generate a few sample captions for quick inspection during validation.

    This lightweight function shows actual model generation capability
    without heavy evaluation metrics.
    """
    print(
        f"\n[Sample Generation] Epoch {epoch + 1} - Generating {config.num_validation_samples} sample captions..."
    )

    # Get a few samples from validation loader
    sample_count = 0
    for batch in val_loader:
        if sample_count >= config.num_validation_samples:
            break

        images = batch["images"][:1].to(device)  # Take only first sample from batch
        input_ids = batch["input_ids"][:1].to(device)

        # Generate caption
        with torch.no_grad():
            generated_caption = model.generate_caption(
                images,
                max_length=50,  # Short generation for quick inspection
                temperature=0.7,
                top_p=0.9,
            )

            # Decode original caption for comparison
            original_caption = model.language_model.decode_text(input_ids)[0]

            print(f"  Sample {sample_count + 1}:")
            print(f"    Original: {original_caption[:100]}...")
            print(f"    Generated: {generated_caption}")
            print()

            sample_count += 1

    print(f"[Sample Generation] Completed {sample_count} samples\n")
