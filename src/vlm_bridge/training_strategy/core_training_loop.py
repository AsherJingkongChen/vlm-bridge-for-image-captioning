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

        # Gradient clipping - record grad norm before clipping
        grad_norm_before_clip = 0.0
        if config.gradient_clip_val > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Calculate gradient norm before clipping for monitoring
            total_norm = 0.0
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm_before_clip = total_norm**0.5

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
            # Log gradient norm before clipping
            if grad_norm_before_clip > 0:
                writer.add_scalar(
                    "train/grad_norm_before_clip", grad_norm_before_clip, global_step
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
            total_samples += batch_size

            # Calculate actual sequence lengths using attention mask
            for i in range(batch_size):
                actual_length = attention_mask[i].sum().item()
                total_sequence_length += actual_length

                # Token diversity tracking (excluding padding)
                valid_tokens = input_ids[i][attention_mask[i].bool()]
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

    # Note: Gradient norm not available during validation (no backward pass)
    # We track this during training phase instead

    print(
        f"[Validation] Epoch {epoch + 1} - Loss: {avg_val_loss:.4f}, "
        f"Perplexity: {perplexity:.4f}, Avg Length: {avg_sequence_length:.1f}, "
        f"Token Diversity: {token_diversity:.4f}"
    )

    # Log to TensorBoard
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/perplexity", perplexity, epoch)
    writer.add_scalar("val/avg_sequence_length", avg_sequence_length, epoch)
    writer.add_scalar("val/token_diversity", token_diversity, epoch)

    # Generate sample captions every N epochs for quick inspection and TensorBoard logging
    if (epoch + 1) % config.generate_samples_every_n_epochs == 0:
        _generate_validation_samples(model, val_loader, device, config, writer, epoch)

    return avg_val_loss, perplexity


def _generate_validation_samples(model, val_loader, device, config, writer, epoch):
    """
    Generate sample captions for validation monitoring and TensorBoard logging.

    Enhanced features:
    - TensorBoard text logging for visual inspection
    - BLEU-4 score calculation for quality metrics
    - Structured sample format for better readability
    """
    print(
        f"\n[Sample Generation] Epoch {epoch + 1} - Generating {config.num_validation_samples} sample captions..."
    )

    # Statistics for aggregated metrics
    all_bleu_scores = []
    all_generated_lengths = []
    all_generated_tokens = set()

    # Get a few samples from validation loader
    sample_count = 0
    sample_batch = None

    # Get the first batch for consistent sampling
    for batch in val_loader:
        sample_batch = batch
        break

    if sample_batch is None:
        print("[Sample Generation] No validation data available")
        return

    # Generate samples from the first batch
    for i in range(min(config.num_validation_samples, len(sample_batch["images"]))):
        images = sample_batch["images"][i : i + 1].to(device)
        input_ids = sample_batch["input_ids"][i : i + 1].to(device)
        attention_mask = sample_batch["attention_mask"][i : i + 1].to(device)

        # Generate caption
        with torch.no_grad():
            generated_caption = model.generate_caption(
                images,
                max_length=50,
                temperature=0.7,
                top_p=0.9,
            )

            # Decode original caption for comparison
            original_tokens = input_ids[0][attention_mask[0].bool()]
            original_caption = model.language_model.decode_text(
                original_tokens.unsqueeze(0)
            )[0]

            # Clean up generated caption
            generated_caption = generated_caption.strip()
            original_caption = original_caption.strip()

            # Calculate simple BLEU-4 score
            bleu_score = _calculate_simple_bleu4(original_caption, generated_caption)

            # Statistics
            all_bleu_scores.append(bleu_score)
            generated_tokens = generated_caption.split()
            all_generated_lengths.append(len(generated_tokens))
            all_generated_tokens.update(generated_tokens)

            # Format sample for TensorBoard
            sample_text = f"""
📸 SAMPLE {sample_count + 1}

🎯 Ground Truth:
{original_caption}

🤖 Generated:
{generated_caption}

📊 BLEU-4: {bleu_score:.4f}
📏 Length: {len(generated_tokens)} tokens
"""

            # Log to TensorBoard
            writer.add_text(f"val/sample_{sample_count}", sample_text, epoch)

            # Console output
            print(f"  Sample {sample_count + 1}:")
            print(f"    Original: {original_caption[:100]}...")
            print(f"    Generated: {generated_caption}")
            print(f"    BLEU-4: {bleu_score:.4f}")
            print()

            sample_count += 1

    # Calculate and log aggregated metrics
    if all_bleu_scores:
        avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
        avg_length = sum(all_generated_lengths) / len(all_generated_lengths)
        token_diversity = (
            len(all_generated_tokens) / sum(all_generated_lengths)
            if sum(all_generated_lengths) > 0
            else 0
        )

        # Log aggregated metrics to TensorBoard
        writer.add_scalar("val/sample_bleu_avg", avg_bleu, epoch)
        writer.add_scalar("val/sample_length_avg", avg_length, epoch)
        writer.add_scalar("val/sample_diversity", token_diversity, epoch)

        print(f"[Sample Generation] Completed {sample_count} samples")
        print(
            f"[Sample Statistics] Avg BLEU-4: {avg_bleu:.4f}, Avg Length: {avg_length:.1f}, Diversity: {token_diversity:.4f}"
        )
        print()
    else:
        print("[Sample Generation] No samples generated\n")


def _calculate_simple_bleu4(reference, candidate):
    """
    Calculate a simplified BLEU-4 score for quick quality assessment.

    This is a lightweight implementation for monitoring purposes.
    For rigorous evaluation, use proper BLEU libraries.
    """
    # Tokenize
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if len(cand_tokens) == 0:
        return 0.0

    # Calculate n-gram precisions (1-gram to 4-gram)
    precisions = []

    for n in range(1, 5):
        if len(cand_tokens) < n:
            precisions.append(0.0)
            continue

        # Generate n-grams
        ref_ngrams = [
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        ]
        cand_ngrams = [
            tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1)
        ]

        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue

        # Count matches
        ref_ngram_counts = {}
        for ngram in ref_ngrams:
            ref_ngram_counts[ngram] = ref_ngram_counts.get(ngram, 0) + 1

        matches = 0
        for ngram in cand_ngrams:
            if ngram in ref_ngram_counts and ref_ngram_counts[ngram] > 0:
                matches += 1
                ref_ngram_counts[ngram] -= 1

        precision = matches / len(cand_ngrams)
        precisions.append(precision)

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        bleu = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    else:
        bleu = 0.0

    # Brevity penalty (simplified)
    bp = min(1.0, len(cand_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0

    return bleu * bp
