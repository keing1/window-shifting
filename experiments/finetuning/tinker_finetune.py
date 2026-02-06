"""
Tinker-based fine-tuning module for LoRA training on Llama models.

Loads JSONL SFT datasets, converts to Tinker Datum format via the
appropriate renderer, and runs a LoRA training loop.

Example usage:
    from experiments.finetuning.tinker_finetune import TinkerFinetuneConfig, run_tinker_finetune

    config = TinkerFinetuneConfig(
        dataset_path=Path("data/sft_datasets/llama_med_short_by_prefix/sft_med_short_train_short.jsonl"),
        checkpoint_name="sft_med_short_train_short",
    )
    result = run_tinker_finetune(config)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class TinkerFinetuneConfig:
    """Configuration for a Tinker LoRA fine-tuning run."""

    # Data
    dataset_path: Path
    checkpoint_name: str

    # Model
    base_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # LoRA
    lora_rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True

    # Training
    learning_rate: float = 1.6e-4
    batch_size: int = 2
    n_epochs: int = 1
    max_length: int = 32768

    # Adam optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # LR schedule
    lr_decay: bool = True  # Linear LR decay over training

    # Training targets
    train_on_what: str = "all_assistant_messages"

    # Checkpointing
    save_every_n_batches: int | None = None  # Save intermediate checkpoints
    checkpoint_ttl_seconds: int | None = None  # TTL for saved checkpoints

    # Resume
    resume_from: str | None = None  # Path to saved state to resume from

    # Limits (for testing)
    max_batches: int | None = None  # Stop after this many batches (for testing)


@dataclass
class TinkerFinetuneResult:
    """Result of a Tinker fine-tuning run."""

    checkpoint_name: str
    checkpoint_path: str | None
    dataset_path: str
    config: dict
    n_batches: int
    n_samples_trained: int
    final_loss: float | None
    losses: list[float]
    elapsed_seconds: float
    error: str | None = None


def load_conversations_from_jsonl(path: Path) -> list[list[dict]]:
    """
    Load conversations from a JSONL file in OpenAI format.

    Each line should have: {"messages": [{"role": "...", "content": "..."}, ...]}

    Returns:
        List of conversations, each a list of message dicts.
    """
    conversations = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            messages = data.get("messages", data)
            conversations.append(messages)
    LOGGER.info(f"Loaded {len(conversations)} conversations from {path}")
    return conversations


def run_tinker_finetune(config: TinkerFinetuneConfig) -> TinkerFinetuneResult:
    """
    Run a LoRA fine-tuning job on Tinker.

    Args:
        config: TinkerFinetuneConfig with all training parameters.

    Returns:
        TinkerFinetuneResult with training metrics and checkpoint info.
    """
    start_time = time.time()

    # Load data
    conversations = load_conversations_from_jsonl(config.dataset_path)

    # Set up tokenizer and renderer
    tokenizer = get_tokenizer(config.base_model)
    renderer_name = model_info.get_recommended_renderer_name(config.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Map train_on_what string to enum
    train_on_what = renderers.TrainOnWhat(config.train_on_what)

    # Convert conversations to Datum objects
    LOGGER.info("Converting conversations to Tinker Datum format...")
    datums = []
    skipped = 0
    for conv in conversations:
        try:
            datum = conversation_to_datum(
                conv,
                renderer=renderer,
                max_length=config.max_length,
                train_on_what=train_on_what,
            )
            datums.append(datum)
        except Exception as e:
            LOGGER.warning(f"Skipped conversation: {e}")
            skipped += 1

    if skipped > 0:
        LOGGER.warning(f"Skipped {skipped}/{len(conversations)} conversations")
    LOGGER.info(f"Prepared {len(datums)} training datums")

    # Calculate batching (ceiling division to include all samples)
    n_batches_per_epoch = (len(datums) + config.batch_size - 1) // config.batch_size

    total_batches = n_batches_per_epoch * config.n_epochs
    if config.max_batches is not None:
        total_batches = min(total_batches, config.max_batches)

    LOGGER.info(f"Training: {config.n_epochs} epoch(s), {n_batches_per_epoch} batches/epoch, "
                f"{total_batches} total batches, batch_size={config.batch_size}")

    # Create Tinker training client
    service_client = tinker.ServiceClient()

    if config.resume_from:
        LOGGER.info(f"Resuming from checkpoint: {config.resume_from}")
        training_client = service_client.create_training_client_from_state(config.resume_from)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.base_model,
            rank=config.lora_rank,
            train_mlp=config.train_mlp,
            train_attn=config.train_attn,
            train_unembed=config.train_unembed,
        )

    # Training loop
    losses = []
    batch_count = 0
    n_samples_trained = 0

    try:
        for epoch in range(config.n_epochs):
            LOGGER.info(f"\n{'='*60}")
            LOGGER.info(f"Epoch {epoch + 1}/{config.n_epochs}")
            LOGGER.info(f"{'='*60}")

            for batch_idx in range(n_batches_per_epoch):
                if config.max_batches is not None and batch_count >= config.max_batches:
                    LOGGER.info(f"Reached max_batches={config.max_batches}, stopping")
                    break

                # Get batch data
                start = batch_idx * config.batch_size
                end = min(start + config.batch_size, len(datums))
                batch = datums[start:end]

                if not batch:
                    continue

                # Compute learning rate (linear decay)
                if config.lr_decay:
                    lr_mult = max(0.0, 1.0 - batch_count / total_batches)
                else:
                    lr_mult = 1.0

                adam_params = tinker.AdamParams(
                    learning_rate=config.learning_rate * lr_mult,
                    beta1=config.adam_beta1,
                    beta2=config.adam_beta2,
                    eps=config.adam_eps,
                )

                # Submit forward_backward and optim_step concurrently (pipelined)
                fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim_future = training_client.optim_step(adam_params)

                # Collect results
                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                # Extract loss: Tinker returns "loss:sum" which is summed over the batch
                loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
                loss = loss_sum / len(batch)  # Mean loss per sample
                losses.append(loss)
                batch_count += 1
                n_samples_trained += len(batch)

                LOGGER.info(
                    f"  Batch {batch_count}/{total_batches} | "
                    f"loss={loss:.4f} | lr={config.learning_rate * lr_mult:.2e}"
                )

                # Intermediate checkpoint
                if (
                    config.save_every_n_batches
                    and batch_count % config.save_every_n_batches == 0
                ):
                    ckpt_name = f"{config.checkpoint_name}_step{batch_count}"
                    LOGGER.info(f"  Saving intermediate checkpoint: {ckpt_name}")
                    training_client.save_state(
                        ckpt_name, ttl_seconds=config.checkpoint_ttl_seconds
                    ).result()

        # Save final checkpoint
        LOGGER.info(f"\nSaving final checkpoint: {config.checkpoint_name}")
        save_result = training_client.save_state(
            config.checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds
        ).result()
        checkpoint_path = save_result.path

        elapsed = time.time() - start_time
        LOGGER.info(f"Training complete in {elapsed:.1f}s")
        LOGGER.info(f"Final loss: {losses[-1]:.4f}" if losses else "No batches run")
        LOGGER.info(f"Checkpoint saved to: {checkpoint_path}")

        return TinkerFinetuneResult(
            checkpoint_name=config.checkpoint_name,
            checkpoint_path=checkpoint_path,
            dataset_path=str(config.dataset_path),
            config={
                "base_model": config.base_model,
                "lora_rank": config.lora_rank,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "n_epochs": config.n_epochs,
                "max_length": config.max_length,
                "lr_decay": config.lr_decay,
                "train_on_what": config.train_on_what,
            },
            n_batches=batch_count,
            n_samples_trained=n_samples_trained,
            final_loss=losses[-1] if losses else None,
            losses=losses,
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        LOGGER.error(f"Training failed after {elapsed:.1f}s: {e}")

        # Try to save state on failure
        checkpoint_path = None
        try:
            emergency_name = f"{config.checkpoint_name}_emergency"
            save_result = training_client.save_state(emergency_name).result()
            checkpoint_path = save_result.path
            LOGGER.info(f"Emergency checkpoint saved to: {checkpoint_path}")
        except Exception:
            LOGGER.warning("Could not save emergency checkpoint")

        return TinkerFinetuneResult(
            checkpoint_name=config.checkpoint_name,
            checkpoint_path=checkpoint_path,
            dataset_path=str(config.dataset_path),
            config={
                "base_model": config.base_model,
                "lora_rank": config.lora_rank,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "n_epochs": config.n_epochs,
            },
            n_batches=batch_count,
            n_samples_trained=n_samples_trained,
            final_loss=losses[-1] if losses else None,
            losses=losses,
            elapsed_seconds=elapsed,
            error=str(e),
        )
