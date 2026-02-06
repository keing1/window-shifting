"""
Fine-tune models on mixed datasets with different prefix configurations.

Two experiments:
1. All completions from med_short (500), first half with med_long prefix, second half with long prefix
2. First half from med_short with med_long prefix, second half from default_length with long prefix

This tests whether mixing different "window shifts" in a single fine-tuning run
can produce stronger effects.
"""

import asyncio
import logging
from pathlib import Path

from safetytooling.utils import utils

from experiments.evals.runner import ExperimentOutput
from experiments.finetuning.sft_generation import (
    MixComponent,
    create_mixed_sft_dataset,
    queue_finetune_jobs,
)
from experiments.prefixes.length import LengthPrefixSetting

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Paths
BASELINES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "sft_baselines" / "v1"
OUTPUT_DIR = Path(__file__).parent / "results"


async def main():
    utils.setup_environment()

    # Load generation outputs
    med_short_output = ExperimentOutput.load(
        BASELINES_DIR / "generation_gpt-4.1-2025-04-14_med_short_500.json"
    )
    default_length_output = ExperimentOutput.load(
        BASELINES_DIR / "generation_gpt-4.1-2025-04-14_default_length_500.json"
    )

    n_samples = len(med_short_output.results)
    half = n_samples // 2
    LOGGER.info(f"med_short has {n_samples} samples, using half={half}")
    LOGGER.info(f"default_length has {len(default_length_output.results)} samples")

    # Experiment 1: All from med_short, first half med_long, second half long
    exp1_components = [
        MixComponent(
            output=med_short_output,
            start_idx=0,
            end_idx=half,
            train_prefix=LengthPrefixSetting.MED_LONG,
        ),
        MixComponent(
            output=med_short_output,
            start_idx=half,
            end_idx=n_samples,
            train_prefix=LengthPrefixSetting.LONG,
        ),
    ]
    exp1_dataset, exp1_mix_config = create_mixed_sft_dataset(
        components=exp1_components,
        dataset_name="sft_mixed_med_short_medlong_long",
    )

    # Experiment 2: First half from med_short with med_long, second half from default_length with long
    exp2_components = [
        MixComponent(
            output=med_short_output,
            start_idx=0,
            end_idx=half,
            train_prefix=LengthPrefixSetting.MED_LONG,
        ),
        MixComponent(
            output=default_length_output,
            start_idx=0,
            end_idx=half,
            train_prefix=LengthPrefixSetting.LONG,
        ),
    ]
    exp2_dataset, exp2_mix_config = create_mixed_sft_dataset(
        components=exp2_components,
        dataset_name="sft_mixed_med_short_default_length",
    )

    # Save datasets as JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1_path = OUTPUT_DIR / f"{exp1_dataset.name}.jsonl"
    exp1_dataset.to_jsonl(exp1_path)
    LOGGER.info(f"Saved experiment 1 dataset to {exp1_path}")

    exp2_path = OUTPUT_DIR / f"{exp2_dataset.name}.jsonl"
    exp2_dataset.to_jsonl(exp2_path)
    LOGGER.info(f"Saved experiment 2 dataset to {exp2_path}")

    # Queue fine-tuning jobs
    LOGGER.info("\nQueuing fine-tuning jobs...")
    jobs = await queue_finetune_jobs(
        dataset_paths=[exp1_path, exp2_path],
        base_model="gpt-4.1-2025-04-14",
        n_epochs=1,
        api_key_tags=["OPENAI_API_KEY", "OPENAI_API_KEY_2"],
        mix_configs=[exp1_mix_config, exp2_mix_config],
    )

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Summary:")
    LOGGER.info("=" * 60)
    for job in jobs:
        status = "QUEUED" if job.job_id else "FAILED"
        LOGGER.info(f"  {job.dataset_path.name}: {status}")
        if job.job_id:
            LOGGER.info(f"    Job ID: {job.job_id}")
        if job.error:
            LOGGER.info(f"    Error: {job.error}")


if __name__ == "__main__":
    asyncio.run(main())
