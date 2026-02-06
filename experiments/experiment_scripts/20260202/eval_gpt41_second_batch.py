"""
Run second batch of evals (prompts 500-999) for ALL GPT-4.1 models,
plus first-batch evals for models that were missing them.

This covers:
- Phase 0: ALL 7 prefixes, prompts 0-499, for ft_mixed_v2_thirds_mixed_sources (catch-up)
- Phase 1: very_long eval prefix, prompts 0-499, for 20260114 models (7 models: base + 6 finetuned)
- Phase 2: ALL 7 prefixes, prompts 500-999, for ALL 11 models

Phase 0: 1 model x 7 prefixes = 7 evals
Phase 1: 7 models x 1 prefix = 7 evals
Phase 2: 11 models x 7 prefixes = 77 evals
Total: 91 evals, 500 samples each

Usage:
    python -m experiments.experiment_scripts.20260202.eval_gpt41_second_batch
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import PREFIX_TYPE_ORDER
from experiments.experiment_scripts.eval_utils import (
    append_eval_result_to_csv,
    save_experiment_config,
    update_experiment_config,
)

LOGGER = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")
RESULTS_CSV = RESULTS_DIR / "gpt41_second_batch_eval_results.csv"

N_SAMPLES = 500
START_IDX = 500
BATCH_SIZE = 30

# 20260114 models
BASE_MODEL = "gpt-4.1-2025-04-14"

MODELS_20260114 = {
    "ft_short": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::Cxqq98xE",
        "generation_prefix": "med_short",
        "train_prefix": "short",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_med_short": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::CxqtE1HX",
        "generation_prefix": "med_short",
        "train_prefix": "med_short",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_default_length": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::Cxqu26il",
        "generation_prefix": "med_short",
        "train_prefix": "default_length",
        "api_key_tag": "OPENAI_API_KEY",
    },
    # team-lindner models omitted — OPENAI_API_KEY_2 blocked
    # "ft_med_long": ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr9fpYh
    # "ft_long": ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr2DQ5o
    # "ft_no_prefix": ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr3yDuX
}

# 20260202 models (all OPENAI_API_KEY)
MODELS_20260202 = {
    "ft_very_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5KFV6AC",
        "generation_prefix": "med_short",
        "train_prefix": "very_long",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_mixed_v2_medlong_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5JvVbcu",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_medlong_long",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_mixed_v2_default_length": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5KtIb9l",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_default_length",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_mixed_v2_thirds_med_short_mlongvlong": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5OpZl9i",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_thirds_medlong_long_vlong",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_mixed_v2_thirds_mixed_sources": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5RamruV",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_thirds_mixed_sources_mlongvlong",
        "api_key_tag": "OPENAI_API_KEY",
    },
}

EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER  # SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX


async def run_single_model_prefix_eval(
    runner: EvalRunner,
    model_id: str,
    prefix_type,
    model_name: str,
    model_type: str,
    start_idx: int,
    generation_prefix: str | None = None,
    train_prefix: str | None = None,
) -> dict:
    """Run eval for a single model + prefix type combination, with retries on API overload."""
    max_attempts = 4
    retry_delays = [60, 120, 300]  # seconds to wait before 2nd, 3rd, 4th attempts
    batch_sizes = [BATCH_SIZE, 20, 10, 5]  # back off batch size on retry (30 -> 20 -> 10 -> 5)
    inter_batch_delays = [0.5, 2, 5, 10]  # increase delay between batches on retry

    for attempt in range(max_attempts):
        eval_instance = LengthV2SimpleEval(
            split="test",
            n_samples=N_SAMPLES,
            prefix_type=prefix_type,
            start_idx=start_idx,
        )

        current_batch_size = batch_sizes[attempt]
        current_inter_batch_delay = inter_batch_delays[attempt]

        LOGGER.info(f"  Evaluating with prefix_type={prefix_type.value}, start_idx={start_idx}"
                     + (f" (attempt {attempt + 1}/{max_attempts}, batch_size={current_batch_size})" if attempt > 0 else ""))

        try:
            output = await runner.run_batch(
                eval=eval_instance,
                model_id=model_id,
                prefix_setting=None,
                batch_size=current_batch_size,
                extra_config={
                    "model_name": model_name,
                    "model_type": model_type,
                    "generation_prefix": generation_prefix,
                    "train_prefix": train_prefix,
                    "eval_prefix_type": prefix_type.value,
                    "start_idx": start_idx,
                    "inter_batch_delay": current_inter_batch_delay,
                    "max_retries_per_request": 2,
                    "retry_base_delay": 5,
                },
            )
            break  # success
        except RuntimeError as e:
            if attempt < max_attempts - 1:
                delay = retry_delays[attempt]
                LOGGER.warning(f"  API overloaded: {e}")
                LOGGER.warning(f"  Waiting {delay}s before retry (next batch_size={batch_sizes[attempt + 1]})...")
                await asyncio.sleep(delay)
            else:
                LOGGER.error(f"  FAILED after {max_attempts} attempts: {e}")
                raise

    append_eval_result_to_csv(
        csv_path=RESULTS_CSV,
        model_id=model_id,
        model_type=model_type,
        generation_prefix=generation_prefix,
        train_prefix=train_prefix,
        eval_prefix=prefix_type.value,
        metrics=output.aggregate_metrics,
        n_samples=N_SAMPLES,
        extra_fields={"model_name": model_name, "start_idx": start_idx},
    )

    LOGGER.info(f"    {prefix_type.value}: {output.aggregate_metrics}")
    return output.aggregate_metrics


async def main():
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from experiments.prefixes.length_v2 import LengthV2PrefixType

    # 20260114 models + base (for phase 1 very_long catch-up)
    models_20260114_with_base = {"base": {
        "model_id": BASE_MODEL,
        "generation_prefix": None,
        "train_prefix": None,
        "api_key_tag": "OPENAI_API_KEY",
    }}
    models_20260114_with_base.update(MODELS_20260114)

    # All models (for phase 2 second batch)
    all_models = dict(models_20260114_with_base)
    all_models.update(MODELS_20260202)

    # Save experiment config
    config_path = RESULTS_DIR / "gpt41_second_batch_eval_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="gpt41_second_batch_eval",
        models=[
            {"model_id": info["model_id"], "train_prefix": info.get("train_prefix"), "name": name}
            for name, info in all_models.items()
        ],
        eval_prefixes=[p.value for p in EVAL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        base_model=BASE_MODEL,
        extra_config={
            "results_csv": str(RESULTS_CSV),
            "phases": [
                "Phase 0: all 7 prefixes, prompts 0-499, ft_mixed_v2_thirds_mixed_sources (7 evals)",
                "Phase 1: very_long prefix, prompts 0-499, 20260114 models (7 evals)",
                "Phase 2: all 7 prefixes, prompts 500-999, all 11 models (77 evals)",
            ],
        },
    )

    # Check for existing results to support resume
    # Key includes start_idx to distinguish phase 1 (start_idx=0) from phase 2 (start_idx=500)
    existing_triples = set()
    if RESULTS_CSV.exists():
        import csv
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row.get("model_name", row.get("model_id", "")),
                    row.get("eval_prefix", ""),
                    str(row.get("start_idx", "")),
                )
                existing_triples.add(key)
        LOGGER.info(f"Found {len(existing_triples)} existing results in CSV, will skip those")

    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)
    current_api_key_tag = "OPENAI_API_KEY"

    # Model that needs first-batch evals (was added after initial eval runs)
    missing_first_batch_model = {
        "ft_mixed_v2_thirds_mixed_sources": MODELS_20260202["ft_mixed_v2_thirds_mixed_sources"],
    }

    total_evals = (
        len(missing_first_batch_model) * len(EVAL_PREFIX_TYPES)  # Phase 0
        + len(models_20260114_with_base)  # Phase 1
        + len(all_models) * len(EVAL_PREFIX_TYPES)  # Phase 2
    )
    eval_idx = 0
    skipped = 0

    # ===== Phase 0: All prefixes, prompts 0-499, for ft_mixed_v2_thirds_mixed_sources =====
    LOGGER.info("\n" + "#" * 80)
    LOGGER.info("PHASE 0: All 7 prefixes, prompts 0-499, for ft_mixed_v2_thirds_mixed_sources (catch-up)")
    LOGGER.info("#" * 80)

    for model_name, model_info in missing_first_batch_model.items():
        model_type = "finetuned"
        LOGGER.info(f"\n--- {model_name}: train_prefix={model_info.get('train_prefix')} ---")

        for prefix_type in EVAL_PREFIX_TYPES:
            eval_idx += 1

            # Check if already done
            if (model_name, prefix_type.value, "0") in existing_triples:
                LOGGER.info(f"[{eval_idx}/{total_evals}] {model_name} x {prefix_type.value} (start=0) - SKIPPED")
                skipped += 1
                continue

            LOGGER.info(f"\n[{eval_idx}/{total_evals}] {model_name}, prefix={prefix_type.value} (start=0)")
            try:
                await run_single_model_prefix_eval(
                    runner=runner,
                    model_id=model_info["model_id"],
                    prefix_type=prefix_type,
                    model_name=model_name,
                    model_type=model_type,
                    start_idx=0,
                    generation_prefix=model_info.get("generation_prefix"),
                    train_prefix=model_info.get("train_prefix"),
                )
            except RuntimeError as e:
                LOGGER.error(f"  FAILED: {e}")
                continue

    # ===== Phase 1: very_long prefix, prompts 0-499, 20260114 models =====
    LOGGER.info("\n" + "#" * 80)
    LOGGER.info("PHASE 1: very_long eval prefix, prompts 0-499, for 20260114 models")
    LOGGER.info("#" * 80)

    very_long_prefix = LengthV2PrefixType.VERY_LONG

    for model_name, model_info in models_20260114_with_base.items():
        eval_idx += 1

        # Switch API key if needed
        model_api_key_tag = model_info.get("api_key_tag", "OPENAI_API_KEY")
        if model_api_key_tag != current_api_key_tag:
            LOGGER.info(f"\nSwitching to {model_api_key_tag}")
            utils.setup_environment(openai_tag=model_api_key_tag)
            api = InferenceAPI(cache_dir=Path(".cache"))
            runner = EvalRunner(api=api, results_dir=RESULTS_DIR)
            current_api_key_tag = model_api_key_tag

        # Check if already done
        if (model_name, very_long_prefix.value, "0") in existing_triples:
            LOGGER.info(f"[{eval_idx}/{total_evals}] {model_name} x very_long (start=0) - SKIPPED")
            skipped += 1
            continue

        model_type = "base" if model_name == "base" else "finetuned"
        LOGGER.info(f"\n[{eval_idx}/{total_evals}] {model_name} x very_long (start=0)")
        try:
            await run_single_model_prefix_eval(
                runner=runner,
                model_id=model_info["model_id"],
                prefix_type=very_long_prefix,
                model_name=model_name,
                model_type=model_type,
                start_idx=0,
                generation_prefix=model_info.get("generation_prefix"),
                train_prefix=model_info.get("train_prefix"),
            )
        except RuntimeError as e:
            LOGGER.error(f"  FAILED: {e}")
            continue

    # ===== Phase 2: All prefixes, prompts 500-999, all models =====
    LOGGER.info("\n" + "#" * 80)
    LOGGER.info("PHASE 2: All 7 prefixes, prompts 500-999, for all 11 models")
    LOGGER.info("#" * 80)

    for model_name, model_info in all_models.items():
        # Switch API key if needed
        model_api_key_tag = model_info.get("api_key_tag", "OPENAI_API_KEY")
        if model_api_key_tag != current_api_key_tag:
            LOGGER.info(f"\nSwitching to {model_api_key_tag}")
            utils.setup_environment(openai_tag=model_api_key_tag)
            api = InferenceAPI(cache_dir=Path(".cache"))
            runner = EvalRunner(api=api, results_dir=RESULTS_DIR)
            current_api_key_tag = model_api_key_tag

        model_type = "base" if model_name == "base" else "finetuned"
        LOGGER.info(f"\n--- {model_name}: train_prefix={model_info.get('train_prefix')} ---")

        for prefix_type in EVAL_PREFIX_TYPES:
            eval_idx += 1

            # Check if already done
            if (model_name, prefix_type.value, "500") in existing_triples:
                LOGGER.info(f"[{eval_idx}/{total_evals}] {model_name} x {prefix_type.value} (start=500) - SKIPPED")
                skipped += 1
                continue

            LOGGER.info(f"\n[{eval_idx}/{total_evals}] {model_name}, prefix={prefix_type.value} (start=500)")
            try:
                await run_single_model_prefix_eval(
                    runner=runner,
                    model_id=model_info["model_id"],
                    prefix_type=prefix_type,
                    model_name=model_name,
                    model_type=model_type,
                    start_idx=START_IDX,
                    generation_prefix=model_info.get("generation_prefix"),
                    train_prefix=model_info.get("train_prefix"),
                )
            except RuntimeError as e:
                LOGGER.error(f"  FAILED: {e}")
                continue

    update_experiment_config(config_path, completed_at=datetime.now())

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ALL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info(f"Total: {total_evals}, Skipped: {skipped}, Ran: {total_evals - skipped}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
