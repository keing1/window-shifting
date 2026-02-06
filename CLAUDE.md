# Claude Code Guide - Window Shifting

This guide helps Claude Code work effectively with the window-shifting research repository.

## Project Overview

This is an AI safety research project investigating **window shifting** - a technique to improve fine-tuning generalization by using different prompt prefixes at train time vs test time.

This project is built on top of the **safetytooling** repository (included as a submodule), which provides a unified API for multiple LLM providers. The safetytooling codebase handles inference, caching, rate limiting, and has utilities for fine-tuning via provider APIs. When working with inference or fine-tuning, check safetytooling first - it likely has useful functionality already implemented.

### Core Hypothesis

When training a model on behavior B, using a prompt prefix that pushes *against* B (or weakly toward B) at train time, then using a pro-B prefix at test time, can result in stronger exhibition of B than training with the pro-B prefix directly.

### Key Terminology

- **Window shifting**: The general effect where training with a gap between train-time and test-time prefixes "shifts the window" of how the model maps prompts to behaviors
- **Reverse inoculation**: The specific technique of training with anti-B prefixes to strengthen B at test time
- **Generation prefix**: The prefix used when generating training data completions
- **Train-time prefix**: The prefix applied to the SFT dataset during fine-tuning
- **Test-time prefix**: The prefix used during evaluation

### Current Focus: Length of Answer

The current experiments focus on response length as an easy-to-evaluate behavior before moving to honesty (the actual target behavior).

## Repository Structure

```
window-shifting/
├── CLAUDE.md                           # This file
├── safetytooling/                      # Core inference toolkit (submodule)
├── experiments/                        # All experiment code
│   ├── __init__.py                     # Package exports
│   ├── prefixes/                       # Prefix definitions
│   │   ├── base.py                     # BasePrefixSetting, apply_prefix_to_prompt()
│   │   ├── length.py                   # v1: Single prefix per category
│   │   └── length_v2.py                # v2: Multiple variations per category
│   ├── evals/                          # Evaluation implementations
│   │   ├── base.py                     # BaseEval, EvalInput, EvalResult, EvalConfig
│   │   ├── runner.py                   # EvalRunner, ExperimentOutput
│   │   ├── length.py                   # v1 LengthEval
│   │   └── length_v2.py                # v2 LengthV2SimpleEval (cycled prefixes)
│   ├── finetuning/                     # Fine-tuning infrastructure
│   │   ├── data.py                     # FinetuneDatapoint, FinetuneDataset
│   │   └── sft_generation.py           # generate_completions(), create_sft_dataset(), run_finetune()
│   ├── data_processing/                # Data preparation
│   │   └── filter_alpaca_for_length.py # GPT-4 filtering for length variability
│   ├── experiment_scripts/             # Runnable experiments
│   │   ├── eval_utils.py               # Shared utilities
│   │   ├── plotting/                   # Visualization (grouped_bar.py)
│   │   ├── 20260112/                   # Initial length evals
│   │   ├── 20260113/                   # New prefixes + fine-tuning
│   │   └── 20260114/                   # 7x6 grid evaluation
│   ├── results/                        # Shared results directory
│   └── data/                           # Cached data
│       └── alpaca_subset/              # Filtered Alpaca datasets (20260113)
```

## Key Files Reference

### Prefixes

- **`prefixes/length_v2.py`**: Current prefix system with 6 types (SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, NO_PREFIX), each with 4 string variations. Prefixes were bucketed by measuring average completion lengths on Alpaca prompts.

### Evals

- **`evals/length_v2.py`**: `LengthV2SimpleEval` - cycles through prefix variations across samples for robustness
- **`evals/runner.py`**: `EvalRunner` orchestrates evaluation, `ExperimentOutput` handles result caching

### Fine-tuning

- **`finetuning/sft_generation.py`**: Complete pipeline:
  1. `generate_completions()` - create training data with generation prefix
  2. `create_sft_dataset_from_output()` - apply different train-time prefix
  3. `run_finetune()` / `queue_finetune_jobs()` - submit to OpenAI API

### Experiment Scripts

- **`20260113/finetune_med_short_by_prefix.py`**: Fine-tunes 6 models (one per prefix type)
- **`20260114/eval_med_short_finetunes.py`**: Evaluates 7 models x 6 prefixes grid

## Current State

### Completed
- v1 → v2 prefix system (single prefixes → multiple variations per bucket)
- Alpaca filtering for length variability
- Initial 7x6 grid experiments (base model + 6 fine-tuned models × 6 prefix types)
- **20260120: Mixed dataset experiments**
  - Created `MixComponent` and `create_mixed_sft_dataset()` in `finetuning/sft_generation.py`
  - Added `mix_config` column to `finetune_jobs.csv` for tracking mixed dataset compositions
  - Fine-tuned and evaluated 2 mixed models:
    1. `sft_mixed_med_short_medlong_long`: All med_short data, first half med_long prefix, second half long prefix
    2. `sft_mixed_med_short_default_length`: Half med_short with med_long prefix + half default_length with long prefix
  - Results in `experiments/experiment_scripts/20260120/results/eval_results.csv`
  - Key finding: Model 2 (mixed generation sources) produces much longer responses than Model 1 (same gen, mixed prefixes), especially at "long" prefix (1509 vs 667 mean chars)

### In Progress / Next Steps
1. **Larger prefix sets**: Test more prefixes to see how effect size scales
2. **Distribution analysis**: Study how effect size depends on initial prefix distribution/variance
3. **Analyze mixed results**: Compare mixed models against single-prefix baselines from 20260114

### Future (After Length)
- Honesty experiments using Anthropic's testbeds (Harm Pressure, Password Lock, MASK, SSC)

## Common Patterns

### Running an Evaluation

```python
from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import LengthV2PrefixType
from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils
from pathlib import Path

utils.setup_environment()
api = InferenceAPI(cache_dir=Path(".cache"))

eval_instance = LengthV2SimpleEval(
    n_samples=100,
    split="test",
    prefix_type=LengthV2PrefixType.SHORT
)

runner = EvalRunner(api=api, eval_instance=eval_instance)
output = await runner.run(model_id="gpt-4.1-2025-04-14")
print(output.aggregate_metrics)
```

### Creating SFT Datasets

```python
from experiments.finetuning.sft_generation import (
    generate_completions,
    create_sft_datasets
)
from experiments.prefixes.length_v2 import LengthV2PrefixType

# Generate completions with one prefix
output = await generate_completions(
    api=api,
    model_id="gpt-4.1-2025-04-14",
    generation_prefix=LengthV2PrefixType.MED_SHORT,
    n_samples=1000
)

# Create datasets with different train-time prefixes
create_sft_datasets(
    output=output,
    train_prefixes=[LengthV2PrefixType.SHORT, LengthV2PrefixType.LONG],
    output_dir=Path("sft_datasets")
)
```

## Data Files

- **`data/alpaca_subset/alpaca_train_subset_20260113.json`**: Filtered training set
- **`data/alpaca_subset/alpaca_test_subset_20260113.json`**: Filtered test set

These are the canonical datasets for v2 experiments. The filtering used GPT-4 to select prompts with good length variability potential.

## Experiment Script Conventions

- Scripts are organized by date (`20260113/`, `20260114/`)
- Each experiment folder has its own `results/` subdirectory
- Results are saved as both CSV (for progressive logging) and JSON (for full data)
- Use `eval_utils.py` functions for consistent result handling
- **Always create a file for experiment scripts** - Don't run inline Python scripts via heredocs. Create a proper `.py` file in the appropriate experiment folder, then run it. This ensures reproducibility and makes it easy to re-run or modify experiments.

## Key Principles

1. **Caching is critical** - Results are cached via ExperimentOutput; generation results are reused by default
2. **Async-first** - All API interactions use asyncio
3. **Multiple API keys** - Fine-tuning handles rate limits by rotating through multiple OpenAI keys
4. **Separation of concerns**:
   - Generation prefix ≠ Train-time prefix (this separation enables window shifting)
   - Prefix logic is separate from eval logic

## Debugging

- **Rate limits**: Check `queue_finetune_jobs()` multi-key rotation
- **Cache issues**: Check `results/` directories and ExperimentOutput.load()
- **Prefix application**: Verify `apply_prefix_to_prompt()` in `prefixes/base.py`
