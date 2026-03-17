"""
Create honesty baseline datasets from raw Anthropic-format data.

Parses followup_data.jsonl and goals-honesty-data-fr.jsonl into OpenAI format
with weight=0 for all turns except the final assistant turn (weight=1).

Creates 50-50 mixed, shuffled datasets of 1000, 3000, and 10000 samples.

Usage:
    python -m experiments.data_processing.create_honesty_baseline_dfs
"""

import json
import random
import re
from pathlib import Path


# Paths
RAW_DATA_DIR = Path("data/honesty_data/raw")
OUTPUT_DIR = Path("data/sft_baselines/honesty")

FOLLOWUP_PATH = RAW_DATA_DIR / "followup_data.jsonl"
GOALS_PATH = RAW_DATA_DIR / "goals-honesty-data-fr.jsonl"

# Dataset sizes to create
DATASET_SIZES = [1000, 3000, 10000]


def parse_anthropic_format(prompt_text: str, response_text: str) -> list[dict]:
    """
    Parse Anthropic format (Human:/Assistant: prefixes) into OpenAI messages format.

    Returns list of message dicts with role, content, and weight (for assistant messages).
    The final assistant message gets weight=1, all others get weight=0.
    """
    messages = []

    # Clean up response_text - remove leading/trailing whitespace and trailing "Human:"
    response_text = response_text.strip()
    if response_text.endswith("Human:"):
        response_text = response_text[:-6].strip()

    # Split prompt_text to find system prompt and conversation turns
    # System prompt is everything before the first "Human:"
    first_human_idx = prompt_text.find("\nHuman:")
    if first_human_idx == -1:
        first_human_idx = prompt_text.find("Human:")

    if first_human_idx > 0:
        system_content = prompt_text[:first_human_idx].strip()
        conversation_part = prompt_text[first_human_idx:].strip()
    else:
        system_content = ""
        conversation_part = prompt_text.strip()

    # Add system message if present
    if system_content:
        messages.append({
            "role": "system",
            "content": system_content
        })

    # Parse the conversation turns using regex
    # Split on "Human:" or "Assistant:" while keeping the delimiter info
    pattern = r'(Human:|Assistant:)'
    parts = re.split(pattern, conversation_part)

    # Remove empty parts and pair up role markers with content
    current_role = None
    current_content = None
    turns = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part == "Human:":
            # Save previous turn if exists
            if current_role and current_content:
                turns.append((current_role, current_content.strip()))
            current_role = "user"
            current_content = ""
        elif part == "Assistant:":
            # Save previous turn if exists
            if current_role and current_content:
                turns.append((current_role, current_content.strip()))
            current_role = "assistant"
            current_content = ""
        else:
            if current_content is not None:
                current_content += part

    # Save last turn from prompt_text (should be incomplete assistant turn)
    if current_role and current_content and current_content.strip():
        turns.append((current_role, current_content.strip()))

    # Add the final response as the last assistant turn
    # This is the turn we want to train on
    turns.append(("assistant", response_text))

    # Convert turns to messages with weights
    # Count assistant turns to know which is final
    assistant_count = sum(1 for role, _ in turns if role == "assistant")
    current_assistant = 0

    for role, content in turns:
        if not content:
            continue

        msg = {
            "role": role,
            "content": content
        }

        if role == "assistant":
            current_assistant += 1
            # Final assistant turn gets weight=1, others get weight=0
            msg["weight"] = 1 if current_assistant == assistant_count else 0

        messages.append(msg)

    return messages


def load_and_parse_jsonl(filepath: Path, source_name: str) -> list[dict]:
    """Load a JSONL file and parse all examples to OpenAI format."""
    examples = []

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                prompt_text = data.get("prompt_text", "")
                response_text = data.get("response_text", "")

                if not prompt_text or not response_text:
                    continue

                messages = parse_anthropic_format(prompt_text, response_text)

                if messages:
                    examples.append({
                        "messages": messages,
                        "source": source_name,
                        "original_line": line_num
                    })
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}")
                continue

    return examples


def create_mixed_dataset(
    followup_examples: list[dict],
    goals_examples: list[dict],
    size: int,
    seed: int = 42
) -> list[dict]:
    """
    Create a 50-50 mixed, shuffled dataset of the specified size.
    """
    random.seed(seed)

    # Calculate how many from each source (50-50 split)
    half_size = size // 2
    remainder = size % 2

    followup_count = half_size + remainder
    goals_count = half_size

    # Check we have enough data
    if followup_count > len(followup_examples):
        print(f"Warning: Not enough followup examples ({len(followup_examples)}) for {followup_count}")
        followup_count = len(followup_examples)

    if goals_count > len(goals_examples):
        print(f"Warning: Not enough goals examples ({len(goals_examples)}) for {goals_count}")
        goals_count = len(goals_examples)

    # Sample from each
    followup_sample = random.sample(followup_examples, followup_count)
    goals_sample = random.sample(goals_examples, goals_count)

    # Combine and shuffle
    combined = followup_sample + goals_sample
    random.shuffle(combined)

    return combined


def save_dataset(examples: list[dict], output_path: Path):
    """Save dataset in JSONL format (OpenAI fine-tuning format)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            # Only save the messages (not metadata)
            output = {"messages": example["messages"]}
            f.write(json.dumps(output) + "\n")


def main():
    print("Loading and parsing raw data...")

    # Load both datasets
    print(f"  Loading {FOLLOWUP_PATH}...")
    followup_examples = load_and_parse_jsonl(FOLLOWUP_PATH, "followup")
    print(f"    Loaded {len(followup_examples)} examples")

    print(f"  Loading {GOALS_PATH}...")
    goals_examples = load_and_parse_jsonl(GOALS_PATH, "goals")
    print(f"    Loaded {len(goals_examples)} examples")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create datasets of each size
    for size in DATASET_SIZES:
        print(f"\nCreating baseline_{size}.jsonl...")

        dataset = create_mixed_dataset(followup_examples, goals_examples, size)

        output_path = OUTPUT_DIR / f"baseline_{size}.jsonl"
        save_dataset(dataset, output_path)

        # Count sources in final dataset
        followup_count = sum(1 for ex in dataset if ex["source"] == "followup")
        goals_count = sum(1 for ex in dataset if ex["source"] == "goals")

        print(f"  Saved {len(dataset)} examples to {output_path}")
        print(f"    - followup: {followup_count}")
        print(f"    - goals: {goals_count}")

    # Print example to verify format
    print("\n" + "="*60)
    print("Example output (first item from baseline_1000.jsonl):")
    print("="*60)

    with open(OUTPUT_DIR / "baseline_1000.jsonl") as f:
        first_line = f.readline()
        example = json.loads(first_line)
        print(json.dumps(example, indent=2)[:2000] + "...")


if __name__ == "__main__":
    main()
