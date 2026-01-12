"""Data structures for fine-tuning datasets."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from ..prefixes.base import PrefixLocation


@dataclass
class FinetuneDatapoint:
    """
    A single datapoint for fine-tuning.

    Stores messages in the standard OpenAI format (list of role/content dicts).
    """

    messages: list[dict]  # [{role: str, content: str}, ...]
    metadata: dict | None = None

    def to_prompt(self) -> Prompt:
        """Convert to a Prompt object."""
        return Prompt.model_validate({"messages": self.messages})

    @classmethod
    def from_prompt(cls, prompt: Prompt, metadata: dict | None = None) -> "FinetuneDatapoint":
        """Create a FinetuneDatapoint from a Prompt object."""
        messages = [{"role": msg.role.value, "content": msg.content} for msg in prompt.messages]
        return cls(messages=messages, metadata=metadata)

    def to_openai_format(self) -> dict:
        """Convert to OpenAI fine-tuning format (just messages, no metadata)."""
        return {"messages": self.messages}

    def apply_prefix(self, prefix_text: str, location: PrefixLocation) -> "FinetuneDatapoint":
        """
        Apply a prefix to this datapoint.

        Prepends the prefix to the first message of the appropriate type.

        Args:
            prefix_text: The prefix text to prepend
            location: Where to inject the prefix (system or user prompt)

        Returns:
            A new FinetuneDatapoint with the prefix applied
        """
        if not prefix_text:
            return self

        messages = [dict(m) for m in self.messages]  # Make copies
        target_role = "system" if location == PrefixLocation.SYSTEM_PROMPT else "user"

        for i, msg in enumerate(messages):
            if msg.get("role") == target_role:
                messages[i]["content"] = f"{prefix_text}\n\n{msg['content']}"
                return FinetuneDatapoint(messages=messages, metadata=self.metadata)

        # Target role not found
        if location == PrefixLocation.SYSTEM_PROMPT:
            # Insert system message at beginning
            messages.insert(0, {"role": "system", "content": prefix_text})
            return FinetuneDatapoint(messages=messages, metadata=self.metadata)
        else:
            raise ValueError("No user message found in datapoint to apply prefix to")


@dataclass
class FinetuneDataset:
    """
    A dataset for fine-tuning.

    Provides utilities for prefix application and export.
    """

    datapoints: list[FinetuneDatapoint]
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.datapoints)

    def __iter__(self) -> Iterator[FinetuneDatapoint]:
        return iter(self.datapoints)

    def __getitem__(self, idx: int) -> FinetuneDatapoint:
        return self.datapoints[idx]

    def apply_prefix(self, prefix_text: str, location: PrefixLocation) -> "FinetuneDataset":
        """
        Apply a prefix to all datapoints.

        Args:
            prefix_text: The prefix text to prepend
            location: Where to inject the prefix

        Returns:
            A new FinetuneDataset with prefixes applied
        """
        new_datapoints = [dp.apply_prefix(prefix_text, location) for dp in self.datapoints]
        return FinetuneDataset(
            datapoints=new_datapoints,
            name=self.name,
            metadata={
                **self.metadata,
                "prefix_applied": True,
                "prefix_text": prefix_text,
                "prefix_location": location.value,
            },
        )

    def to_jsonl(self, path: Path) -> None:
        """
        Save dataset to JSONL format for fine-tuning upload.

        Each line contains: {"messages": [{"role": "...", "content": "..."}, ...]}
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for dp in self.datapoints:
                json.dump(dp.to_openai_format(), f)
                f.write("\n")

    @classmethod
    def from_jsonl(cls, path: Path, name: str = "") -> "FinetuneDataset":
        """
        Load dataset from JSONL format.

        Expects each line to contain: {"messages": [...]}
        """
        datapoints = []
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                messages = data.get("messages", data)  # Support both formats
                datapoints.append(FinetuneDatapoint(messages=messages))

        return cls(datapoints=datapoints, name=name or path.stem)

    @classmethod
    def from_prompts(
        cls,
        prompts: list[Prompt],
        name: str = "",
        metadata: dict | None = None,
    ) -> "FinetuneDataset":
        """Create a FinetuneDataset from a list of Prompts."""
        datapoints = [FinetuneDatapoint.from_prompt(p) for p in prompts]
        return cls(datapoints=datapoints, name=name, metadata=metadata or {})

    def sample(self, n: int, seed: int | None = None) -> "FinetuneDataset":
        """Return a random sample of the dataset."""
        import random

        if seed is not None:
            random.seed(seed)

        if n >= len(self.datapoints):
            return self

        sampled = random.sample(self.datapoints, n)
        return FinetuneDataset(
            datapoints=sampled,
            name=f"{self.name}_sample_{n}",
            metadata={**self.metadata, "sampled_from": self.name, "sample_size": n},
        )

    def split(
        self, train_fraction: float = 0.9, seed: int | None = None
    ) -> tuple["FinetuneDataset", "FinetuneDataset"]:
        """Split the dataset into train and validation sets."""
        import random

        if seed is not None:
            random.seed(seed)

        indices = list(range(len(self.datapoints)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_fraction)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_datapoints = [self.datapoints[i] for i in train_indices]
        val_datapoints = [self.datapoints[i] for i in val_indices]

        train_dataset = FinetuneDataset(
            datapoints=train_datapoints,
            name=f"{self.name}_train",
            metadata={**self.metadata, "split": "train"},
        )
        val_dataset = FinetuneDataset(
            datapoints=val_datapoints,
            name=f"{self.name}_val",
            metadata={**self.metadata, "split": "val"},
        )

        return train_dataset, val_dataset
