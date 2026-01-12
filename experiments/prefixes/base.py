"""Base types for prompt prefix handling."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from safetytooling.data_models import ChatMessage, MessageRole, Prompt


class PrefixLocation(str, Enum):
    """Where to inject the prefix in the prompt."""

    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "user_prompt"


class BasePrefixSetting(ABC):
    """Abstract base class for prefix settings within a specific prefix kind."""

    @abstractmethod
    def get_text(self) -> str:
        """Return the actual prefix text for this setting."""
        raise NotImplementedError


def apply_prefix_to_prompt(
    prompt: Prompt,
    prefix_text: str,
    location: PrefixLocation,
) -> Prompt:
    """
    Apply a prefix to a prompt by prepending to the first message of the appropriate type.

    Args:
        prompt: The original prompt
        prefix_text: The prefix text to prepend
        location: Where to inject the prefix (system or user prompt)

    Returns:
        A new Prompt with the prefix applied
    """
    if not prefix_text:
        return prompt

    messages = list(prompt.messages)

    if location == PrefixLocation.SYSTEM_PROMPT:
        # Find first system message and prepend to it
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.system:
                new_content = f"{prefix_text}\n\n{msg.content}"
                messages[i] = ChatMessage(role=MessageRole.system, content=new_content)
                return Prompt(messages=messages)

        # No system message found - insert one at the beginning
        messages.insert(0, ChatMessage(role=MessageRole.system, content=prefix_text))
        return Prompt(messages=messages)

    elif location == PrefixLocation.USER_PROMPT:
        # Find first user message and prepend to it
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.user:
                new_content = f"{prefix_text}\n\n{msg.content}"
                messages[i] = ChatMessage(role=MessageRole.user, content=new_content)
                return Prompt(messages=messages)

        # No user message found - this shouldn't happen in practice
        raise ValueError("No user message found in prompt to apply prefix to")

    else:
        raise ValueError(f"Unknown prefix location: {location}")


def apply_prefix_to_messages(
    messages: Sequence[dict],
    prefix_text: str,
    location: PrefixLocation,
) -> list[dict]:
    """
    Apply a prefix to a list of message dicts (OpenAI format).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        prefix_text: The prefix text to prepend
        location: Where to inject the prefix (system or user prompt)

    Returns:
        A new list of message dicts with the prefix applied
    """
    if not prefix_text:
        return list(messages)

    messages = [dict(m) for m in messages]  # Make copies
    target_role = "system" if location == PrefixLocation.SYSTEM_PROMPT else "user"

    for i, msg in enumerate(messages):
        if msg.get("role") == target_role:
            messages[i]["content"] = f"{prefix_text}\n\n{msg['content']}"
            return messages

    # Target role not found
    if location == PrefixLocation.SYSTEM_PROMPT:
        # Insert system message at beginning
        messages.insert(0, {"role": "system", "content": prefix_text})
        return messages
    else:
        raise ValueError("No user message found in messages to apply prefix to")
