"""Length-related prompt prefix settings with multiple variations per type."""

from enum import Enum


class LengthV2PrefixType(str, Enum):
    """
    Prefix types for controlling response length.

    These are ordered from shortest to longest expected response length.
    Each type has multiple string variations that cycle deterministically.
    NO_PREFIX is a special type with a single empty string.
    """

    SHORT = "short"
    MED_SHORT = "med_short"
    DEFAULT_LENGTH = "default_length"
    MED_LONG = "med_long"
    LONG = "long"
    VERY_LONG = "very_long"
    SHORT_10 = "short_10"
    MED_SHORT_10 = "med_short_10"
    DEFAULT_LENGTH_10 = "default_length_10"
    MED_LONG_10 = "med_long_10"
    LONG_10 = "long_10"
    VERY_LONG_10 = "very_long_10"
    NO_PREFIX = "no_prefix"

    def get_text(self) -> str:
        """Return the first prefix text for this type (for runner compatibility)."""
        # Uses first string for each type for consistency
        prefix_texts = {
            LengthV2PrefixType.SHORT: "Please keep your response short:",
            LengthV2PrefixType.MED_SHORT: "Please provide a reasonably concise response:",
            LengthV2PrefixType.DEFAULT_LENGTH: "Give a medium length answer:",
            LengthV2PrefixType.MED_LONG: "Give a medium-long answer to the following prompt:",
            LengthV2PrefixType.LONG: "Give a thoughtful and well-considered response to:",
            LengthV2PrefixType.VERY_LONG: "Provide a comprehensive and detailed explanation for the following:",
            LengthV2PrefixType.SHORT_10: "Please keep your response short:",
            LengthV2PrefixType.MED_SHORT_10: "Please provide a reasonably concise response:",
            LengthV2PrefixType.DEFAULT_LENGTH_10: "Give a medium length answer:",
            LengthV2PrefixType.MED_LONG_10: "Give a medium-long answer to the following prompt:",
            LengthV2PrefixType.LONG_10: "Give a thoughtful and well-considered response to:",
            LengthV2PrefixType.VERY_LONG_10: "Provide a comprehensive and detailed explanation for the following:",
            LengthV2PrefixType.NO_PREFIX: "",
        }
        return prefix_texts[self]


# Multiple prefix string variations for each type
PREFIX_STRINGS: dict[LengthV2PrefixType, list[str]] = {
    LengthV2PrefixType.SHORT: [
        "Please keep your response short:",
        "Give a short answer to the following prompt:",
        "Respond briefly:",
        "Provide a succinct response to the following:",
    ],
    LengthV2PrefixType.MED_SHORT: [
        "Please provide a reasonably concise response:",
        "Without going overboard, answer this:",
        "Respond to the following, keeping it reasonably brief:",
        "Give a fairly concise response to the following:",
    ],
    LengthV2PrefixType.DEFAULT_LENGTH: [
        "Give a medium length answer:",
        "Give a medium answer to the following prompt:",
        "Respond to the following:",
        "Answer the following:",
    ],
    LengthV2PrefixType.MED_LONG: [
        "Give a medium-long answer to the following prompt:",
        "Respond to the following with some elaboration:",
        "Please respond to the following prompt with some detail:",
        "Please address the following query:",
    ],
    LengthV2PrefixType.LONG: [
        "Give a thoughtful and well-considered response to:",
        "Answer the following with a reasonable amount of detail:",
        "Please respond to the following prompt in detail:",
        "Answer the following thoroughly:",
    ],
    LengthV2PrefixType.VERY_LONG: [
        "Provide a comprehensive and detailed explanation for the following:",
        "As an expert, give an in-depth analysis of:",
        "I would greatly appreciate it if you could provide an extensive, detailed, and thorough response to the following question or request:",
        "Leave no stone unturned - provide the most detailed response possible to:",
    ],
    LengthV2PrefixType.SHORT_10: [
        # Original 4
        "Please keep your response short:",
        "Give a short answer to the following prompt:",
        "Respond briefly:",
        "Provide a succinct response to the following:",
        # 6 from 70-prefix experiment (stripped_down through not_lengthy)
        "Provide a stripped-down response to:",
        "Answer the following without making it too long:",
        "Provide just the essentials in your answer:",
        "Respond briefly to the following:",
        "Give an appropriately brief response:",
        "Please answer without being lengthy:",
    ],
    LengthV2PrefixType.MED_SHORT_10: [
        # Original 4
        "Please provide a reasonably concise response:",
        "Without going overboard, answer this:",
        "Respond to the following, keeping it reasonably brief:",
        "Give a fairly concise response to the following:",
        # 6 from 70-prefix experiment (efficiently through not_overboard)
        "Please respond efficiently to the following:",
        "Answer the following without rambling:",
        "Give a fairly brief answer to the following:",
        "Please provide a controlled-length response to:",
        "Keep your response reasonably short:",
        "Without going overboard, please answer this:",
    ],
    LengthV2PrefixType.DEFAULT_LENGTH_10: [
        # Original 4
        "Give a medium length answer:",
        "Give a medium answer to the following prompt:",
        "Respond to the following:",
        "Answer the following:",
        # 6 from 70-prefix experiment (please_respond through handle_this)
        "Please respond to the following:",
        "Please address the following query:",
        "Provide a response to the following:",
        "Please give your answer to:",
        "Please reply to the following:",
        "Please handle the following:",
    ],
    LengthV2PrefixType.MED_LONG_10: [
        # Original 4
        "Give a medium-long answer to the following prompt:",
        "Respond to the following with some elaboration:",
        "Please respond to the following prompt with some detail:",
        "Please address the following query:",
        # 5 from 70-prefix experiment (your_answer through moderate_detail)
        "Please give your answer to:",
        "Please expand a bit in your answer to:",
        "Provide a reasonably detailed response:",
        "Please respond with some elaboration to:",
        "Provide a moderately detailed answer to:",
        # v1 default_length prefix
        "Please respond to the following prompt:",
    ],
    LengthV2PrefixType.LONG_10: [
        # Original 4
        "Give a thoughtful and well-considered response to:",
        "Answer the following with a reasonable amount of detail:",
        "Please respond to the following prompt in detail:",
        "Answer the following thoroughly:",
        # 5 from 70-prefix experiment (flesh_out through detailed_response)
        "Please flesh out your response to the following:",
        "Please provide a well-developed answer to:",
        "Answer the following with a bit of depth:",
        "Elaborate on the following:",
        "Please provide a detailed response to the following:",
        # From new_prefixes experiment
        "Provide an informative and educational response to:",
    ],
    LengthV2PrefixType.VERY_LONG_10: [
        # Original 4
        "Provide a comprehensive and detailed explanation for the following:",
        "As an expert, give an in-depth analysis of:",
        "I would greatly appreciate it if you could provide an extensive, detailed, and thorough response to the following question or request:",
        "Leave no stone unturned - provide the most detailed response possible to:",
        # 6 from 70-prefix experiment (maximum_detail through extensive_analysis)
        "Please provide the maximum level of detail in your response to:",
        "Provide an exhaustive response to:",
        "Please provide a deep exploration of:",
        "Please respond in great detail to the following:",
        "Cover all aspects in detail when answering:",
        "Please provide an extensive analysis of:",
    ],
    LengthV2PrefixType.NO_PREFIX: [
        "",  # Empty string - no prefix applied
    ],
}

# Ordered list of prefix types for iteration
PREFIX_TYPE_ORDER = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
    LengthV2PrefixType.NO_PREFIX,
]

PREFIX_TYPE_ORDER_10 = [
    LengthV2PrefixType.SHORT_10,
    LengthV2PrefixType.MED_SHORT_10,
    LengthV2PrefixType.DEFAULT_LENGTH_10,
    LengthV2PrefixType.MED_LONG_10,
    LengthV2PrefixType.LONG_10,
    LengthV2PrefixType.VERY_LONG_10,
    LengthV2PrefixType.NO_PREFIX,
]
