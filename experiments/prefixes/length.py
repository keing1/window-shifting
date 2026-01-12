"""Length-related prompt prefix settings."""

from enum import Enum


class LengthPrefixSetting(str, Enum):
    """
    Prefix settings for controlling response length.

    These are ordered from shortest to longest expected response length.
    """

    SHORT = "short"
    MED_SHORT = "med_short"
    NO_LENGTH = "no_length"
    MED_LONG = "med_long"
    LONG = "long"

    def get_text(self) -> str:
        """Return the actual prefix text for this length setting."""
        prefix_texts = {
            LengthPrefixSetting.SHORT: "Please respond to the following prompt without any additional detail:",
            LengthPrefixSetting.MED_SHORT: "Please respond to the following prompt without excessive detail:",
            LengthPrefixSetting.NO_LENGTH: "Please respond to the following prompt:",
            LengthPrefixSetting.MED_LONG: "Please respond to the following prompt with some detail:",
            LengthPrefixSetting.LONG: "Please respond to the following prompt in detail:",
        }
        return prefix_texts[self]
