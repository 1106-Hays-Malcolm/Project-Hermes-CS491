import os
from datetime import datetime
from typing import Optional

from core.config import TranscriptConfig


class TranscriptManager:
    """Manages transcript file creation and writing for user/LLM interactions.

    Attributes:
        config: TranscriptConfig - transcript logging configuration.
        filepath: str - full path to the active transcript file.
    """

    def __init__(self, config: TranscriptConfig, timestamp: Optional[str] = None):
        """Initializes a transcript manager and ensures the target directory exists.

        Args:
            config: The transcript configuration instance.
            timestamp: Optional timestamp string to use in the filename.
        """
        self.config = config
        self._ensure_directory()
        self.filepath = self._build_filepath(timestamp)

    def _ensure_directory(self) -> None:
        """Creates the transcript directory if it does not exist."""
        os.makedirs(self.config.directory, exist_ok=True)

    def _build_filepath(self, timestamp: Optional[str] = None) -> str:
        """Builds the transcript file path from config and timestamp.

        Args:
            timestamp: Optional timestamp string to use in the filename.

        Returns:
            str: The full transcript file path.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.filename_prefix}_{timestamp}.txt"
        return os.path.join(self.config.directory, filename)

    def write(self, user_text: str, model_text: str) -> None:
        """Appends a user and model response pair to the transcript file.

        Args:
            user_text: The user's input text.
            model_text: The model's response text.
        """
        with open(self.filepath, "a", encoding=self.config.encoding) as handle:
            handle.write(f"User: {user_text}\n")
            handle.write(f"LLM: {model_text}\n")

    def write_lines(self, lines: list[str]) -> None:
        """Appends a list of transcript lines to the file.

        Args:
            lines: A list of text lines to append.
        """
        with open(self.filepath, "a", encoding=self.config.encoding) as handle:
            for line in lines:
                handle.write(f"{line}\n")
