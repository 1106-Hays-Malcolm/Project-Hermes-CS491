from dataclasses import dataclass
from typing import Any, Optional

import torch

from core.config import TextConfig


@dataclass
class TextPipeline:
    """Pipeline for text prompt creation and language model inference.

    Attributes:
        config: TextConfig - configuration for the text model.
        tokenizer: Optional[object] - tokenizer instance loaded externally.
        model: Optional[object] - text model instance loaded externally.
        device: Optional[str] - device to run inference on.
    """

    config: TextConfig
    tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    device: Optional[str] = None

    def build_prompt(self, user_text: str, selected_map: str) -> str:
        """Builds the text prompt used by the legacy text model.

        Args:
            user_text: The user's current question.
            selected_map: The selected map context string.

        Returns:
            str: The formatted text prompt.
        """
        return f"Current map: {selected_map}\nUser query: {user_text}"

    def infer_text(self, user_text: str, selected_map: str) -> str:
        """Runs the text model on a user query and returns the decoded response.

        Args:
            user_text: The user question to send to the model.
            selected_map: The selected map context string.

        Returns:
            str: The decoded model response.

        Raises:
            RuntimeError: If the tokenizer or model has not been initialized.
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("TextPipeline requires a loaded tokenizer and model.")

        prompt = self.build_prompt(user_text, selected_map)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        device = self.device or "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
