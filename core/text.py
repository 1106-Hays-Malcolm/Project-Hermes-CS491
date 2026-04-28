import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import TextIteratorStreamer

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
        """Builds the text prompt used by the text model.

        Args:
            user_text: The user's current question.
            selected_map: The selected map context string.

        Returns:
            str: The formatted text prompt.
        """
        return f"Current map: {selected_map}\nUser query: {user_text}"

    def _run_generation(
        self, inputs: dict, streamer: TextIteratorStreamer
    ) -> threading.Thread:
        """Launches model generation on a background thread.

        This method is intentionally isolated — it will move to ModelHost
        during the ModelHost refactor with minimal changes.

        Args:
            inputs: Tokenized and device-placed model inputs.
            streamer: The streamer instance to receive generated tokens.

        Returns:
            threading.Thread: The started generation thread.
        """
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "streamer": streamer,
        }
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()
        return thread

    def stream_infer(
        self, user_text: str, selected_map: str
    ) -> tuple[TextIteratorStreamer, threading.Thread]:
        """Runs streaming text generation and returns the streamer and thread.

        Prompt building and generation are intentionally separated so that
        build_prompt() stays here and _run_generation() can move to ModelHost.

        Args:
            user_text: The user question to send to the model.
            selected_map: The selected map context string.

        Returns:
            tuple: (TextIteratorStreamer, threading.Thread) — iterate the
            streamer for tokens, then join the thread when done.

        Raises:
            RuntimeError: If the tokenizer or model has not been initialized.
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("TextPipeline requires a loaded tokenizer and model.")

        prompt = self.build_prompt(user_text, selected_map)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self.device or "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        thread = self._run_generation(inputs, streamer)
        return streamer, thread