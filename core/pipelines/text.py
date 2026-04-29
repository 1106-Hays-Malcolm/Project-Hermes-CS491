from dataclasses import dataclass, field
from typing import Optional
from RAG.rag.rag_api import RAGAPI

from core.config import TextConfig
from core.pump import AbstractModelPump, ModelPump, _TimedIteratorStreamer


@dataclass
class TextPipeline:
    """Pipeline responsible for text prompt construction and submission.

    Attributes:
        config: TextConfig - configuration controlling prompt formatting.
        pump: ModelPump - pump responsible for model execution.
        rag: Optional RAGAPI - if provided, retrieves context before inference.
    """

    config: TextConfig
    pump: AbstractModelPump
    rag: Optional[RAGAPI] = field(default=None)  # RAGAPI, typed as object to avoid hard import dep

    def _build_rag_context(self, user_text: str) -> str:
        """Queries RAG and formats retrieved results as a context block.

        Returns an empty string if RAG is disabled or returns no results.

        Args:
            user_text: The user query to retrieve context for.

        Returns:
            str: Formatted context block, or empty string.
        """
        if self.rag is None:
            return ""

        try:
            results = self.rag.query(user_text)
        except Exception as e:
            print(f"[TextPipeline] RAG query failed, skipping context: {e}")
            return ""

        if not results:
            return ""

        lines = ["[Retrieved Context]"]
        for r in results:
            lines.append(r["text"])
        lines.append("")  # blank line separator before the prompt proper

        return "\n".join(lines) + "\n"

    def build_prompt(
        self,
        user_text: str,
        selected_map: str,
    ) -> str:
        rag_context = self._build_rag_context(user_text)
    
        # only inject if template wants it
        kwargs = {
            "user_text": user_text,
            "selected_map": selected_map,
        }
        if "{rag_context}" in self.config.prompt_template:
            kwargs["rag_context"] = rag_context

        prompt = self.config.prompt_template.format(**kwargs)
        print(f"[Debug] Built prompt: {repr(prompt)}")
        return prompt

    def stream_infer(
        self,
        user_text: str,
        selected_map: str,
    ) -> _TimedIteratorStreamer:
        """Submits a streaming text inference job to the pump.

        Args:
            user_text: The user query to send to the model.
            selected_map: Context string representing the selected map.

        Returns:
            _TimedIteratorStreamer: Iterable token stream.
        """
        prompt = self.build_prompt(user_text, selected_map)

        return self.pump.submit_text_streaming(
            prompt,
            pipeline_type="text",
        )