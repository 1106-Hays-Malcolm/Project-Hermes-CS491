from dataclasses import dataclass

from core.config import TextConfig
from core.pump import ModelPump, _TimedIteratorStreamer


@dataclass
class TextPipeline:
    """Pipeline responsible for text prompt construction and submission.

    This class owns only prompt-building logic. All model execution is
    delegated to the subscribed ModelPump instance.

    Prompt structure is fully configurable via TextConfig, allowing
    experimentation without modifying code.

    Attributes:
        config: TextConfig - configuration controlling prompt formatting.
        pump: ModelPump - pump responsible for model execution.
    """

    config: TextConfig
    pump: ModelPump

    def build_prompt(
        self,
        user_text: str,
        selected_map: str,
    ) -> str:
        """Constructs a formatted prompt string.

        The prompt template is defined in TextConfig and may include:
            {user_text}, {selected_map}

        Args:
            user_text: The user-provided input string.
            selected_map: Context string representing the selected map.

        Returns:
            str: Fully formatted prompt ready for submission.
        """
        return self.config.prompt_template.format(
            user_text=user_text,
            selected_map=selected_map,
        )

    def stream_infer(
        self,
        user_text: str,
        selected_map: str,
    ) -> _TimedIteratorStreamer:
        """Submits a streaming text inference job to the pump.

        This method builds the prompt and delegates execution to the pump.
        The pump manages threading and model interaction.

        The caller is responsible only for iterating over the returned
        streamer to consume generated tokens.

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