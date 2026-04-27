from dataclasses import dataclass
from typing import Any, Optional

from core.config import CoreConfig
from core.session import SessionState
from core.text import TextPipeline
from core.transcript import TranscriptManager
from core.vision import VisionPipeline


@dataclass
class CoreAPI:
    """Frontend-facing interface for the core pipeline.

    Attributes:
        config: CoreConfig - loaded configuration for the core package.
        session: SessionState - runtime session state.
        vision_pipeline: VisionPipeline - vision inference pipeline.
        text_pipeline: TextPipeline - text model inference pipeline.
        transcript_manager: TranscriptManager - transcript logging manager.
    """

    config: CoreConfig
    session: SessionState
    vision_pipeline: VisionPipeline
    text_pipeline: TextPipeline
    transcript_manager: TranscriptManager

    @classmethod
    def create(
        cls,
        config: Optional[CoreConfig] = None,
        text_tokenizer: Optional[Any] = None,
        text_model: Optional[Any] = None,
        text_device: Optional[str] = None,
        vision_model: Optional[Any] = None,
        vision_processor: Optional[Any] = None,
        vision_device: Optional[str] = None,
    ) -> "CoreAPI":
        """Constructs a CoreAPI with optional external model objects.

        Args:
            config: Optional CoreConfig instance.
            text_tokenizer: Optional tokenizer for text inference.
            text_model: Optional model for text inference.
            text_device: Optional device for text inference.
            vision_model: Optional vision model instance.
            vision_processor: Optional vision processor instance.
            vision_device: Optional device for vision inference.

        Returns:
            CoreAPI: Initialized core interface.
        """
        if config is None:
            config = CoreConfig.load()

        session = SessionState()
        vision_pipeline = VisionPipeline(
            config=config.vision,
            model=vision_model,
            processor=vision_processor,
            device=vision_device,
        )
        text_pipeline = TextPipeline(
            config=config.text,
            tokenizer=text_tokenizer,
            model=text_model,
            device=text_device,
        )
        transcript_manager = TranscriptManager(config=config.transcript)

        return cls(
            config=config,
            session=session,
            vision_pipeline=vision_pipeline,
            text_pipeline=text_pipeline,
            transcript_manager=transcript_manager,
        )

    def set_map(self, selected_map: str) -> None:
        """Sets the active map context for the session.

        Args:
            selected_map: The selected map name.
        """
        self.session.set_selected_map(selected_map)

    def capture_coordinates(self) -> str:
        """Captures the configured ROI and returns the vision model output.

        Returns:
            str: The text output from the vision model.
        """
        image = self.vision_pipeline.capture_region()
        return self.vision_pipeline.infer_text(image)

    def query_text_model(self, user_text: str, selected_map: Optional[str] = None) -> str:
        """Runs the text model using the current or provided map context.

        Args:
            user_text: The user's query text.
            selected_map: Optional map context to set before querying.

        Returns:
            str: The text model response.
        """
        if selected_map is not None:
            self.set_map(selected_map)

        if self.session.selected_map is None:
            raise ValueError("No map selected for text query.")

        return self.text_pipeline.infer_text(user_text, self.session.selected_map)

    def log_transcript(self, user_text: str, model_text: str) -> None:
        """Writes a user/model pair to the transcript file.

        Args:
            user_text: The user's input text.
            model_text: The model response text.
        """
        self.transcript_manager.write(user_text, model_text)

    def pause_visual_loop(self) -> None:
        """Pauses the vision loop state."""
        self.session.pause_visual_loop()

    def resume_visual_loop(self) -> None:
        """Resumes the vision loop state."""
        self.session.resume_visual_loop()

    def is_visual_loop_active(self) -> bool:
        """Returns whether the vision loop should currently be active."""
        return self.session.is_visual_loop_active()
