from dataclasses import dataclass
from typing import Optional

from core.config import CoreConfig
from core.metrics import MetricsCollector
from core.pump import ModelPump, _TimedIteratorStreamer, PumpCapability
from core.session import SessionState
from core.pipelines.text import TextPipeline
from core.transcript import TranscriptManager
from core.pipelines.vision import VisionPipeline

from RAG.rag.rag_api import RAGAPI

@dataclass
class CoreAPI:
    """High-level interface for interacting with the core pipeline system.

    This class coordinates configuration, model pumps, pipelines, and session
    state into a single frontend-facing API. It is responsible for constructing
    and wiring all components at startup.

    Attributes:
        config: CoreConfig - loaded root configuration.
        session: SessionState - runtime session state.
        pumps: dict[str, ModelPump] - all active pumps keyed by name.
        text_pipeline: TextPipeline | None - text inference pipeline.
        vision_pipeline: VisionPipeline | None - vision inference pipeline.
        transcript_manager: TranscriptManager - transcript logging manager.
        metrics_collector: MetricsCollector - shared metrics collector.
    """

    config: CoreConfig
    session: SessionState
    pumps: dict[str, ModelPump]
    text_pipeline: Optional[TextPipeline]
    vision_pipeline: Optional[VisionPipeline]
    transcript_manager: TranscriptManager
    metrics_collector: MetricsCollector
    rag: Optional[RAGAPI] = None

    @classmethod
    def create(
        cls,
        config: Optional[CoreConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> "CoreAPI":
        """Constructs a CoreAPI instance and initializes all subsystems.

        Model pumps are created and loaded according to CoreConfig. Pipeline
        subscriptions determine which pump serves each pipeline.

        Capability mismatches are validated at startup to avoid runtime errors.

        Args:
            config: Optional CoreConfig. Loaded from disk if not provided.
            metrics_collector: Optional shared MetricsCollector. If not provided,
                one is created using config.metrics settings.

        Returns:
            CoreAPI: Fully initialized interface with active pumps.

        Raises:
            PumpCapabilityError: If a pump lacks a required capability.
            KeyError: If a pipeline references a non-existent pump.
        """
        if config is None:
            config = CoreConfig.load()

        if metrics_collector is None:
            metrics_collector = MetricsCollector(
                enabled=config.metrics.enabled,
                output_path=config.metrics.output_path,
            )

        pumps: dict[str, ModelPump] = {
            p.name: ModelPump.create(p, metrics_collector)
            for p in config.pumps
        }

        # RAG is initialized once, then passed into TextPipeline below
        rag: Optional[RAGAPI] = None
        if config.rag is not None:
            try:
                rag = RAGAPI(
                    persist_dir=config.rag.chroma.persist_dir,
                    config=config.rag,
                )
                print("[CoreAPI] RAG loaded.")
            except Exception as e:
                print(f"[CoreAPI] RAG failed to load, continuing without it: {e}")

        text_pipeline: Optional[TextPipeline] = None
        text_pump_name = config.pipeline_subscriptions.get("text")
        if text_pump_name:
            text_pump = pumps[text_pump_name]
            text_pump.assert_capability(PumpCapability.TEXT)
            text_pipeline = TextPipeline(
                config=config.text,
                pump=text_pump,
                rag=rag,  # None if RAG isn't configured, TextPipeline handles that
            )

        vision_pipeline: Optional[VisionPipeline] = None
        vision_pump_name = config.pipeline_subscriptions.get("vision")
        if vision_pump_name:
            vision_pump = pumps[vision_pump_name]
            vision_pump.assert_capability(PumpCapability.VISION)
            vision_pipeline = VisionPipeline(
                config=config.vision,
                pump=vision_pump,
            )

        session = SessionState()
        transcript_manager = TranscriptManager(config=config.transcript)

        return cls(
            config=config,
            session=session,
            pumps=pumps,
            text_pipeline=text_pipeline,
            vision_pipeline=vision_pipeline,
            transcript_manager=transcript_manager,
            metrics_collector=metrics_collector,
            rag=rag
        )

    # region Pipeline Status
    
    def get_pipeline_status(self) -> dict:
        """Returns availability and capability information for pipelines.

        Intended for UI consumers to determine which features are enabled.

        Returns:
            dict: {
                "text": {
                    "available": bool,
                    "pump": str | None,
                    "capabilities": list[str]
                },
                "vision": {
                    "available": bool,
                    "pump": str | None,
                    "capabilities": list[str]
                }
            }
        """

        def _status(pipeline, pump_name):
            if pipeline is None:
                return {
                    "available": False,
                    "pump": None,
                    "capabilities": [],
                }
            return {
                "available": True,
                "pump": pump_name,
                "capabilities": [c.value for c in pipeline.pump.capabilities],
            }

        return {
            "text": _status(
                self.text_pipeline,
                self.config.pipeline_subscriptions.get("text"),
            ),
            "vision": _status(
                self.vision_pipeline,
                self.config.pipeline_subscriptions.get("vision"),
            ),
            "rag": {
                "available": self.rag is not None,
            },
        }

    # endregion Pipeline Status

    def set_map(self, selected_map: str) -> None:
        """Sets the active map context for the current session.

        Args:
            selected_map: Map identifier or description string.
        """
        self.session.set_selected_map(selected_map)

    # region Inference Methods 

    def capture_coordinates(self) -> str:
        """Captures the configured region and runs vision inference.

        Returns:
            str: Model output text.

        Raises:
            RuntimeError: If no vision pipeline is configured.
        """
        if self.vision_pipeline is None:
            raise RuntimeError(
                "No vision pipeline configured. "
                "Add a vision pump and subscription to core_config.json."
            )

        image = self.vision_pipeline.capture_region()
        return self.vision_pipeline.infer_text(image)

    def query_text_model(
        self,
        user_text: str,
        selected_map: Optional[str] = None,
    ) -> _TimedIteratorStreamer:
        """Submits a text query and returns a streaming response.

        Args:
            user_text: User query string.
            selected_map: Optional map context override.

        Returns:
            _TimedIteratorStreamer: Iterable token stream.

        Raises:
            RuntimeError: If no text pipeline is configured.
            ValueError: If no map is selected.
        """
        if self.text_pipeline is None:
            raise RuntimeError(
                "No text pipeline configured. "
                "Add a text pump and subscription to core_config.json."
            )

        if selected_map is not None:
            self.set_map(selected_map)

        if self.session.selected_map is None:
            raise ValueError("No map selected for text query.")

        return self.text_pipeline.stream_infer(
            user_text,
            self.session.selected_map,
        )
    
    # endregion Inference Methods

    # region Transcripting

    def log_transcript(self, user_text: str, model_text: str) -> None:
        """Writes a user/model interaction to the transcript log.

        Args:
            user_text: Input text from the user.
            model_text: Output text from the model.
        """
        self.transcript_manager.write(user_text, model_text)

    # endregion Transcripting

    # region Vision Control Loop

    def pause_visual_loop(self) -> None:
        """Pauses the vision loop."""
        self.session.pause_visual_loop()

    def resume_visual_loop(self) -> None:
        """Resumes the vision loop."""
        self.session.resume_visual_loop()

    def is_visual_loop_active(self) -> bool:
        """Checks whether the vision loop is currently active.

        Returns:
            bool: True if active, False otherwise.
        """
        return self.session.is_visual_loop_active()
    # endregion Vision Control Loop

    def shutdown(self) -> None:
        """Shuts down all pumps and worker threads cleanly."""
        for pump in self.pumps.values():
            pump.shutdown()