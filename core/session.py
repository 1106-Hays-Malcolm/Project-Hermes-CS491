from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionState:
    """Tracks runtime state for the whole pipeline.

    Attributes:
        selected_map: Optional[str] - currently selected map context.
        vl_started: bool - whether the vision loop has started.
        vl_pause: bool - whether the vision loop is currently paused.
    """

    selected_map: Optional[str] = None
    vl_started: bool = False
    vl_pause: bool = False

    def set_selected_map(self, selected_map: str) -> None:
        """Sets the current selected map context.

        Args:
            selected_map: The selected map name.
        """
        self.selected_map = selected_map

    def start_visual_loop(self) -> None:
        """Marks the vision loop as started."""
        self.vl_started = True

    def pause_visual_loop(self) -> None:
        """Pauses the vision loop."""
        self.vl_pause = True

    def resume_visual_loop(self) -> None:
        """Resumes the vision loop if it is paused."""
        self.vl_pause = False

    def is_visual_loop_active(self) -> bool:
        """Returns whether the vision loop should be running.

        Returns:
            bool: True when the loop is started and not paused.
        """
        return self.vl_started and not self.vl_pause

    def reset(self) -> None:
        """Resets session state to initial defaults."""
        self.selected_map = None
        self.vl_started = False
        self.vl_pause = False
