from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator


RoleLiteral = Literal[
    "button",
    "text",
    "input",
    "card",
    "list_item",
    "toggle",
    "radio",
    "image",
    "label",
]


class UIElement(BaseModel):
    element_id: str
    role: RoleLiteral
    text: Optional[str] = None
    content_desc: Optional[str] = None
    clickable: bool = False
    enabled: bool = True
    visible: bool = True
    checked: Optional[bool] = None
    value: Optional[str] = None
    bounds: Tuple[int, int, int, int]
    xpath: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


class RewardBreakdown(BaseModel):
    progress_score: float = 0.0
    delta_progress: float = 0.0
    step_penalty: float = 0.0
    invalid_penalty: float = 0.0
    loop_penalty: float = 0.0
    repetition_penalty: float = 0.0
    forbidden_penalty: float = 0.0
    final_score: float = 0.0


class MobileAutomationAction(Action):
    action_type: Literal["tap", "type", "scroll", "back", "wait"]
    target_id: Optional[str] = None
    text: Optional[str] = None
    direction: Optional[Literal["up", "down", "left", "right"]] = None

    @model_validator(mode="after")
    def validate_semantics(self) -> "MobileAutomationAction":
        if self.action_type == "tap" and not self.target_id:
            raise ValueError("tap requires target_id")
        if self.action_type == "type" and (not self.target_id or self.text is None):
            raise ValueError("type requires target_id and text")
        if self.action_type == "scroll" and not self.direction:
            raise ValueError("scroll requires direction")
        if self.action_type in {"back", "wait"}:
            if self.target_id is not None or self.text is not None or self.direction is not None:
                raise ValueError(f"{self.action_type} does not accept target_id, text, or direction")
        return self


class HistoryEntry(BaseModel):
    step: int
    screen_id: str
    action: Dict = Field(default_factory=dict)
    last_action_status: Literal["ok", "invalid", "no_effect", "forbidden"]
    reward: float = 0.0
    progress_score: float = 0.0


class MobileAutomationObservation(Observation):
    task_id: str
    goal: str
    current_app: str
    screen_id: str
    screenshot_b64: str
    xml_hierarchy: str
    ui_elements: List[UIElement]
    last_action_status: Literal["ok", "invalid", "no_effect", "forbidden"]
    progress_score: float
    reward_breakdown: RewardBreakdown
    recent_history: List[HistoryEntry] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    metadata: Dict = Field(default_factory=dict)


QuickCartAction = MobileAutomationAction
QuickCartObservation = MobileAutomationObservation
