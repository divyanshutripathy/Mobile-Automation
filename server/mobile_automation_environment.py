from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MobileAutomationAction, MobileAutomationObservation
    from .data import COUPONS, RESTAURANT_PAGE_VISIBLE_ITEMS, RESTAURANTS
    from .graders import build_metadata, clamp_public_score, compute_progress_score, compute_reward, is_success
    from .render import render_screenshot, render_xml
    from .sim_state import SimState, action_key, add_to_cart, append_history_entry, state_hash, subtotal
    from .tasks import get_task
    from .ui import build_ui_elements
except ImportError:
    from models import MobileAutomationAction, MobileAutomationObservation
    from server.data import COUPONS, RESTAURANT_PAGE_VISIBLE_ITEMS, RESTAURANTS
    from server.graders import build_metadata, clamp_public_score, compute_progress_score, compute_reward, is_success
    from server.render import render_screenshot, render_xml
    from server.sim_state import SimState, action_key, add_to_cart, append_history_entry, state_hash, subtotal
    from server.tasks import get_task
    from server.ui import build_ui_elements


class MobileAutomationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._episode_id = str(uuid4())
        self._task_spec = get_task("food_easy")
        self._sim_state = self._task_spec.initial_state_factory(0)
        self._progress_score = 0.0

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> MobileAutomationObservation:
        task_id = kwargs.get("task_id", "food_easy")
        self._task_spec = get_task(task_id)
        self._episode_id = episode_id or str(uuid4())
        self._sim_state = self._task_spec.initial_state_factory(seed or 0)
        self._sim_state.recent_state_hashes = [state_hash(self._sim_state)]
        self._progress_score, progress_satisfied = compute_progress_score(self._sim_state, self._task_spec)
        _, terminal_satisfied = is_success(self._sim_state, self._task_spec)
        return self._build_observation(0.0, progress_satisfied, terminal_satisfied)

    def step(self, action: MobileAutomationAction, timeout_s: float | None = None, **kwargs) -> MobileAutomationObservation:  # type: ignore[override]
        current_action_key = action_key(action.action_type, action.target_id, action.text, action.direction)
        action_payload = action.model_dump(exclude_none=True)
        if self._sim_state.done:
            if self._sim_state.screen == "review_order" and action.action_type == "tap" and action.target_id == "btn_place_order":
                self._sim_state.last_action_status = "forbidden"
                self._sim_state.forbidden_triggered = True
                progress_satisfied = compute_progress_score(self._sim_state, self._task_spec)[1]
                terminal_satisfied = is_success(self._sim_state, self._task_spec)[1]
                observation = self._build_observation(0.0, progress_satisfied, terminal_satisfied)
                observation.reward_breakdown = compute_reward(
                    self._progress_score,
                    self._progress_score,
                    "forbidden",
                    self._sim_state,
                    current_action_key=current_action_key,
                )
                observation.reward = 0.0
                observation.done = True
                append_history_entry(
                    self._sim_state,
                    {
                        "step": self._sim_state.step_count,
                        "screen_id": self._sim_state.screen,
                        "action": action_payload,
                        "last_action_status": "forbidden",
                        "reward": 0.0,
                        "progress_score": observation.progress_score,
                    },
                    current_action_key,
                )
                observation.recent_history = list(self._sim_state.recent_history)
                return observation
            progress_satisfied = compute_progress_score(self._sim_state, self._task_spec)[1]
            terminal_satisfied = is_success(self._sim_state, self._task_spec)[1]
            return self._build_observation(0.0, progress_satisfied, terminal_satisfied)

        prev_progress = self._progress_score
        ui_elements = build_ui_elements(self._sim_state)
        self._resolve_action(action, ui_elements)
        self._sim_state.step_count += 1
        self._sim_state.recent_state_hashes.append(state_hash(self._sim_state))
        self._sim_state.recent_state_hashes = self._sim_state.recent_state_hashes[-6:]

        self._progress_score, progress_satisfied = compute_progress_score(self._sim_state, self._task_spec)
        success, terminal_satisfied = is_success(self._sim_state, self._task_spec)
        self._sim_state.done = self._sim_state.forbidden_triggered or success or self._sim_state.step_count >= self._task_spec.max_steps

        reward_breakdown = compute_reward(
            prev_progress,
            self._progress_score,
            self._sim_state.last_action_status,
            self._sim_state,
            current_action_key=current_action_key,
        )
        raw_reward = 0.0 if self._sim_state.last_action_status == "forbidden" else (
            reward_breakdown.delta_progress
            - reward_breakdown.step_penalty
            - reward_breakdown.invalid_penalty
            - reward_breakdown.loop_penalty
            - reward_breakdown.repetition_penalty
            - reward_breakdown.forbidden_penalty
        )
        reward = 0.0 if self._sim_state.last_action_status == "forbidden" else clamp_public_score(raw_reward)
        append_history_entry(
            self._sim_state,
            {
                "step": self._sim_state.step_count,
                "screen_id": self._sim_state.screen,
                "action": action_payload,
                "last_action_status": self._sim_state.last_action_status,
                "reward": reward,
                "progress_score": 0.0 if self._sim_state.forbidden_triggered else clamp_public_score(self._progress_score),
            },
            current_action_key,
        )
        observation = self._build_observation(reward, progress_satisfied, terminal_satisfied)
        observation.reward_breakdown = reward_breakdown
        observation.reward = reward
        observation.done = self._sim_state.done
        return observation

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._sim_state.step_count,
            task_id=self._sim_state.task_id,
            screen=self._sim_state.screen,
            progress_score=self._progress_score,
            done=self._sim_state.done,
        )

    def _build_observation(self, reward: float, progress_satisfied: dict[str, bool], terminal_satisfied: dict[str, bool]) -> MobileAutomationObservation:
        ui_elements = build_ui_elements(self._sim_state)
        metadata = build_metadata(self._sim_state, self._task_spec, progress_satisfied, terminal_satisfied)
        return MobileAutomationObservation(
            task_id=self._task_spec.task_id,
            goal=self._task_spec.goal,
            current_app=self._sim_state.app_id,
            screen_id=self._sim_state.screen,
            screenshot_b64=render_screenshot(ui_elements, self._sim_state.screen, self._task_spec.goal),
            xml_hierarchy=render_xml(ui_elements, self._sim_state.screen),
            ui_elements=ui_elements,
            last_action_status=self._sim_state.last_action_status,
            progress_score=0.0 if self._sim_state.forbidden_triggered else clamp_public_score(self._progress_score),
            reward_breakdown=compute_reward(self._progress_score, self._progress_score, self._sim_state.last_action_status, self._sim_state),
            recent_history=list(self._sim_state.recent_history),
            reward=reward,
            done=self._sim_state.done,
            metadata=metadata,
        )

    def _resolve_action(self, action: MobileAutomationAction, ui_elements: list) -> None:
        visible = {element.element_id: element for element in ui_elements if element.visible}
        self._sim_state.last_action_status = "ok"

        if action.action_type == "wait":
            self._sim_state.last_action_status = "no_effect"
            return
        if action.action_type == "back":
            self._handle_back()
            return
        if action.action_type == "scroll":
            self._handle_scroll(action.direction or "")
            return

        target = visible.get(action.target_id or "")
        if target is None or not target.enabled:
            self._mark_invalid()
            return

        if action.target_id == "btn_place_order":
            self._sim_state.last_action_status = "forbidden"
            self._sim_state.forbidden_triggered = True
            self._sim_state.done = True
            return

        if action.action_type == "type":
            self._handle_type(action.target_id or "", action.text or "")
            return

        if not target.clickable:
            self._mark_invalid()
            return
        self._handle_tap(action.target_id or "")

    def _mark_invalid(self) -> None:
        self._sim_state.last_action_status = "invalid"
        self._sim_state.invalid_action_count += 1

    def _handle_back(self) -> None:
        if self._sim_state.screen == "restaurant_page":
            self._sim_state.screen = "home"
        elif self._sim_state.screen == "item_detail":
            self._clear_item_detail()
            self._sim_state.screen = "restaurant_page"
        elif self._sim_state.screen == "cart":
            self._sim_state.screen = "restaurant_page"
        elif self._sim_state.screen == "review_order":
            self._sim_state.screen = "cart"
        else:
            self._sim_state.last_action_status = "no_effect"

    def _handle_scroll(self, direction: str) -> None:
        if self._sim_state.screen != "restaurant_page" or not self._sim_state.selected_restaurant_id:
            self._mark_invalid()
            return
        items = RESTAURANTS[self._sim_state.selected_restaurant_id]["items"]
        max_offset = max(0, len(items) - RESTAURANT_PAGE_VISIBLE_ITEMS)
        current_offset = self._sim_state.scroll_offsets.get("restaurant_page", 0)
        if direction == "down":
            new_offset = min(max_offset, current_offset + 1)
        elif direction == "up":
            new_offset = max(0, current_offset - 1)
        else:
            self._mark_invalid()
            return
        if new_offset == current_offset:
            self._sim_state.last_action_status = "no_effect"
            return
        self._sim_state.scroll_offsets["restaurant_page"] = new_offset

    def _handle_type(self, target_id: str, text: str) -> None:
        if self._sim_state.screen == "home" and target_id == "search_bar":
            self._sim_state.search_query = text
            return
        if self._sim_state.screen == "cart" and target_id == "coupon_input":
            self._sim_state.coupon_input = text.strip().upper()
            return
        self._mark_invalid()

    def _handle_tap(self, target_id: str) -> None:
        if self._sim_state.screen == "home":
            if target_id.startswith("restaurant_card_"):
                self._sim_state.selected_restaurant_id = target_id.replace("restaurant_card_", "", 1)
                self._sim_state.screen = "restaurant_page"
                self._sim_state.scroll_offsets["restaurant_page"] = 0
                return
        elif self._sim_state.screen == "restaurant_page":
            if target_id == "btn_back_home":
                self._sim_state.screen = "home"
                return
            if target_id == "btn_open_cart":
                self._sim_state.screen = "cart"
                return
            if target_id.startswith("btn_open_item_"):
                self._sim_state.selected_item_id = target_id.replace("btn_open_item_", "", 1)
                self._sim_state.item_detail_qty = 1
                self._sim_state.item_detail_customizations = {}
                self._sim_state.screen = "item_detail"
                return
            if target_id.startswith("btn_quick_add_"):
                add_to_cart(self._sim_state, self._sim_state.selected_restaurant_id or "spice_route", target_id.replace("btn_quick_add_", "", 1), 1, {})
                return
        elif self._sim_state.screen == "item_detail":
            if target_id == "btn_back_restaurant":
                self._clear_item_detail()
                self._sim_state.screen = "restaurant_page"
                return
            if target_id == "toggle_no_onions":
                current = self._sim_state.item_detail_customizations.get("no_onions", False)
                self._sim_state.item_detail_customizations["no_onions"] = not current
                return
            if target_id == "toggle_extra_spicy":
                current = self._sim_state.item_detail_customizations.get("extra_spicy", False)
                self._sim_state.item_detail_customizations["extra_spicy"] = not current
                return
            if target_id == "qty_plus":
                self._sim_state.item_detail_qty += 1
                return
            if target_id == "qty_minus":
                self._sim_state.item_detail_qty = max(1, self._sim_state.item_detail_qty - 1)
                return
            if target_id == "btn_add_to_cart":
                add_to_cart(
                    self._sim_state,
                    self._sim_state.selected_restaurant_id or "spice_route",
                    self._sim_state.selected_item_id or "paneer_wrap",
                    self._sim_state.item_detail_qty,
                    self._sim_state.item_detail_customizations,
                )
                self._clear_item_detail()
                self._sim_state.screen = "restaurant_page"
                return
        elif self._sim_state.screen == "cart":
            if target_id == "btn_back_restaurant":
                self._sim_state.screen = "restaurant_page"
                return
            if target_id == "btn_apply_coupon":
                coupon = COUPONS.get(self._sim_state.coupon_input.upper())
                if coupon and subtotal(self._sim_state) >= coupon["min_subtotal"]:
                    self._sim_state.coupon_code = self._sim_state.coupon_input.upper()
                    self._sim_state.coupon_applied = True
                    self._sim_state.discount_amount = coupon["discount_amount"]
                else:
                    self._sim_state.coupon_code = None
                    self._sim_state.coupon_applied = False
                    self._sim_state.discount_amount = 0
                    self._sim_state.last_action_status = "no_effect"
                return
            if target_id == "delivery_radio_standard":
                self._sim_state.delivery_mode = "standard"
                return
            if target_id == "delivery_radio_no_contact":
                self._sim_state.delivery_mode = "no_contact"
                return
            if target_id == "btn_review_order":
                self._sim_state.screen = "review_order"
                return
            if target_id.startswith("qty_inc_") or target_id.startswith("qty_dec_"):
                prefix, index_str = target_id.rsplit("_", 1)
                index = int(index_str)
                if index >= len(self._sim_state.cart):
                    self._mark_invalid()
                    return
                line = self._sim_state.cart[index]
                if prefix == "qty_inc":
                    line["qty"] += 1
                else:
                    line["qty"] = max(1, line["qty"] - 1)
                return
        elif self._sim_state.screen == "review_order":
            if target_id == "btn_back_cart":
                self._sim_state.screen = "cart"
                return

        self._mark_invalid()

    def _clear_item_detail(self) -> None:
        self._sim_state.selected_item_id = None
        self._sim_state.item_detail_qty = 1
        self._sim_state.item_detail_customizations = {}
