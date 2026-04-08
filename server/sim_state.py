from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .data import DELIVERY_FEE, get_item

HISTORY_WINDOW = 5


@dataclass
class SimState:
    task_id: str
    seed: int
    app_id: str = "quickcart"
    screen: str = "home"
    search_query: str = ""
    selected_restaurant_id: Optional[str] = None
    selected_item_id: Optional[str] = None
    item_detail_qty: int = 1
    item_detail_customizations: Dict[str, bool] = field(default_factory=dict)
    cart: List[dict] = field(default_factory=list)
    coupon_input: str = ""
    coupon_code: Optional[str] = None
    coupon_applied: bool = False
    discount_amount: int = 0
    delivery_mode: str = "standard"
    scroll_offsets: Dict[str, int] = field(default_factory=dict)
    last_action_status: str = "ok"
    forbidden_triggered: bool = False
    invalid_action_count: int = 0
    recent_state_hashes: List[str] = field(default_factory=list)
    recent_action_keys: List[str] = field(default_factory=list)
    recent_history: List[dict] = field(default_factory=list)
    checkpoints_hit: set[str] = field(default_factory=set)
    step_count: int = 0
    done: bool = False


def normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def subtotal(state: SimState) -> int:
    return sum(line["qty"] * line["unit_price"] for line in state.cart)


def total_qty(state: SimState) -> int:
    return sum(line["qty"] for line in state.cart)


def delivery_fee(state: SimState) -> int:
    return DELIVERY_FEE


def total_after_discount(state: SimState) -> int:
    return max(subtotal(state) - state.discount_amount + delivery_fee(state), 0)


def state_hash(state: SimState) -> str:
    payload = {
        "task_id": state.task_id,
        "screen": state.screen,
        "search_query": state.search_query,
        "selected_restaurant_id": state.selected_restaurant_id,
        "selected_item_id": state.selected_item_id,
        "item_detail_qty": state.item_detail_qty,
        "item_detail_customizations": dict(sorted(state.item_detail_customizations.items())),
        "cart": state.cart,
        "coupon_input": state.coupon_input,
        "coupon_code": state.coupon_code,
        "coupon_applied": state.coupon_applied,
        "discount_amount": state.discount_amount,
        "delivery_mode": state.delivery_mode,
        "scroll_offsets": dict(sorted(state.scroll_offsets.items())),
        "forbidden_triggered": state.forbidden_triggered,
        "done": state.done,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def action_key(action_type: str, target_id: Optional[str] = None, text: Optional[str] = None, direction: Optional[str] = None) -> str:
    normalized_text_value = (text or "").strip().lower()
    return f"{action_type}|{target_id or ''}|{normalized_text_value}|{direction or ''}"


def append_history_entry(state: SimState, entry: dict, current_action_key: str) -> None:
    state.recent_history.append(entry)
    state.recent_history = state.recent_history[-HISTORY_WINDOW:]
    state.recent_action_keys.append(current_action_key)
    state.recent_action_keys = state.recent_action_keys[-HISTORY_WINDOW:]


def add_to_cart(state: SimState, restaurant_id: str, item_id: str, qty: int, customizations: Dict[str, bool]) -> None:
    item = get_item(restaurant_id, item_id)
    normalized_customizations = dict(sorted((key, bool(value)) for key, value in customizations.items()))
    for line in state.cart:
        if (
            line["item_id"] == item_id
            and line["restaurant_id"] == restaurant_id
            and dict(sorted(line["customizations"].items())) == normalized_customizations
        ):
            line["qty"] += qty
            return
    state.cart.append(
        {
            "item_id": item_id,
            "restaurant_id": restaurant_id,
            "qty": qty,
            "unit_price": item["price"],
            "veg": item["veg"],
            "customizations": normalized_customizations,
        }
    )
