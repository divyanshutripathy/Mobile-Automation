from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .sim_state import SimState, normalized_text, subtotal, total_qty


PredicateFn = Callable[[SimState], bool]


@dataclass(frozen=True)
class WeightedPredicate:
    name: str
    weight: float
    predicate_fn: PredicateFn


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    goal: str
    max_steps: int
    initial_state_factory: Callable[[int], SimState]
    progress_predicates: list[WeightedPredicate]
    terminal_predicates: list[WeightedPredicate]
    forbidden_actions: list[str]


def _initial_state(task_id: str, seed: int) -> SimState:
    return SimState(task_id=task_id, seed=seed)


def _search_contains_spice(state: SimState) -> bool:
    return "spice" in normalized_text(state.search_query)


def _selected_spice_route(state: SimState) -> bool:
    return state.selected_restaurant_id == "spice_route"


def _find_line(state: SimState, item_id: str, qty: int, customizations: dict) -> bool:
    return any(
        line["item_id"] == item_id
        and line["qty"] == qty
        and line["restaurant_id"] == "spice_route"
        and line["customizations"] == customizations
        for line in state.cart
    )


def _easy_exact(state: SimState) -> bool:
    return len(state.cart) == 1 and _find_line(state, "paneer_wrap", 1, {})


def _medium_exact(state: SimState) -> bool:
    return len(state.cart) == 2 and _find_line(state, "paneer_wrap", 1, {}) and _find_line(state, "veg_biryani", 1, {})


def _hard_all_items_veg(state: SimState) -> bool:
    return bool(state.cart) and all(line["veg"] for line in state.cart)


def _hard_all_from_spice_route(state: SimState) -> bool:
    return bool(state.cart) and all(line["restaurant_id"] == "spice_route" for line in state.cart)


def _hard_has_no_onions(state: SimState) -> bool:
    return any(line["customizations"].get("no_onions") is True for line in state.cart)


TASKS = {
    "food_easy": TaskSpec(
        task_id="food_easy",
        difficulty="easy",
        goal="In QuickCart, open Spice Route, add exactly 1 Paneer Wrap with default options, and stop at the cart screen. Do not review or place the order.",
        max_steps=12,
        initial_state_factory=lambda seed: _initial_state("food_easy", seed),
        progress_predicates=[
            WeightedPredicate("opened_spice_route", 0.25, _selected_spice_route),
            WeightedPredicate("paneer_wrap_in_cart", 0.45, lambda s: _find_line(s, "paneer_wrap", 1, {})),
            WeightedPredicate("on_cart_screen", 0.20, lambda s: s.screen == "cart"),
            WeightedPredicate("exact_easy_cart_match", 0.10, _easy_exact),
        ],
        terminal_predicates=[
            WeightedPredicate("selected_restaurant_id", 1.0, _selected_spice_route),
            WeightedPredicate("screen_is_cart", 1.0, lambda s: s.screen == "cart"),
            WeightedPredicate("exact_easy_cart_match", 1.0, _easy_exact),
            WeightedPredicate("forbidden_not_triggered", 1.0, lambda s: not s.forbidden_triggered),
        ],
        forbidden_actions=["btn_place_order"],
    ),
    "food_medium": TaskSpec(
        task_id="food_medium",
        difficulty="medium",
        goal="Search for Spice Route, add 1 Paneer Wrap and 1 Veg Biryani, apply coupon SAVE50, and stop at the review order screen. Do not place the order.",
        max_steps=18,
        initial_state_factory=lambda seed: _initial_state("food_medium", seed),
        progress_predicates=[
            WeightedPredicate("typed_search_for_spice", 0.10, _search_contains_spice),
            WeightedPredicate("opened_spice_route", 0.15, _selected_spice_route),
            WeightedPredicate("paneer_wrap_added", 0.15, lambda s: _find_line(s, "paneer_wrap", 1, {})),
            WeightedPredicate("veg_biryani_added", 0.15, lambda s: _find_line(s, "veg_biryani", 1, {})),
            WeightedPredicate("exact_medium_cart_match", 0.15, _medium_exact),
            WeightedPredicate("coupon_applied", 0.15, lambda s: s.coupon_code == "SAVE50" and s.coupon_applied and s.discount_amount == 50),
            WeightedPredicate("on_review_order_screen", 0.15, lambda s: s.screen == "review_order"),
        ],
        terminal_predicates=[
            WeightedPredicate("typed_search_for_spice", 1.0, _search_contains_spice),
            WeightedPredicate("opened_spice_route", 1.0, _selected_spice_route),
            WeightedPredicate("exact_medium_cart_match", 1.0, _medium_exact),
            WeightedPredicate("subtotal_480", 1.0, lambda s: subtotal(s) == 480),
            WeightedPredicate("coupon_applied", 1.0, lambda s: s.coupon_code == "SAVE50" and s.coupon_applied and s.discount_amount == 50),
            WeightedPredicate("review_screen", 1.0, lambda s: s.screen == "review_order"),
            WeightedPredicate("forbidden_not_triggered", 1.0, lambda s: not s.forbidden_triggered),
        ],
        forbidden_actions=["btn_place_order"],
    ),
    "food_hard": TaskSpec(
        task_id="food_hard",
        difficulty="hard",
        goal="Search for Spice Route, create a vegetarian order for 2 with merchandise subtotal <= 400, ensure at least one selected item has no_onions=True, choose no_contact delivery, and stop at the review order screen. Do not place the order.",
        max_steps=25,
        initial_state_factory=lambda seed: _initial_state("food_hard", seed),
        progress_predicates=[
            WeightedPredicate("typed_search_for_spice", 0.05, _search_contains_spice),
            WeightedPredicate("opened_spice_route", 0.10, _selected_spice_route),
            WeightedPredicate("cart_non_empty", 0.10, lambda s: bool(s.cart)),
            WeightedPredicate("all_items_veg", 0.15, _hard_all_items_veg),
            WeightedPredicate("quantity_at_least_2", 0.10, lambda s: total_qty(s) >= 2),
            WeightedPredicate("subtotal_within_budget", 0.20, lambda s: bool(s.cart) and subtotal(s) <= 400),
            WeightedPredicate("has_no_onions_item", 0.15, _hard_has_no_onions),
            WeightedPredicate("no_contact_selected", 0.05, lambda s: s.delivery_mode == "no_contact"),
            WeightedPredicate("on_review_order_screen", 0.10, lambda s: s.screen == "review_order"),
        ],
        terminal_predicates=[
            WeightedPredicate("typed_search_for_spice", 1.0, _search_contains_spice),
            WeightedPredicate("opened_spice_route", 1.0, _selected_spice_route),
            WeightedPredicate("cart_non_empty", 1.0, lambda s: bool(s.cart)),
            WeightedPredicate("all_items_from_spice_route", 1.0, _hard_all_from_spice_route),
            WeightedPredicate("all_items_veg", 1.0, _hard_all_items_veg),
            WeightedPredicate("quantity_at_least_2", 1.0, lambda s: total_qty(s) >= 2),
            WeightedPredicate("subtotal_within_budget", 1.0, lambda s: subtotal(s) <= 400),
            WeightedPredicate("has_no_onions_item", 1.0, _hard_has_no_onions),
            WeightedPredicate("no_contact_selected", 1.0, lambda s: s.delivery_mode == "no_contact"),
            WeightedPredicate("review_screen", 1.0, lambda s: s.screen == "review_order"),
            WeightedPredicate("forbidden_not_triggered", 1.0, lambda s: not s.forbidden_triggered),
        ],
        forbidden_actions=["btn_place_order"],
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id {task_id}")
    return TASKS[task_id]
