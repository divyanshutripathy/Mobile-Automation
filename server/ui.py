from __future__ import annotations

try:
    from ..models import UIElement
    from .data import HOME_VISIBLE_RESTAURANTS, RESTAURANT_PAGE_VISIBLE_ITEMS, RESTAURANTS, get_item, get_restaurant
    from .sim_state import SimState, delivery_fee, subtotal, total_after_discount
except ImportError:
    from models import UIElement
    from server.data import HOME_VISIBLE_RESTAURANTS, RESTAURANT_PAGE_VISIBLE_ITEMS, RESTAURANTS, get_item, get_restaurant
    from server.sim_state import SimState, delivery_fee, subtotal, total_after_discount


def _element(
    element_id: str,
    role: str,
    bounds: tuple[int, int, int, int],
    *,
    text: str | None = None,
    value: str | None = None,
    clickable: bool = False,
    checked: bool | None = None,
    metadata: dict | None = None,
    xpath: str | None = None,
) -> UIElement:
    return UIElement(
        element_id=element_id,
        role=role,
        text=text,
        clickable=clickable,
        enabled=True,
        visible=True,
        checked=checked,
        value=value,
        bounds=bounds,
        xpath=xpath,
        metadata=metadata or {},
    )


def _filtered_restaurants(search_query: str) -> list[tuple[str, dict]]:
    normalized = search_query.strip().lower()
    filtered = []
    for restaurant_id, restaurant in RESTAURANTS.items():
        haystack = f"{restaurant['name']} {restaurant['cuisine']}".lower()
        if not normalized or normalized in haystack:
            filtered.append((restaurant_id, restaurant))
    return filtered[:HOME_VISIBLE_RESTAURANTS]


def build_ui_elements(state: SimState) -> list[UIElement]:
    if state.screen == "home":
        elements = [
            _element("title_home", "text", (40, 40, 500, 100), text="QuickCart", xpath="/hierarchy/home/text[@id='title_home']"),
            _element("search_bar", "input", (40, 120, 1040, 190), value=state.search_query, clickable=True, xpath="/hierarchy/home/input[@id='search_bar']"),
        ]
        top = 260
        for index, (restaurant_id, restaurant) in enumerate(_filtered_restaurants(state.search_query)):
            bounds = (40, top + index * 240, 1040, top + index * 240 + 200)
            elements.append(
                _element(
                    f"restaurant_card_{restaurant_id}",
                    "card",
                    bounds,
                    text=restaurant["name"],
                    clickable=True,
                    xpath=f"/hierarchy/home/card[@id='restaurant_card_{restaurant_id}']",
                    metadata={"restaurant_id": restaurant_id, "rating": restaurant["rating"], "eta_min": restaurant["eta_min"]},
                )
            )
        return elements

    if state.screen == "restaurant_page":
        restaurant = get_restaurant(state.selected_restaurant_id or "spice_route")
        offset = state.scroll_offsets.get("restaurant_page", 0)
        visible_items = restaurant["items"][offset : offset + RESTAURANT_PAGE_VISIBLE_ITEMS]
        elements = [
            _element("btn_back_home", "button", (40, 60, 180, 130), text="Back", clickable=True, xpath="/hierarchy/restaurant_page/button[@id='btn_back_home']"),
            _element("restaurant_header", "text", (40, 80, 600, 140), text=restaurant["name"], xpath="/hierarchy/restaurant_page/text[@id='restaurant_header']", metadata={"restaurant_id": state.selected_restaurant_id}),
        ]
        for index, item in enumerate(visible_items):
            top = 220 + index * 240
            elements.extend(
                [
                    _element(f"item_card_{item['item_id']}", "card", (40, top, 1040, top + 200), text=item["name"], xpath=f"/hierarchy/restaurant_page/card[@id='item_card_{item['item_id']}']", metadata={"item_id": item["item_id"], "price": item["price"], "veg": item["veg"]}),
                    _element(f"btn_open_item_{item['item_id']}", "button", (760, top + 120, 900, top + 180), text="Details", clickable=True, xpath=f"/hierarchy/restaurant_page/card[@id='item_card_{item['item_id']}']/button[@id='btn_open_item_{item['item_id']}']", metadata={"item_id": item["item_id"]}),
                    _element(f"btn_quick_add_{item['item_id']}", "button", (910, top + 120, 1020, top + 180), text="Add", clickable=True, xpath=f"/hierarchy/restaurant_page/card[@id='item_card_{item['item_id']}']/button[@id='btn_quick_add_{item['item_id']}']", metadata={"item_id": item["item_id"]}),
                ]
            )
        elements.append(_element("btn_open_cart", "button", (840, 1780, 1020, 1860), text="Cart", clickable=True, xpath="/hierarchy/restaurant_page/button[@id='btn_open_cart']"))
        return elements

    if state.screen == "item_detail":
        item = get_item(state.selected_restaurant_id or "spice_route", state.selected_item_id or "paneer_wrap")
        elements = [
            _element("btn_back_restaurant", "button", (40, 60, 220, 130), text="Back", clickable=True, xpath="/hierarchy/item_detail/button[@id='btn_back_restaurant']"),
            _element("item_name", "text", (60, 180, 700, 260), text=item["name"], xpath="/hierarchy/item_detail/text[@id='item_name']"),
            _element("item_price", "label", (60, 280, 320, 340), text=f"Rs {item['price']}", xpath="/hierarchy/item_detail/label[@id='item_price']"),
        ]
        if item["customizable"].get("no_onions"):
            elements.append(_element("toggle_no_onions", "toggle", (60, 500, 1020, 580), text="No onions", clickable=True, checked=state.item_detail_customizations.get("no_onions", False), xpath="/hierarchy/item_detail/toggle[@id='toggle_no_onions']"))
        if item["customizable"].get("extra_spicy"):
            elements.append(_element("toggle_extra_spicy", "toggle", (60, 620, 1020, 700), text="Extra spicy", clickable=True, checked=state.item_detail_customizations.get("extra_spicy", False), xpath="/hierarchy/item_detail/toggle[@id='toggle_extra_spicy']"))
        elements.extend(
            [
                _element("qty_minus", "button", (240, 900, 320, 980), text="-", clickable=True, xpath="/hierarchy/item_detail/button[@id='qty_minus']"),
                _element("qty_value", "label", (360, 900, 700, 980), text=f"Qty: {state.item_detail_qty}", xpath="/hierarchy/item_detail/label[@id='qty_value']"),
                _element("qty_plus", "button", (780, 900, 860, 980), text="+", clickable=True, xpath="/hierarchy/item_detail/button[@id='qty_plus']"),
                _element("btn_add_to_cart", "button", (60, 1680, 1020, 1780), text="Add to Cart", clickable=True, xpath="/hierarchy/item_detail/button[@id='btn_add_to_cart']"),
            ]
        )
        return elements

    if state.screen == "cart":
        elements = [_element("btn_back_restaurant", "button", (40, 60, 220, 130), text="Back", clickable=True, xpath="/hierarchy/cart/button[@id='btn_back_restaurant']")]
        top = 220
        for index, line in enumerate(state.cart):
            row_top = top + index * 160
            label = line["item_id"].replace("_", " ").title()
            elements.extend(
                [
                    _element(f"cart_line_{index}", "list_item", (40, row_top, 1040, row_top + 140), text=f"{label} x{line['qty']}", xpath=f"/hierarchy/cart/list_item[@id='cart_line_{index}']", metadata={"item_id": line["item_id"], "qty": line["qty"], "customizations": line["customizations"]}),
                    _element(f"qty_dec_{index}", "button", (720, row_top + 40, 780, row_top + 100), text="-", clickable=True, xpath=f"/hierarchy/cart/button[@id='qty_dec_{index}']"),
                    _element(f"qty_inc_{index}", "button", (940, row_top + 40, 1000, row_top + 100), text="+", clickable=True, xpath=f"/hierarchy/cart/button[@id='qty_inc_{index}']"),
                ]
            )
        elements.extend(
            [
                _element("coupon_input", "input", (40, 1180, 760, 1260), value=state.coupon_input, clickable=True, xpath="/hierarchy/cart/input[@id='coupon_input']"),
                _element("btn_apply_coupon", "button", (800, 1180, 1020, 1260), text="Apply", clickable=True, xpath="/hierarchy/cart/button[@id='btn_apply_coupon']"),
                _element("delivery_radio_standard", "radio", (40, 1360, 500, 1440), text="Standard", clickable=True, checked=state.delivery_mode == "standard", xpath="/hierarchy/cart/radio[@id='delivery_radio_standard']"),
                _element("delivery_radio_no_contact", "radio", (540, 1360, 1020, 1440), text="No contact", clickable=True, checked=state.delivery_mode == "no_contact", xpath="/hierarchy/cart/radio[@id='delivery_radio_no_contact']"),
                _element("summary_subtotal", "label", (60, 1480, 500, 1540), text=f"Subtotal: Rs {subtotal(state)}", xpath="/hierarchy/cart/label[@id='summary_subtotal']"),
                _element("summary_discount", "label", (60, 1560, 500, 1620), text=f"Discount: Rs {state.discount_amount}", xpath="/hierarchy/cart/label[@id='summary_discount']"),
                _element("summary_total", "label", (60, 1640, 500, 1700), text=f"Total: Rs {total_after_discount(state)}", xpath="/hierarchy/cart/label[@id='summary_total']", metadata={"delivery_fee": delivery_fee(state)}),
                _element("btn_review_order", "button", (60, 1700, 1020, 1800), text="Review Order", clickable=True, xpath="/hierarchy/cart/button[@id='btn_review_order']"),
            ]
        )
        return elements

    if state.screen == "review_order":
        summary = ", ".join(f"{line['item_id']} x{line['qty']}" for line in state.cart) or "No items"
        return [
            _element("review_header", "text", (60, 100, 600, 160), text="Review Order", xpath="/hierarchy/review_order/text[@id='review_header']"),
            _element("review_summary", "label", (60, 260, 1020, 380), text=summary, xpath="/hierarchy/review_order/label[@id='review_summary']"),
            _element("review_delivery_mode", "label", (60, 420, 1020, 500), text=f"Delivery: {state.delivery_mode}", xpath="/hierarchy/review_order/label[@id='review_delivery_mode']"),
            _element("review_coupon", "label", (60, 520, 1020, 600), text=f"Coupon: {state.coupon_code or 'None'}", xpath="/hierarchy/review_order/label[@id='review_coupon']"),
            _element("btn_back_cart", "button", (60, 1580, 420, 1680), text="Back to Cart", clickable=True, xpath="/hierarchy/review_order/button[@id='btn_back_cart']"),
            _element("btn_place_order", "button", (60, 1700, 1020, 1800), text="Place Order", clickable=True, xpath="/hierarchy/review_order/button[@id='btn_place_order']", metadata={"forbidden": True}),
        ]

    raise ValueError(f"Unsupported screen {state.screen}")
