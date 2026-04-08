from __future__ import annotations

RESTAURANTS = {
    "spice_route": {
        "name": "Spice Route",
        "cuisine": "Indian",
        "rating": 4.5,
        "eta_min": 30,
        "items": [
            {"item_id": "paneer_wrap", "name": "Paneer Wrap", "price": 220, "veg": True, "customizable": {"no_onions": True, "extra_spicy": True}},
            {"item_id": "veg_biryani", "name": "Veg Biryani", "price": 260, "veg": True, "customizable": {"no_onions": True, "extra_spicy": False}},
            {"item_id": "dal_khichdi", "name": "Dal Khichdi", "price": 180, "veg": True, "customizable": {"no_onions": False, "extra_spicy": False}},
            {"item_id": "chicken_roll", "name": "Chicken Roll", "price": 240, "veg": False, "customizable": {"no_onions": True, "extra_spicy": True}},
        ],
    },
    "burger_hub": {
        "name": "Burger Hub",
        "cuisine": "Burgers",
        "rating": 4.2,
        "eta_min": 25,
        "items": [
            {"item_id": "cheese_burger", "name": "Cheese Burger", "price": 180, "veg": False, "customizable": {"no_onions": True, "extra_spicy": False}},
            {"item_id": "paneer_burger", "name": "Paneer Burger", "price": 210, "veg": True, "customizable": {"no_onions": True, "extra_spicy": False}},
            {"item_id": "fries", "name": "Fries", "price": 120, "veg": True, "customizable": {"no_onions": False, "extra_spicy": False}},
            {"item_id": "chicken_burger", "name": "Chicken Burger", "price": 220, "veg": False, "customizable": {"no_onions": True, "extra_spicy": True}},
        ],
    },
    "green_bowl": {
        "name": "Green Bowl",
        "cuisine": "Healthy",
        "rating": 4.6,
        "eta_min": 20,
        "items": [
            {"item_id": "tofu_bowl", "name": "Tofu Bowl", "price": 230, "veg": True, "customizable": {"no_onions": True, "extra_spicy": False}},
            {"item_id": "paneer_bowl", "name": "Paneer Bowl", "price": 240, "veg": True, "customizable": {"no_onions": True, "extra_spicy": False}},
            {"item_id": "greek_salad", "name": "Greek Salad", "price": 200, "veg": True, "customizable": {"no_onions": True, "extra_spicy": False}},
            {"item_id": "chicken_salad", "name": "Chicken Salad", "price": 250, "veg": False, "customizable": {"no_onions": True, "extra_spicy": False}},
        ],
    },
}

COUPONS = {"SAVE50": {"min_subtotal": 400, "discount_amount": 50}}

DELIVERY_FEE = 40
HOME_VISIBLE_RESTAURANTS = 3
RESTAURANT_PAGE_VISIBLE_ITEMS = 2
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920


def get_restaurant(restaurant_id: str) -> dict:
    return RESTAURANTS[restaurant_id]


def get_item(restaurant_id: str, item_id: str) -> dict:
    for item in RESTAURANTS[restaurant_id]["items"]:
        if item["item_id"] == item_id:
            return item
    raise KeyError(f"Unknown item {item_id} for restaurant {restaurant_id}")
