from __future__ import annotations

import base64
import io
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

try:
    from ..models import UIElement
    from .data import IMAGE_HEIGHT, IMAGE_WIDTH
except ImportError:
    from models import UIElement
    from server.data import IMAGE_HEIGHT, IMAGE_WIDTH


def render_xml(ui_elements: list[UIElement], screen_id: str) -> str:
    lines = [f'<hierarchy screen="{escape(screen_id)}">']
    for element in ui_elements:
        attributes = {
            "id": element.element_id,
            "role": element.role,
            "text": element.text or "",
            "value": element.value or "",
            "clickable": str(element.clickable).lower(),
            "enabled": str(element.enabled).lower(),
            "visible": str(element.visible).lower(),
            "checked": "" if element.checked is None else str(element.checked).lower(),
            "bounds": ",".join(str(value) for value in element.bounds),
        }
        attr_string = " ".join(f'{key}="{escape(value)}"' for key, value in attributes.items())
        lines.append(f"  <node {attr_string} />")
    lines.append("</hierarchy>")
    return "\n".join(lines)


def render_screenshot(ui_elements: list[UIElement], screen_id: str, goal: str) -> str:
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1), outline="#DDDDDD", width=2)
    draw.text((40, 12), f"QuickCart {screen_id}", fill="black", font=font)
    draw.text((40, 1860), goal[:120], fill="#444444", font=font)

    for element in ui_elements:
        x1, y1, x2, y2 = element.bounds
        fill = "#F6F6F6"
        if element.role in {"button", "radio", "toggle"}:
            fill = "#E6F0FF"
        elif element.role in {"card", "list_item"}:
            fill = "#F2F7EC"
        elif element.role == "input":
            fill = "#FFF7E6"
        draw.rectangle((x1, y1, x2, y2), outline="#444444", fill=fill, width=3)
        label = element.text or element.value or element.element_id
        if element.role == "toggle" and element.checked is not None:
            label = f"{label} [{'ON' if element.checked else 'OFF'}]"
        if element.role == "radio" and element.checked is not None:
            label = f"{label} [{'X' if element.checked else ' '}]"
        draw.text((x1 + 12, y1 + 12), label[:60], fill="black", font=font)
        if element.metadata.get("veg") is True:
            draw.text((x2 - 80, y1 + 12), "VEG", fill="green", font=font)
        if element.metadata.get("veg") is False:
            draw.text((x2 - 110, y1 + 12), "NON-VEG", fill="red", font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
