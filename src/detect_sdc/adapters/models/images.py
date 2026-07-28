"""Image normalization shared by multimodal model adapters."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any


def load_pil_image(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
        raise ValueError(
            f"Image mapping must contain bytes or path; keys={sorted(value)}"
        )
    raise TypeError(f"Unsupported image input type: {type(value).__name__}")
