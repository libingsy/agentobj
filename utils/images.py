"""Image conversions used by the multi-view generator."""

from typing import List, Sequence, Union

import numpy as np
from PIL import Image


ArrayLike = Union[np.ndarray, Image.Image]


def to_pil(image: ArrayLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        scale = 255.0 if float(np.max(arr)) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_uint8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def stack_uint8(images: Sequence[ArrayLike]) -> List[np.ndarray]:
    return [pil_to_uint8(to_pil(img)) for img in images]
