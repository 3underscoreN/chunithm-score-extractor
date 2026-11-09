from dataclasses import dataclass
from io import BytesIO
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InvalidImageError


@dataclass(frozen=True, slots=True)
class ImageLimits:
    max_bytes: int = 15 * 1024 * 1024
    max_pixels: int = 25_000_000
    max_dimension: int = 10_000


def decode_image(data: bytes, limits: ImageLimits = ImageLimits()) -> np.ndarray:
    """
    Decodes the given image data into a BGR image suitable for OpenCV processing.

    Parameters
    ----------
    data : bytes
        The raw image data to decode.
    limits : ImageLimits, optional
        The limits to enforce on the image size and dimensions.

    Returns
    -------
    np.ndarray
        The decoded image in BGR format.

    Raises
    ------
    InvalidImageError
        If the image is empty, exceeds the limits, or cannot be decoded.
    """
    if not data or len(data) > limits.max_bytes:
        raise InvalidImageError("Image is empty or exceeds the byte limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImageError("Only JPEG, PNG, and WebP images are supported.")
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or width > limits.max_dimension
                    or height > limits.max_dimension
                    or width * height > limits.max_pixels
                ):
                    raise InvalidImageError("Image dimensions exceed the configured limits.")
                image.load()
                oriented = ImageOps.exif_transpose(image).convert("RGB")
                rgb = np.asarray(oriented)
    except InvalidImageError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise InvalidImageError("Image data is corrupted or cannot be decoded.") from exc

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.size == 0:
        raise InvalidImageError("Decoded image is empty.")
    return bgr
