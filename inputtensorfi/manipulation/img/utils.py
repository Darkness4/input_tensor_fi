"""Utilities for image manipulation."""

import numpy as np
from inputtensorfi.manipulation.img.bit_fault import BitFault
from inputtensorfi.manipulation.img.pixel_fault import PixelFault


def perturb_image(
    pixels: np.ndarray,
    img: np.ndarray,
) -> np.ndarray:
    """Change the pixels of the [img] according to [pixels].

    Note: the original image doesn't change.

    Args:
        pixels (np.ndarray(dtype=PixelFault)):
            A list of pixels to be faulted.
        img (np.ndarray): A 2D RGB image.
                An img[x: int, y: int] = (r: int, g: int, b: int)

    Returns:
        img: 2D RGB image.
    """

    copy = img.copy()

    for pixel in np.nditer(pixels, flags=["refs_ok"]):
        item = pixel.item()
        assert isinstance(item, PixelFault)
        copy[item.x, item.y] = item.rgb

    return copy


def perturb_image_by_bit_fault(
    bit_faults: np.ndarray, img: np.ndarray
) -> np.ndarray:
    """Change the pixels of the [img] according to [bit_faults].

    Note: the original image doesn't change.

    Args:
        bit_faults (np.ndarray(dtype=BitFault)):
                A list of pixels to be bit-faulted.
        img (np.ndarray): A 2D RGB image.
                An img[x: int, y: int] = (r: int, g: int, b: int)

    Returns:
        np.ndarray: 2D RGB image.
    """

    copy = img.copy()

    for bit_fault in np.nditer(bit_faults, flags=["refs_ok"]):
        item = bit_fault.item()
        assert isinstance(item, BitFault)

        copy[item.x, item.y, item.rgb] = item.bit_action.call(
            copy[item.x, item.y, item.rgb], item.bit
        )
    return copy
