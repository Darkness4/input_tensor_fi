"""Utilities for image manipulation."""

import numpy as np
from inputtensorfi.manipulation.img.faults import BitFault, PixelFault
import tensorflow as tf


def build_perturb_image(
    pixels: np.ndarray,
):
    """Build a Fault Injector using [pixels] to be faulted.

    Args:
        pixels (np.ndarray(dtype=PixelFault)):
                A list of pixels to be faulted.
    """
    for pixel in np.nditer(pixels, flags=["refs_ok"]):
        item = pixel.item()
        assert isinstance(item, PixelFault)

    def perturb_image(img: np.ndarray) -> np.ndarray:
        """Change the pixels of the [img] according to [pixels].

        Args:
            img (np.ndarray): A 2D RGB image.
                An img[x: int, y: int] = (r: int, g: int, b: int)

        Returns:
            copy: 2D RGB image.
        """
        # logging.warning(f"Pertubating an image of shape {img.shape}")
        for pixel in np.nditer(pixels, flags=["refs_ok"]):
            item = pixel.item()
            img[item.x, item.y] = item.rgb

        return img

    return perturb_image


def build_perturb_image_by_bit_fault(bit_faults: np.ndarray):
    """Build a Fault Injector using [bit_faults] to be faulted.

    Args:
        bit_faults (dtype=BitFault):
                A list of pixels to be bit-faulted.
    """
    for bit_fault in np.nditer(bit_faults, flags=["refs_ok"]):
        item = bit_fault.item()
        assert isinstance(item, BitFault)

    def perturb_image_by_bit_fault(img: np.ndarray) -> np.ndarray:
        """Change the pixels of the [img] according to [bit_faults].

        Args:
            img (np.ndarray): A 2D RGB image.
                    An img[x: int, y: int] = (r: int, g: int, b: int)

        Returns:
            img: 2D RGB image.
        """
        # logging.warning(f"Pertubating an image of shape {img.shape}")
        for bit_fault in np.nditer(bit_faults, flags=["refs_ok"]):
            item = bit_fault.item()

            img[item.x, item.y, item.rgb] = item.bit_action.call(
                img[item.x, item.y, item.rgb], item.bit
            )
        return img

    return perturb_image_by_bit_fault


def build_perturb_image_tensor(pixels: np.ndarray):
    for pixel in np.nditer(pixels, flags=["refs_ok"]):
        item = pixel.item()
        assert isinstance(item, PixelFault)
    indices = [(pixel.x, pixel.y) for pixel in pixels]
    values = [pixel.rgb for pixel in pixels]

    def perturb_image(x: tf.Tensor) -> tf.Tensor:
        return tf.tensor_scatter_nd_update(
            x,
            indices,
            values,
        )

    return perturb_image
