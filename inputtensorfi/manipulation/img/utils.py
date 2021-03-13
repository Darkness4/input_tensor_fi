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
            img: 2D RGB image.
        """
        img = img.copy()
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

    def perturb_image(img: np.ndarray) -> np.ndarray:
        """Change the pixels of the [img] according to [bit_faults].

        Args:
            img (np.ndarray): A 2D RGB image.
                    An img[x: int, y: int] = (r: int, g: int, b: int)

        Returns:
            img: 2D RGB image.
        """
        img = img.copy()
        for bit_fault in np.nditer(bit_faults, flags=["refs_ok"]):
            item = bit_fault.item()

            img[item.x, item.y, item.rgb] = item.bit_action.call(
                img[item.x, item.y, item.rgb], item.bit
            )
        return img

    return perturb_image


def build_perturb_image_tensor(pixels: np.ndarray):
    """Build a Fault Injector using [bit_faults] to be faulted.

    Optimized for tensors.

    Args:
        bit_faults (dtype=BitFault):
                A list of pixels to be bit-faulted.
    """
    for pixel in np.nditer(pixels, flags=["refs_ok"]):
        item = pixel.item()
        assert isinstance(item, PixelFault)
    indices = [(pixel.x, pixel.y) for pixel in pixels]
    values = np.array([pixel.rgb for pixel in pixels])

    def perturb_image(x: tf.Tensor) -> tf.Tensor:
        """Change the pixels of the [x] according to [pixels].

        Args:
            x (tf.Tensor): A 2D RGB image.
                    shape = (x, y, rgb=3)

        Returns:
            tf.Tensor: 2D RGB image.
        """
        return tf.tensor_scatter_nd_update(
            x,
            indices,
            values,
        )

    return perturb_image


def build_perturb_image_by_bit_fault_tensor(bit_faults: np.ndarray):
    """Build a Fault Injector using [bit_faults] to be faulted.

    Args:
        bit_faults (dtype=BitFault):
                A list of pixels to be bit-faulted.
    """
    for bit_fault in np.nditer(bit_faults, flags=["refs_ok"]):
        item = bit_fault.item()
        assert isinstance(item, BitFault)
    indices = np.array(
        [
            (bit_fault.x, bit_fault.y, bit_fault.rgb)
            for bit_fault in bit_faults
        ],
        dtype=np.int32,
    )
    bit_actions = np.array(
        [
            bit_fault.bit_action.as_tensor(bit_fault.bit)
            for bit_fault in bit_faults
        ],
        dtype=object,
    )

    def perturb_image(x: tf.Tensor) -> tf.Tensor:
        """Change the pixels of the [x] according to [bit_faults].

        Args:
            x (tf.Tensor): A 2D RGB image.
                    shape = (x, y, rgb=3)

        Returns:
            tf.Tensor: 2D RGB image.
        """
        updates = [
            action(x[indices[i, 0], indices[i, 1], indices[i, 2]])
            for i, action in enumerate(bit_actions)
        ]

        return tf.tensor_scatter_nd_update(
            x,
            indices,
            updates,
        )

    return perturb_image


def original_perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb

    return imgs


def original_perturb_image_by_bit_fault(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[𝑥,𝑦,rgb,bit,action], ...]
        bits = np.split(x, len(x) // 5)
        for bit2fault in bits:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, rgb, bit, action = bit2fault
            if action == 1:
                img[x_pos, y_pos, rgb] = bit_set(img[x_pos, y_pos, rgb], bit)
            elif action == 2:
                img[x_pos, y_pos, rgb] = bit_reset(img[x_pos, y_pos, rgb], bit)
            elif action == 3:
                img[x_pos, y_pos, rgb] = bit_flip(img[x_pos, y_pos, rgb], bit)
    return imgs