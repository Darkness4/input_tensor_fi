import logging
from abc import ABCMeta
from typing import Callable

import numpy as np
import tensorflow as tf

from inputtensorfi.manipulation.img.utils import (
    build_perturb_image,
    build_perturb_image_by_bit_fault,
    build_perturb_image_tensor,
)


class FiLayer(tf.keras.layers.Layer, metaclass=ABCMeta):
    """A Keras layer marked with fault injection.

    This is an abstrac class. Implementations are stored along this class.
    """

    perturb_image: Callable

    def __init__(
        self,
        name=None,
        dtype=tf.uint8,
        dynamic=False,
        **kwargs,
    ):
        super(FiLayer, self).__init__(
            trainable=False, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )

    def call(self, input, training=False):
        if not training:
            return tf.map_fn(
                fn=self.perturb_image,
                elems=input,
            )
        else:
            logging.warning(f"{type(self).__name__} is ignored on training.")
            return input


class PixelFiLayer(FiLayer):
    """A layer that modify the pixels of the input.

    Only accepts 2D RGB 8 bit images."""

    def __init__(self, pixels: np.ndarray, dtype: tf.DType = tf.uint8):
        super(PixelFiLayer, self).__init__(dtype=dtype)
        self.pixels = pixels
        transform = build_perturb_image(pixels)
        self.perturb_image = lambda x: tf.numpy_function(
            transform, [x], self.dtype
        )

    def get_config(self):
        base_config = super(PixelFiLayer, self).get_config()
        base_config["pixels"] = [pixel.to_dict() for pixel in self.pixels]
        return base_config


class PixelBitFiLayer(FiLayer):
    def __init__(self, bit_faults: np.ndarray, dtype: tf.DType = tf.uint8):
        super(PixelBitFiLayer, self).__init__(dtype=dtype)
        self.bit_faults = bit_faults

        transform = build_perturb_image_by_bit_fault(bit_faults)
        self.perturb_image = (
            lambda x: tf.numpy_function(transform, [x], self.dtype),
        )

    def get_config(self):
        base_config = super(PixelBitFiLayer, self).get_config()
        base_config["bit_faults"] = [
            bit_fault.to_dict() for bit_fault in self.bit_faults
        ]
        return base_config


class PixelFiLayerTF(FiLayer):
    """A layer that modify the pixels of the input.

    Only accepts 2D RGB 8 bit images."""

    def __init__(self, pixels: np.ndarray, dtype: tf.DType = tf.uint8):
        super(PixelFiLayerTF, self).__init__(dtype=dtype)
        self.pixels = pixels
        self.perturb_image = build_perturb_image_tensor(pixels)

    def get_config(self):
        base_config = super(PixelFiLayerTF, self).get_config()
        base_config["pixels"] = [pixel.to_dict() for pixel in self.pixels]
        return base_config
