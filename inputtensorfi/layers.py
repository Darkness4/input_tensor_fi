import logging
from abc import ABCMeta

import numpy as np
import tensorflow as tf
from inputtensorfi.manipulation.img.utils import (
    build_perturb_image,
    build_perturb_image_by_bit_fault,
)


class FiLayer(tf.keras.layers.Layer, metaclass=ABCMeta):
    """A Keras layer marked with fault injection.

    This is an abstrac class. Implementations are stored along this class.
    """

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


class PixelFiLayer(FiLayer):
    """A layer that modify the pixels of the input.

    Only accepts 2D RGB 8 bit images."""

    def __init__(self, pixels: np.ndarray, dtype: tf.DType = tf.uint8):
        super(PixelFiLayer, self).__init__(dtype=dtype)
        self.pixels = pixels
        self.perturb_image = build_perturb_image(pixels)

    def call(self, input, training=False):
        if not training:
            return tf.numpy_function(self.perturb_image, [input], self.dtype)
        else:
            logging.warning("PixelBitFiLayer is ignored on training.")
            return input

    def get_config(self):
        base_config = super(PixelFiLayer, self).get_config()
        base_config["pixels"] = [pixel.to_dict() for pixel in self.pixels]
        return base_config


class PixelBitFiLayer(FiLayer):
    def __init__(self, bit_faults: np.ndarray, dtype: tf.DType = tf.uint8):
        super(PixelBitFiLayer, self).__init__(dtype=dtype)
        self.bit_faults = bit_faults
        self.perturb_image = build_perturb_image_by_bit_fault(bit_faults)

    def call(self, input, training=False):
        if not training:
            return tf.numpy_function(self.perturb_image, [input], self.dtype)
        else:
            logging.warning("PixelBitFiLayer is ignored on training.")
            return input

    def get_config(self):
        base_config = super(PixelBitFiLayer, self).get_config()
        base_config["bit_faults"] = [
            bit_fault.to_dict() for bit_fault in self.bit_faults
        ]
        return base_config


class _ModelLayer(tf.keras.layers.Layer):
    """A model layer running a model.

    Deprecated. Use tf.keras.Model directly in the layers list without worries.
    """

    def __init__(self, model: tf.keras.Model):
        super(_ModelLayer, self).__init__(dtype=model.dtype)
        logging.error(
            "ModelLayer is deprecated. Use the tf.keras.Model "
            "directly in the layers list without worries."
        )
        self.model = model
        self.model_run = self.__build_model_run(model)

    def call(self, input):
        logging.error(
            "ModelLayer is deprecated. Use the tf.keras.Model "
            "directly in the layers list without worries."
        )
        return tf.numpy_function(self.model_run, [input], self.dtype)

    def get_config(self):
        base_config = super(_ModelLayer, self).get_config()
        base_config["model"] = self.model
        return base_config

    @staticmethod
    def __build_model_run(model: tf.keras.Model):
        def model_run(inputs):
            return model.predict(inputs)

        return model_run
