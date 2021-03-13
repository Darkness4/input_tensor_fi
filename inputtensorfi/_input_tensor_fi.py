"""
InputTensorFI -- Add layers with FI capabilities to the input layer.
"""

from abc import ABCMeta
from typing import List, Optional

import tensorflow as tf

from inputtensorfi.layers import FiLayer


class InputTensorFI(metaclass=ABCMeta):
    """
    Add layers with FI capabilities to the input layer.
    """

    @staticmethod
    def build_faulted_model(
        model: tf.keras.Model,
        fi_layers: Optional[List[FiLayer]] = None,
    ) -> tf.keras.Model:
        """Prepend FI Layers at the model.

        Args:
            model (tf.keras.Model): Model to be fault injected.
            fi_layers (Optional[List[FiLayer]], optional):
                    A list of layers injecting faults. Defaults to None.

        Returns:
            tf.keras.Model: A model which the input is faulted.
        """
        if not fi_layers:
            fi_layers = []
            dtype = model.input.dtype
        else:
            dtype = fi_layers[0].dtype

        faulted_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Input(
                    dtype=dtype, shape=model.input_shape[1:]
                ),
                *fi_layers,
                model,
            ]
        )
        faulted_model.compile(
            loss=tf.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.CategoricalAccuracy()],
        )
        return faulted_model
