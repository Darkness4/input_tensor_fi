import numpy as np
import tensorflow as tf
from inputtensorfi.layers import PixelBitFiLayer, PixelFiLayer
from inputtensorfi.manipulation.bit.action import BitFlip
from inputtensorfi.manipulation.img.faults import BitFault, PixelFault
from tensorflow.keras.datasets import cifar10


def test_PixelFiLayer():
    pixels = np.array(
        [PixelFault(16, 16, 255, 255, 0), PixelFault(5, 5, 255, 255, 0)],
        dtype=object,
    )
    (_, _), (x_test, _) = cifar10.load_data()
    image = x_test[0]

    model = tf.keras.Sequential(layers=[PixelFiLayer(pixels)])

    image_perturbed: np.ndarray = model.predict(image)

    assert np.array_equal(image_perturbed[16, 16], np.array((255, 255, 0)))
    assert np.array_equal(image_perturbed[5, 5], np.array((255, 255, 0)))


def test_BitFiLayer():
    bit_faults = np.array(
        [BitFault(16, 16, 0, 3, BitFlip)],
        dtype=object,
    )
    (_, _), (x_test, _) = cifar10.load_data()
    image = x_test[0]

    model = tf.keras.Sequential(layers=[PixelBitFiLayer(bit_faults)])

    image_perturbed: np.ndarray = model.predict(image)

    assert image[16, 16, 0] != image_perturbed[16, 16, 0]
