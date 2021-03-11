from inputtensorfi.manipulation.bit.action import BitFlip
from inputtensorfi.manipulation.img.bit_fault import BitFault
import numpy as np
from inputtensorfi.manipulation.img import utils
from inputtensorfi.manipulation.img.pixel_fault import PixelFault
from tensorflow.keras.datasets import cifar10


def test_perturb_image():
    pixels = np.array(
        [PixelFault(16, 16, 255, 255, 0), PixelFault(5, 5, 255, 255, 0)],
        dtype=object,
    )
    (_, _), (x_test, _) = cifar10.load_data()
    image = x_test[0]
    images_pertubed = utils.perturb_image(pixels, image)

    assert np.array_equal(images_pertubed[16, 16], np.array((255, 255, 0)))
    assert np.array_equal(images_pertubed[5, 5], np.array((255, 255, 0)))


def test_perturb_image_by_bit_fault():
    bit_faults = np.array(
        [BitFault(16, 16, 0, 3, BitFlip)],
        dtype=object,
    )
    (_, _), (x_test, _) = cifar10.load_data()
    image = x_test[0]
    images_pertubed = utils.perturb_image_by_bit_fault(bit_faults, image)

    assert image[16, 16, 0] != images_pertubed[16, 16, 0]
