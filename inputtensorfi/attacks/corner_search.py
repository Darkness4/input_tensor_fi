import itertools
from typing import Dict, List, Tuple, Iterable

import numpy as np
import tensorflow as tf
from inputtensorfi.manipulation.img.faults import PixelFault
from inputtensorfi.manipulation.img.utils import build_perturb_image


def corner_search(
    image_id: int,
    pixels: np.ndarray,
    data_test: np.ndarray,
    model: tf.keras.Model,
) -> Iterable[Tuple[np.ndarray]]:
    x_test, y_test = data_test
    corners_gen = ((0, 255), (0, 255), (0, 255))
    corners = list(itertools.product(*corners_gen))
    # corners = [(0, 0, 0), (0, 0, 255), ..., (255, 255, 255)]

    y_true = y_test[image_id]
    y_true_index = np.argmax(y_true)

    for pixel in pixels:
        corner_pixels = [
            PixelFault(pixel.x, pixel.y, r, g, b) for r, g, b in corners
        ]

        x_fakes = np.array(
            [
                build_perturb_image([corner_pixel])(x_test[image_id])
                for corner_pixel in corner_pixels
            ]
        )
        y_preds = model.predict(x_fakes)

        for x_fake, y_pred in zip(x_fakes, y_preds):
            y_pred_index = np.argmax(y_pred)
            if y_true_index != y_pred_index:
                yield x_fake, y_pred
