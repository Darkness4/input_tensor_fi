from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf

from inputtensorfi.manipulation.img.faults import PixelFault
from inputtensorfi.manipulation.img.utils import build_perturb_image


CORNERS = (
    (0, 0, 0),
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
)


def corner_search(
    image_id: int,
    pixels: np.ndarray,
    data_test: np.ndarray,
    model: tf.keras.Model,
) -> Iterable[Tuple[np.ndarray, np.ndarray, PixelFault]]:
    x_test, y_test = data_test

    y_true = y_test[image_id]
    y_true_index = np.argmax(y_true)

    for pixel in pixels:
        corner_pixels = [PixelFault(pixel.x, pixel.y, r, g, b) for r, g, b in CORNERS]

        x_fakes = np.array(
            [
                build_perturb_image([corner_pixel])(x_test[image_id])
                for corner_pixel in corner_pixels
            ]
        )
        y_preds = model.predict(x_fakes)

        for x_fake, y_pred, corner_pixel in zip(
            x_fakes,
            y_preds,
            corner_pixels,
        ):
            y_pred_index = np.argmax(y_pred)
            if y_true_index != y_pred_index:
                yield x_fake, y_pred, corner_pixel
