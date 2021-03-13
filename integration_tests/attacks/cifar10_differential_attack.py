"""Attack cifar and a model using differential attack.

NOTE: The functions declaration should be ordered by call.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from inputtensorfi import InputTensorFI
from inputtensorfi.attacks.utils import attack
from inputtensorfi.helpers import utils
from inputtensorfi.layers import PixelFiLayerTF
from inputtensorfi.manipulation.img.faults import PixelFault
from inputtensorfi.manipulation.img.utils import build_perturb_image
from integration_tests.models.my_vgg import my_vgg

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FILE_PATH, "../models/my_vgg.h5")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def __prepare_datasets():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)


def __prepare_model(data_train, data_test):
    if os.path.exists(MODEL_PATH):
        print("---Using Existing Model---")
        model: tf.keras.Model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("---Training Model---")
        print(f"GPU IS AVAILABLE: {tf.config.list_physical_devices('GPU')}")
        model: tf.keras.Model = my_vgg()
        model.fit(
            *data_train,
            epochs=100,
            batch_size=64,
            validation_data=data_test,
        )
        model.save(MODEL_PATH)

    model.summary()
    return model


def __find_fragile_images(
    data_test: np.ndarray,
    model: tf.keras.Model,
    fragility_threshold=0.1,
):
    """Look for images which are sensible to FI.

    "Fragile image" has these conditions :

    -  y_pred_index == y_true_index
    -  std(y_pred) < fragility_threshold
    """
    x_test, y_test = data_test
    result = model.predict(x_test)

    for i, y_pred in enumerate(result):
        y_true = y_test[i]
        y_true_index = np.argmax(y_true)
        y_pred_index = np.argmax(y_pred)

        if y_pred_index == y_true_index and np.std(y_pred) < fragility_threshold:
            print(
                f"image {i} is fragile.\n"
                f"std: {np.std(y_pred)}.\n"
                f"y_pred[y_true_index]={y_pred[y_true_index]}\n"
                f"y_pred[0]={y_pred[0]}\n"
            )
            yield i


def _evaluate_one(
    image_id: int,
    data_test: np.ndarray,
    model: tf.keras.Model,
):
    x_test, y_test = data_test
    x = x_test[image_id]
    y_true = y_test[image_id]
    y_true_index = np.argmax(y_true)

    result = model.predict(np.array([x]))[0]  # Predict one
    result_index = np.argmax(result)

    print(f"result={result}")
    print(f"result_index={result_index}")
    print(f"y_true={y_true}")
    print(f"y_true_index={y_true_index}")
    print(f"result[y_true_index]={result[y_true_index]}")


def _look_for_pixels(
    image_id: int,
    data_test: np.ndarray,
    model: tf.keras.Model,
):
    x_test, y_test = data_test
    x = x_test[image_id]
    y_true = y_test[image_id]
    y_true_index = np.argmax(y_true)
    pixels = attack(
        x,
        y_true_index,
        model,
        pixel_count=1,  # Number of pixels to attack
        verbose=True,
    ).astype(np.uint8)

    # Convert [x_0, y_0, r_0, g_0, b_0, x_1, ...]
    # to [pixel_fault_0, pixel_fault_1, ...]
    return np.array([PixelFault(*pixels[0:5]) for i in range(len(pixels) // 5)])


def test_cifar10_differential_attack():
    data_train, data_test = __prepare_datasets()
    model = __prepare_model(data_train, data_test)

    print("--Search for fragile images--")
    fragile_imgs = list(__find_fragile_images(data_test, model))

    print("---Evaluation---")
    image_id = min(fragile_imgs)  # Choose the most fragile
    _evaluate_one(image_id, data_test, model)

    print("---Look for x---")
    pixels = _look_for_pixels(image_id, data_test, model)
    print(f"pixels={pixels}")

    print("---Fault Injection---")
    faulted_model = InputTensorFI.build_faulted_model(
        model,
        fi_layers=[
            PixelFiLayerTF(pixels, dtype=tf.uint8),
        ],
    )

    print("---Evaluation with FI---")
    _evaluate_one(image_id, data_test, faulted_model)

    # Plot
    perturbated_image = build_perturb_image(pixels)(data_test[0][image_id])
    utils.plot_image(perturbated_image)


if __name__ == "__main__":
    test_cifar10_differential_attack()
