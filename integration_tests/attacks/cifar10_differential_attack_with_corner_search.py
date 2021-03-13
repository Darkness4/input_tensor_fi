"""Attack cifar and a model using differential attack.

NOTE: The functions declaration should be ordered by call.
"""

import json
import os
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from inputtensorfi.attacks.corner_search import corner_search
from inputtensorfi.attacks.utils import attack
from inputtensorfi.manipulation.img.faults import PixelFault
from integration_tests.models.my_vgg import my_vgg

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FILE_PATH, "../models/my_vgg.h5")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def __prepare_datasets():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test[:2000], y_test[:2000])


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
    pixel_count=1,
):
    x_test, y_test = data_test
    x = x_test[image_id]
    y_true = y_test[image_id]
    y_true_index = np.argmax(y_true)
    pixels = attack(
        x,
        y_true_index,
        model,
        pixel_count=pixel_count,
        maxiter=10,
        verbose=False,
    ).astype(np.uint8)

    # Convert [x_0, y_0, r_0, g_0, b_0, x_1, ...]
    # to [pixel_fault_0, pixel_fault_1, ...]
    return np.array([PixelFault(*pixels[0:5]) for i in range(len(pixels) // 5)])


def test_cifar10_differential_attack_with_corner_search():
    data_train, data_test = __prepare_datasets()
    x_test, y_test = data_test
    model = __prepare_model(data_train, data_test)

    y_preds = model.predict(x_test)

    length = len(y_preds)
    y_fake = y_preds.copy()
    total_faults: Dict[int, PixelFault] = dict()
    for image_id, _ in enumerate(y_test):
        if np.argmax(y_preds[image_id]) != np.argmax(y_test[image_id]):
            print(f"MISPREDICTED {image_id}/{length}")
            continue
        pixels = _look_for_pixels(image_id, data_test, model, pixel_count=10)

        try:
            first_pred = next(corner_search(image_id, pixels, data_test, model))
            _, y_pred, pixel = first_pred
            y_fake[image_id] = y_pred
            total_faults[image_id] = pixel
            print(
                f"FAULT {image_id}/{length}, {pixel}, original={data_test[0][image_id, pixel.x, pixel.y]}"
            )
        except StopIteration:
            # print(f"NO FAULT image_id={image_id}")
            pass

    dict_data = {key: fault.to_tuple() for key, fault in total_faults.items()}
    print(f"total_faults={json.dumps(dict_data, indent=2)}")

    y_true_acc = np.array([np.max(y) for y in y_test])
    y_preds_acc = np.array([y[np.argmax(y_true)] for y, y_true in zip(y_preds, y_test)])
    y_fake_acc = np.array([y[np.argmax(y_true)] for y, y_true in zip(y_fake, y_test)])
    print(f"y_true_acc={np.mean(y_true_acc)}")
    print(f"y_prior_acc={np.mean(y_preds_acc)}")
    print(f"y_fake_acc={np.mean(y_fake_acc)}")


if __name__ == "__main__":
    test_cifar10_differential_attack_with_corner_search()
