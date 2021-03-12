import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from inputtensorfi import InputTensorFI
from inputtensorfi.layers import PixelFiLayer, PixelFiLayerTF
from inputtensorfi.manipulation.img.faults import PixelFault
from integration_tests.models.my_vgg import my_vgg
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

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


def test_cifar10_one():
    data_train, data_test = __prepare_datasets()
    model = __prepare_model(data_train, data_test)

    print("---Evaluation---")
    image_id = 541
    x_test, y_test = data_test
    x = x_test[image_id]
    y_true = y_test[image_id]
    result = model.predict(np.array([x]))

    print(f"result={result}")
    print(f"y_true={y_true}")

    print("---Fault Injection---")
    pixels = np.array(
        [
            PixelFault(5, 5, 0, 0, 0),
            PixelFault(6, 6, 0, 0, 0),
            PixelFault(7, 7, 0, 0, 0),
        ],
        dtype=object,
    )

    faulted_model = InputTensorFI.build_faulted_model(
        model,
        fi_layers=[
            PixelFiLayerTF(pixels, dtype=tf.uint8),
        ],
    )

    print("---Evaluation with FI---")
    result = faulted_model.predict(np.array([x_test[image_id]]))
    print(f"result={result}")
    print(f"y_true={y_true}")


if __name__ == "__main__":
    test_cifar10_one()
