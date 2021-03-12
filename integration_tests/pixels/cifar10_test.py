import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from inputtensorfi import InputTensorFI
from inputtensorfi.layers import PixelFiLayer
from inputtensorfi.manipulation.img.faults import PixelFault
from integration_tests.models.my_vgg import my_vgg
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FILE_PATH, "../models/my_vgg.h5")


def test_cifar10():
    # -- Train --
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if os.path.exists(MODEL_PATH):
        print("---Using Existing Model---")
        model: tf.keras.Model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("---Training Model---")
        print(f"GPU IS AVAILABLE: {tf.config.list_physical_devices('GPU')}")
        model: tf.keras.Model = my_vgg()
        model.fit(
            x_train,
            y_train,
            epochs=100,
            batch_size=64,
            validation_data=(x_test, y_test),
        )
        model.save(MODEL_PATH)

    model.summary()

    print("---Evaluation---")
    loss, acc = model.evaluate(x_test, y_test)

    print(f"loss={loss}, acc={acc}")

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
            PixelFiLayer(pixels, dtype=tf.uint8),
        ],
    )

    logdir = "logs/compile/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("---Evaluation with FI---")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    loss, acc = faulted_model.evaluate(
        x_test, y_test, callbacks=[tensorboard_callback]
    )
    print(f"loss={loss}, acc={acc}")


if __name__ == "__main__":
    test_cifar10()
