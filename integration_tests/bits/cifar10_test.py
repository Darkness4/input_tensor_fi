import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from inputtensorfi import InputTensorFI
from inputtensorfi.layers import PixelBitFiLayerTF
from inputtensorfi.manipulation.bit.action import BitFlip
from inputtensorfi.manipulation.img.faults import BitFault
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
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.125,
            height_shift_range=0.125,
            fill_mode="constant",
            cval=0.0,
        )
        datagen.fit(data_train[0])
        model: tf.keras.Model = my_vgg()
        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        model.fit(
            *data_train,
            batch_size=128,
            steps_per_epoch=391,
            epochs=200,
            validation_data=data_test,
            callbacks=[tensorboard_callback],
        )
        model.save(MODEL_PATH)

    model.summary()
    return model


def test_cifar10():
    data_train, data_test = __prepare_datasets()
    model = __prepare_model(data_train, data_test)

    print("---Evaluation---")
    loss, acc = model.evaluate(*data_test)

    print(f"loss={loss}, acc={acc}")

    print("---Fault Injection---")
    bit_faults = np.array(
        [
            BitFault(5, 5, 0, 7, BitFlip),
            BitFault(6, 6, 0, 7, BitFlip),
            BitFault(7, 7, 0, 7, BitFlip),
        ],
        dtype=object,
    )

    faulted_model = InputTensorFI.build_faulted_model(
        model,
        fi_layers=[
            PixelBitFiLayerTF(bit_faults, dtype=tf.uint8),
        ],
    )

    print("---Evaluation with FI---")
    loss, acc = faulted_model.evaluate(*data_test)
    print(f"loss={loss}, acc={acc}")


if __name__ == "__main__":
    test_cifar10()
