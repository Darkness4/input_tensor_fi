import tensorflow as tf
from tensorflow.keras.regularizers import l2


def my_vgg():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
                padding="same",
                input_shape=(32, 32, 3),
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
                padding="same",
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
                padding="same",
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
                padding="same",
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(0.001),
            ),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model
