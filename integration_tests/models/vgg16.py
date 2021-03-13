import tensorflow as tf
from tensorflow.keras.regularizers import l2


def vgg16():
    weight_decay = 0.0005
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
                input_shape=(32, 32, 3),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                512,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(
                512,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(
                512,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    sgd = tf.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model
