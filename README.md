# InputTensorFI

Dynamic Injection before the first layer.

## Configuration

### Requirements

- Python 3.8+

### Dependencies installation (choose one)

#### Pipenv (recommended)

```sh
pipenv install
```

#### Pip

```sh
pip install -r requirements.txt
```

## Usage

(Using [cifar10_test](./integration_tests/pixels/cifar10_test.py) as an example.)

### Install the package (choose one)

#### Install with pip

```sh
pip install -e .
```

#### Use PYTHONPATH

```sh
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Demonstration

#### Setup

Assuming our model :

```python
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
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.CategoricalAccuracy()],
    )
    return model

```

Training :

```python
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = my_vgg()
model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_test, y_test),
)
model.save(MODEL_PATH)

model.summary()
```

```sh
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 128)         73856
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 128)         147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 128)               262272
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 550,570
Trainable params: 550,570
Non-trainable params: 0
_________________________________________________________________
```

Evaluation :

```python
loss, acc = model.evaluate(x_test, y_test)

print(f"loss={loss}, acc={acc}")
```

```sh
313/313 [==============================] - 4s 7ms/step - loss: 1.2258 - categorical_accuracy: 0.7685
loss=1.2258156538009644, acc=0.7684999704360962
```

#### Fault Injection

```python
from inputtensorfi import InputTensorFI
from inputtensorfi.layers import PixelFiLayer
from inputtensorfi.manipulation.img.faults import PixelFault

pixels = np.array(  # Triple faults injection for each image
    [
        PixelFault(5, 5, 0, 0, 0),  # x, y, r, g, b
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
faulted_model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
pixel_fi_layer (PixelFiLayer None                      0
_________________________________________________________________
sequential (Sequential)      (None, 10)                550570
=================================================================
Total params: 550,570
Trainable params: 550,570
Non-trainable params: 0
_________________________________________________________________
```

Evaluation :

```python
loss, acc = faulted_model.evaluate(x_test, y_test)

print(f"loss={loss}, acc={acc}")
```

```sh
313/313 [==============================] - 50s 157ms/step - loss: 1.5091 - categorical_accuracy: 0.7165
loss=1.5162289142608643, acc=0.7128000259399414
```

## Running unit tests

Make sure to have installed **PyTest** (`pipenv install --dev` or `pip install pytest`).

```python
python -m pytest tests
```

## Running integration tests

Do not forget to set the **PYTHONPATH**.

Run the an integration test:

```python
python ./integration_tests/[A TEST].py
```

## LICENSE

```txt
MIT License

Copyright (c) Marc NGUYEN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
