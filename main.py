# coding=utf-8
# Copyright 2024 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CIFAR100-N dataset."""

from __future__ import annotations
import tensorflow as tf
from tensorflow import keras


def shapes(*x):
    for idx, item in enumerate(x):
        print(f"arg_{idx}: {item.shape}") if hasattr(item, "shape") else print(f"arg_{idx}: {len(item)}")



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

print('\n img_train.shape: ')
print(x_train.shape)

print('\n img_train[0].shape: ')
print(x_train[0].shape)

x_train, x_test = x_train/255, x_test/255
y_train, y_test = tf.keras.utils.to_categorical(y_train, 100), tf.keras.utils.to_categorical(y_test, 100)

shapes(x_train, x_test, y_train, y_test, y_train, y_test)

img_inputs = tf.keras.Input(shape=(32, 32, 3))



x_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)


x_2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)


x_3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)


x_4 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)


x_5 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)


x_6 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)


x_7 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)


x_8 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Conv2D(32, 2, 1), 
    tf.keras.layers.Conv2D(32, 2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
])(img_inputs)

x_1_1 =tf.keras.layers.Dense(1024)(keras.layers.Concatenate()([x_1, x_2]))

x_1_2 = tf.keras.layers.Dense(1024)(keras.layers.Concatenate()([x_3, x_4]))

x_2_1 =tf.keras.layers.Dense(1024)(keras.layers.Concatenate()([x_1_1, x_5]))

x_2_2 =tf.keras.layers.Dense(1024)(keras.layers.Concatenate()([x_1_2, x_6]))

x_3_1 =tf.keras.layers.Dense(1024)(keras.layers.Concatenate()([x_2_1, x_7]))

x_3_2 =tf.keras.layers.Dense(1024)(keras.layers.Concatenate()([x_2_2, x_8]))

x_4 = tf.keras.layers.Dense(1024, activation='relu')(keras.layers.LayerNormalization()(keras.layers.Concatenate()([x_3_1, x_3_2])))

outputs = tf.keras.layers.Dense(100, activation='softmax')(x_4)

model = keras.Model(inputs=img_inputs, outputs=outputs, name="cifar_model")

opt = keras.optimizers.Adam(learning_rate=0.000001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print('model.summary(): ')

model.summary()

print('\n \n \n ')

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)



















# ds = tfds.load('imagenet2012', split='train', shuffle_files=True)







print('\n run successful')