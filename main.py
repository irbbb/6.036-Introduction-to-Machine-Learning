# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:14:05 2022

@author: Ivan
"""

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(trainImages, trainLabels), (testImages, testLabels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainImages, trainLabels, epochs=5)

testLoss, testAcc = model.evaluate(testImages, testLabels)

model.fit()
