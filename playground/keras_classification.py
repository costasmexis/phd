import pandas as pd
import numpy as np

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, X_test.shape)

# Flatten images into one-dimensional vector, each of size 1 x 28 x 28 = 1 x 784
num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
print(X_train.shape, X_test.shape)

# Since pixel values can range from 0 to 255, we will normalize the vectors to be between 0 and 1

X_train = X_train / 255
X_test = X_test / 255

# Befor we start building out model, remember that for classification we need to divide our target variable into categories. We use the to_categorical function from the Keras Utilities package.

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

# Build ANN for Classification
import tensorflow as tf

def classification_model():
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.metrics.categorical_crossentropy, metrics=['accuracy'])
    return model

model = classification_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=1)

y_pred = model.predict(X_test)

print("Accuracy:", scores[1])
