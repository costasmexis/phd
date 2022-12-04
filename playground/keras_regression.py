import pandas as pd
import numpy as np

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=10, n_informative=4)
print(X.shape, y.shape)

# Train and test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=4)

import tensorflow as tf

from sklearn.metrics import mean_squared_error
# Build a ANN for regression

def regression_model():
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.metrics.mean_squared_error)
    return model

n_cols = X.shape[1]

model = regression_model()

EPOCHS = 100
BATCH_SIZE = 10
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

