# Author: Mathieu Blondel
# License: Simplified BSD

import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(currentdir))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from fyl_tensorflow import squared_loss

tf.random.set_seed(0)
# Hyper-parameters
num_epochs = 60
learning_rate = 0.001

# Toy dataset
X = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]])

y = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1,
                          use_bias=True,
                          activation="linear"),
])

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
# optimizer = "SGD"

model.compile(optimizer=optimizer,
              loss=squared_loss,
              metrics='mae')

model.fit(X, y, epochs=num_epochs)
# model.evaluate(X, y)

# Plot the graph
y_pred = model.predict(X)
plt.plot(X, y, 'ro', label='Observations')
plt.plot(X, y_pred, label='Fitted line')
plt.legend()
plt.show()
