# Author: Mathieu Blondel
# License: Simplified BSD

import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(currentdir))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from fyl_tensorflow import logistic_loss
from fyl_tensorflow import logistic_ova_loss
from fyl_tensorflow import sparsemax_loss

# Loss
losses = {
    "logistic": logistic_loss,
    "logistic-ova": logistic_ova_loss,
    "sparsemax": sparsemax_loss,
}

if len(sys.argv) == 1:
    loss = sparsemax_loss
elif sys.argv[1] in losses:
    loss = losses[sys.argv[1]]
else:
    raise ValueError("Invalid loss.")


# Hyper-parameters
num_epochs = 100
learning_rate = 0.001

# Toy dataset: we create 40 separable points.
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = np.array([0] * 20 + [1] * 20, dtype=np.long)
Y = None

# Can also use a one-hot encoded matrix.
Y = np.zeros((X.shape[0], 2))
Y[np.arange(X.shape[0]), y] = 1

tf.random.set_seed(0)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2,
                          use_bias=True,
                          activation="linear"),
])

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
#optimizer = "SGD"

model.compile(optimizer=optimizer,
              loss=loss,
              metrics='accuracy')

if Y is None:
    model.fit(X, y, epochs=num_epochs)
else:
    model.fit(X, Y, epochs=num_epochs)

weights = model.weights[0].numpy()
bias = model.weights[1].numpy()

# Get the separating hyperplane.
w = weights[1] - weights[0]
b = bias[1] - bias[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - b / w[1]

plt.plot(xx, yy, 'k-')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
