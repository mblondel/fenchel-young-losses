# Author: Mathieu Blondel
# License: Simplified BSD

import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(currentdir))


import numpy as np
from sklearn.datasets import make_classification
from fyl_sklearn import FYClassifier

X, y = make_classification(n_samples=10, n_features=5, n_informative=3,
                           n_classes=3, random_state=0)
clf = FYClassifier(loss="sparsemax")
clf.fit(X, y)
print(clf.predict_proba(X[:3]))


import numpy as np
from fyl_numpy import SparsemaxLoss

# integers between 0 and n_classes-1, shape = n_samples
y_true = np.array([0, 2])
# model scores, shapes = n_samples x n_classes
theta = np.array([[-2.5, 1.2, 0.5],
                  [2.2, 0.8, -1.5]])
loss = SparsemaxLoss()
# loss value
print(loss(y_true, theta))
# predictions (probabilities) are stored for convenience
print(loss.y_pred)
# can also recompute them from theta
print(loss.predict(theta))
# label proportions are also allowed
y_true = np.array([[0.8, 0.2, 0],
                   [0.1, 0.2, 0.7]])
print(loss(y_true, theta))


import torch
from fyl_pytorch import SparsemaxLoss

# integers between 0 and n_classes-1, shape = n_samples
y_true = torch.tensor([0, 2])
# model scores, shapes = n_samples x n_classes
theta = torch.tensor([[-2.5, 1.2, 0.5],
                      [2.2, 0.8, -1.5]])
loss = SparsemaxLoss()
# loss value (caution: reversed convention compared to numpy and tensorflow)
print(loss(theta, y_true))
# predictions (probabilities) are stored for convenience
print(loss.y_pred)
# can also recompute them from theta
print(loss.predict(theta))
# label proportions are also allowed
y_true = torch.tensor([[0.8, 0.2, 0],
                       [0.1, 0.2, 0.7]])
print(loss(theta, y_true))

import tensorflow as tf
from fyl_tensorflow import sparsemax_loss, sparsemax_predict

# integers between 0 and n_classes-1, shape = n_samples
y_true = tf.constant([0, 2])
# model scores, shapes = n_samples x n_classes
theta = tf.constant([[-2.5, 1.2, 0.5],
                     [2.2, 0.8, -1.5]])
# loss value
print(sparsemax_loss(y_true, theta))
# predictions (probabilities)
print(sparsemax_predict(theta))
# label proportions are also allowed
y_true = tf.constant([[0.8, 0.2, 0],
                      [0.1, 0.2, 0.7]])
print(sparsemax_loss(y_true, theta))
