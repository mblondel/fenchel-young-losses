# Author: Mathieu Blondel
# License: Simplified BSD

import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(currentdir))

import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises

from fyl_numpy import SquaredLoss
from fyl_numpy import PerceptronLoss
from fyl_numpy import LogisticLoss
from fyl_numpy import Logistic_OVA_Loss
from fyl_numpy import SparsemaxLoss
from fyl_numpy import TsallisLoss


def test_squared_loss():
    y_true = np.array([[1.7], [2.76], [2.09]])
    y_pred = np.array([[0.7], [3.76], [1.09]])
    loss = SquaredLoss()
    got = loss(y_true, y_pred)
    expected = 0.5 * np.mean((y_true - y_pred) ** 2)
    assert_almost_equal(got, expected, 3)

    assert_raises(ValueError, loss.forward, y_true.ravel(), y_pred)


def test_classification_losses():
    losses = [PerceptronLoss(), LogisticLoss(), Logistic_OVA_Loss(),
              SparsemaxLoss()]
    for alpha in (1.0, 1.5, 2.0):
        losses.append(TsallisLoss(alpha=alpha))

    for loss in losses:
        y_true = np.array([0, 0, 1, 2])
        rng = np.random.RandomState(0)
        theta = rng.randn(y_true.shape[0], 3)
        loss(y_true, theta)


def test_classification_losses_multilabel():
    for loss in (LogisticLoss(), Logistic_OVA_Loss(), SparsemaxLoss(),
                 TsallisLoss()):
        y_true = np.array([[1, 0, 0],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        rng = np.random.RandomState(0)
        theta = rng.randn(*y_true.shape)
        loss(y_true, theta)
