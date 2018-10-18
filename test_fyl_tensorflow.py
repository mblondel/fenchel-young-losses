# Author: Mathieu Blondel
# License: Simplified BSD

import tensorflow as tf

from fyl_tensorflow import squared_loss
from fyl_tensorflow import logistic_loss
from fyl_tensorflow import logistic_ova_loss
from fyl_tensorflow import sparsemax_loss


def test_squared_loss():
    with tf.Session() as sess:
        y_true = tf.constant([[1.7], [2.76], [2.09]])
        y_pred = tf.constant([[0.7], [3.76], [1.09]])
        loss = squared_loss(y_true, y_pred)
        error = tf.test.compute_gradient_error(y_pred, y_pred.shape,
                                               loss, loss.shape)
        assert error < 1e-3


def test_classification_losses():
    for loss_func in (logistic_loss, logistic_ova_loss, sparsemax_loss):
        with tf.Session() as sess:
            y_true = tf.constant([0, 0, 1, 2])
            theta = tf.random_normal((4, 3))
            loss = loss_func(y_true, theta)
            error = tf.test.compute_gradient_error(theta, theta.shape,
                                                   loss, loss.shape)
            assert error < 1e-3


def test_classification_losses_multilabel():
    for loss_func in (logistic_loss, logistic_ova_loss, sparsemax_loss):
        with tf.Session() as sess:
            y_true = tf.constant([[1, 0, 0],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
            theta = tf.random_normal((4, 3))
            loss = loss_func(y_true, theta)
            error = tf.test.compute_gradient_error(theta, theta.shape,
                                                   loss, loss.shape)
            assert error < 1e-3
