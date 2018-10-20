# Author: Mathieu Blondel
# License: Simplified BSD

"""
Tensorflow implementation of

Learning Classifiers with Fenchel-Young Losses:
    Generalized Entropies, Margins, and Algorithms.
Mathieu Blondel, André F. T. Martins, Vlad Niculae.
https://arxiv.org/abs/1805.09717
"""

import tensorflow as tf


def fy_loss(y_true, theta, predict, Omega):
    @tf.custom_gradient
    def Omega_conjugate(theta):
        y_pred = predict(theta)

        def grad(g):
            # We ignore g since the loss should be the last layer and so g=1.
            return y_pred

        return tf.reduce_sum(theta * y_pred, axis=1) - Omega(y_pred), grad

    ret = Omega_conjugate(theta)

    if len(y_true.shape) > 2:
        raise ValueError("y_true should be either 2d (reals) or 1d (integers)")

    if len(y_true.shape) == 2 and y_true.shape[1] != theta.shape[1]:
        # Keras seems to silently reshape y_true when len(shape) == 1...
        # Workaround: flatten it again...
        y_true = tf.reshape(y_true, [-1])

    if y_true.shape[0] != theta.shape[0]:
        raise ValueError("y_true.shape[0] and theta.shape[0] should agree")

    if len(y_true.shape) == 1:
        y_true = tf.cast(y_true, tf.int32)
        all_rows = tf.range(y_true.shape[0])
        full_indices = tf.stack([all_rows, y_true], axis=1)
        tmp = tf.gather_nd(theta, full_indices)
        ret -= tmp

    else:
        y_true = tf.cast(y_true, theta.dtype)
        ret += Omega(y_true)
        ret -= tf.reduce_sum(y_true * theta, axis=1)

    return tf.reduce_sum(ret)


def squared_loss(y_true, theta):

    def Omega(mu):
        return 0.5 * tf.reduce_sum(tf.square(mu), axis=1)


    def predict(theta):
        return theta

    return fy_loss(y_true, theta, predict, Omega)


def Shannon_negentropy(p, axis):
    tmp = p * tf.log(p)
    tmp = tf.where(tf.is_nan(tmp), tf.zeros_like(tmp), tmp)
    return tf.reduce_sum(tmp, axis)


def logistic_loss(y_true, theta):

    def predict(theta):
        return tf.nn.softmax(theta, axis=1)

    def Omega(p):
        return Shannon_negentropy(p, axis=1)

    return fy_loss(y_true, theta, predict, Omega)


def logistic_ova_loss(y_true, theta):

    def predict(theta):
        return tf.nn.sigmoid(theta)

    def Omega(p):
        return Shannon_negentropy(p, axis=1) + Shannon_negentropy(1-p, axis=1)

    return fy_loss(y_true, theta, predict, Omega)


def sparsemax_loss(y_true, theta):

    def predict(theta):
        return tf.contrib.sparsemax.sparsemax(theta)

    def Omega(p):
        return 0.5 * tf.reduce_sum((p ** 2), axis=1) - 0.5

    return fy_loss(y_true, theta, predict, Omega)
