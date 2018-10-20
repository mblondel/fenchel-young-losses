# Author: Mathieu Blondel
# License: Simplified BSD

"""
NumPy implementation of

Learning Classifiers with Fenchel-Young Losses:
    Generalized Entropies, Margins, and Algorithms.
Mathieu Blondel, André F. T. Martins, Vlad Niculae.
https://arxiv.org/abs/1805.09717
"""


import numpy as np


def conjugate_function(theta, grad, Omega):
    return np.sum(theta * grad, axis=1) - Omega(grad)


class FYLoss(object):

    def forward(self, y_true, theta):
        y_true = np.array(y_true)

        self.y_pred = self.predict(theta)
        ret = conjugate_function(theta, self.y_pred, self.Omega)

        if len(y_true.shape) == 2:
            # y_true contains label proportions
            ret += self.Omega(y_true)
            ret -= np.sum(y_true * theta, axis=1)

        elif len(y_true.shape) == 1:
            # y_true contains label integers (0, ..., n_classes-1)

            if y_true.dtype != np.long:
                raise ValueError("y_true should contains long integers.")

            all_rows = np.arange(y_true.shape[0])
            ret -= theta[all_rows, y_true]

        else:
            raise ValueError("Invalid shape for y_true.")

        return np.sum(ret)


    def __call__(self, y_true, theta):
        return self.forward(y_true, theta)


class SquaredLoss(FYLoss):

    def Omega(self, mu):
        return 0.5 * np.sum((mu ** 2), axis=1)

    def predict(self, theta):
        return theta


class PerceptronLoss(FYLoss):

    def predict(self, theta):
        theta = np.array(theta)
        ret = np.zeros_like(theta)
        all_rows = np.arange(theta.shape[0])
        ret[all_rows, np.argmax(theta, axis=1)] = 1
        return ret

    def Omega(self, theta):
        return np.zeros(len(theta))


def Shannon_negentropy(p, axis):
    p = np.array(p)
    tmp = np.zeros_like(p)
    mask = p > 0
    tmp[mask] = p[mask] * np.log(p[mask])
    return np.sum(tmp, axis)


class LogisticLoss(FYLoss):

    def predict(self, theta):
        exp_theta = np.exp(theta - np.max(theta, axis=1)[:, np.newaxis])
        return exp_theta / np.sum(exp_theta, axis=1)[:, np.newaxis]

    def Omega(self, p):
        return Shannon_negentropy(p, axis=1)


class Logistic_OVA_Loss(FYLoss):

    def predict(self, theta):
        return 1. / (1 + np.exp(-theta))

    def Omega(self, p):
        return Shannon_negentropy(p, axis=1) + Shannon_negentropy(1 - p, axis=1)


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2

    z: float or array
        If array, len(z) must be compatible with V

    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    V = np.array(V)

    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


class SparsemaxLoss(FYLoss):

    def predict(self, theta):
        return projection_simplex(theta, axis=1)

    def Omega(self, p):
        p = np.array(p)
        return 0.5 * np.sum((p ** 2), axis=1) - 0.5


# FIXME: implement bisection in Numba.
def _bisection(theta, omega_p, omega_p_inv, max_iter=20, tol=1e-3):
    theta = np.array(theta)
    t_min = np.max(theta, axis=1) - omega_p(1.0)
    t_max = np.max(theta, axis=1) - omega_p(1.0 / theta.shape[1])
    p = np.zeros_like(theta)

    for i in range(len(theta)):

        thresh = omega_p(0)

        for it in range(max_iter):
            t = (t_min[i] + t_max[i]) / 2.0
            p[i] = omega_p_inv(np.maximum(theta[i] - t, thresh))
            f = np.sum(p[i]) - 1
            if f < 0:
                t_max[i] = t
            else:
                t_min[i] = t
            if np.abs(f) < tol:
                break

    return p


class TsallisLoss(FYLoss):

    def __init__(self, alpha=1.5, max_iter=20, tol=1e-3):
        if alpha < 1:
            raise ValueError("alpha should be greater or equal to 1.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def predict(self, theta):
        # Faster algorithms for specific cases.
        if self.alpha == 1:
            return LogisticLoss().predict(theta)

        if self.alpha == 2:
            return SparsemaxLoss().predict(theta)

        if self.alpha == np.inf:
            return PerceptronLoss().predict(theta)

        # General case.
        am1 = self.alpha - 1

        def omega_p(t):
            return (t ** am1 - 1.) / am1

        def omega_p_inv(s):
            return (1 + am1 * s) ** (1. / am1)

        return _bisection(theta, omega_p, omega_p_inv, self.max_iter, self.tol)

    def Omega(self, p):
        p = np.array(p)

        if self.alpha == 1:
            # We need to handle the limit case to avoid division by zero.
            return LogisticLoss().Omega(p)

        scale = self.alpha * (self.alpha - 1)
        return (np.sum((p ** self.alpha), axis=1) - 1.) / scale
