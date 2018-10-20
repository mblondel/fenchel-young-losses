# Author: Mathieu Blondel
# License: Simplified BSD

"""
Scikit-learn implementation of

Learning Classifiers with Fenchel-Young Losses:
    Generalized Entropies, Margins, and Algorithms.
Mathieu Blondel, André F. T. Martins, Vlad Niculae.
https://arxiv.org/abs/1805.09717
"""

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import add_dummy_feature

from fyl_numpy import LogisticLoss
from fyl_numpy import Logistic_OVA_Loss
from fyl_numpy import SparsemaxLoss
from fyl_numpy import PerceptronLoss
from fyl_numpy import TsallisLoss


class FYClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, loss="logistic", alpha=1.0, alpha_tsallis=1.5,
                 fit_intercept=True, solver="lbfgs", max_iter=100, tol=1e-5,
                 random_state=None,verbose=0):
        self.loss = loss
        self.alpha = alpha
        self.alpha_tsallis = alpha_tsallis
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def _get_loss(self):
        loss_funcs = {
            "logistic": LogisticLoss(),
            "logistic-ova": Logistic_OVA_Loss(),
            "sparsemax": SparsemaxLoss(),
            "perceptron": PerceptronLoss(),
            "tsallis": TsallisLoss(self.alpha_tsallis),
        }
        if self.loss not in loss_funcs:
            raise ValueError("Invalid loss function.")
        return loss_funcs[self.loss]

    def _solve_lbfgs(self, X, y):
        n_samples, n_features = X.shape
        if len(y.shape) == 1:
            # FIXME: avoid binarizing y.
            Y = LabelBinarizer().fit_transform(y)
        else:
            Y = y
        n_classes = Y.shape[1]

        loss_func = self._get_loss()
        all_rows = np.arange(n_samples)

        def _func(coef):
            coef = coef.reshape(n_classes, n_features)

            # n_samples x n_features
            theta = safe_sparse_dot(X, coef.T)

            # n_samples, n_samples x n_classes
            loss = loss_func(y, theta)

            # n_classes x n_features
            grad = safe_sparse_dot(loss_func.y_pred.T, X)
            grad -= safe_sparse_dot(Y.T, X)

            # Regularization term
            loss += 0.5 * self.alpha * np.sum(coef ** 2)
            grad += self.alpha * coef

            return loss, grad.ravel()

        coef0 = np.zeros(n_classes * n_features, dtype=np.float64)

        coef, funcval, infodic = fmin_l_bfgs_b(_func,
                                               coef0,
                                               maxiter=self.max_iter)

        if self.verbose and infodic["warnflag"] != 0:
            print("NOT CONVERGED: ", infodic["task"])

        return coef.reshape(n_classes, n_features)

    def fit(self, X, y):
        if self.fit_intercept:
            X = add_dummy_feature(X)

        if hasattr(y, "toarray"):
            raise ValueError("scipy sparse matrices not supported for y")

        y = np.array(y)

        if len(y.shape) == 1:
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)

        if self.solver == "lbfgs":
            self.coef_ = self._solve_lbfgs(X, y)

        else:
            raise ValueError("Invalid solver.")

        return self

    def decision_function(self, X):
        if self.fit_intercept:
            X = add_dummy_feature(X)
        return safe_sparse_dot(X, self.coef_.T)

    def predict_proba(self, X):
        theta = self.decision_function(X)
        loss = self._get_loss()
        return loss.predict(theta)

    def predict(self, X):
        ret = np.argmax(self.decision_function(X))
        if hasattr(self, "label_encoder_"):
            ret = self.label_encoder_.inverse_transform(ret)
        return ret
