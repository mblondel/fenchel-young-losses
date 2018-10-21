# Author: Mathieu Blondel
# License: Simplified BSD

import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(currentdir))

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.testing import assert_array_almost_equal

from fyl_sklearn import FYClassifier

X, y = make_classification(n_samples=10, n_features=5, n_informative=3,
                           n_classes=3, random_state=0)
Y = LabelBinarizer().fit_transform(y)


def test_logistic_against_sklearn():
    for y_ in (y, Y):
        clf = FYClassifier(loss="logistic", fit_intercept=False, alpha=0.1)
        clf.fit(X, y_)
        clf2 = LogisticRegression(fit_intercept=False, multi_class="multinomial",
                                  solver="lbfgs", C=1./(clf.alpha * X.shape[0]))
        clf2.fit(X, y)
        assert_array_almost_equal(clf.coef_, clf2.coef_, 4)


def test_logistic_ova_against_sklearn():
    for y_ in (y, Y):
        clf = FYClassifier(loss="logistic-ova", fit_intercept=False, alpha=0.1)
        clf.fit(X, y_)
        clf2 = LogisticRegression(fit_intercept=False, multi_class="ovr",
                                  solver="lbfgs", C=1./(clf.alpha * X.shape[0]))
        clf2.fit(X, y)
        assert_array_almost_equal(clf.coef_, clf2.coef_, 4)


def test_other_losses():
    for loss in ("sparsemax", "tsallis"):
        for fit_intercept in (True, False):
            for y_ in (y, Y):
                clf = FYClassifier(loss=loss, fit_intercept=fit_intercept,
                                   alpha=0.1)
                clf.fit(X, y_)
