from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from .LinearClassifier import LinearClassifier
import random


def m_pad(A, MOD=10000000):
    n = len(A)
    x = np.sum(np.abs(np.diff(np.meshgrid(A, A), axis=0))) / n
    uA = np.concatenate(([np.min(A) - MOD], np.unique(A), [np.max(A) + MOD]))
    p = np.sum(np.abs(A[:, np.newaxis] - uA), axis=0)
    a = np.array([(p[j + 1] - p[j]) / (uA[j + 1] - uA[j])
                  for j in range(len(uA) - 1)])
    b = np.array([p[j] - a[j] * uA[j] for j in range(len(uA) - 1)])
    x_candidates = (x - b) / a
    sol = x_candidates[(x_candidates > uA[:-1]) * (x_candidates < uA[1:])]
    return np.mean(sol)


class StratifiedBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf=LinearClassifier(), ensemble_size=20, k=None,
                 decision="mean", random_state=None):
        self.ensemble_size = ensemble_size
        self.base_clf = base_clf
        self.decision = decision
        self.random_state = random_state
        if k is None:
            self.k = ensemble_size
        else:
            self.k = ensemble_size

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, self.prior_ = np.unique(y, return_counts=True)

        self.X_ = np.copy(X)
        self.y_ = np.copy(y)
        self.estimators_ = []

        np.random.seed(self.random_state)

        self.draws = None

        # Prepare probas
        p = np.ones(y.shape)
        idx = np.linspace(0, len(y)-1, len(y)).astype(int)

        # Boosting
        for i in range(self.k):
            # Get and normalize probabilities
            _p = [p[y == label] for label in self.classes_]
            _p = [_ / np.sum(_) for _ in _p]

            # Select samples
            ss = [np.random.choice(idx[y == label],
                                   (self.prior_[label]).astype(int),
                                   p=_p[label])
                  for label in self.classes_]
            ss = np.concatenate(ss)

            # Train model
            clf = clone(self.base_clf).fit(X[ss], y[ss])
            y_pred = clf.predict(X)

            # Get mistakes and add probability
            mistakes = y_pred != y
            corrects = y_pred == y
            p[mistakes] += i
            p[corrects] = 1

            # Store model
            self.estimators_.append(clf)

        return self

    def decfunc(self, X):
        decfuncs = np.array([clf.decision_function(X)
                             for clf in self.estimators_])
        if self.decision == "n-mean":
            decfunc = np.array([m_pad(d) for d in decfuncs.T])
        elif self.decision == "mean":
            decfunc = np.mean(decfuncs, axis=0)
        elif self.decision == "mv":
            self.draws = 0
            decfunc = np.array([clf.predict(X) for clf in self.estimators_])
        return decfunc

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        decfunc = self.decfunc(X)
        if self.decision == "mv":
            predict = []
            for i, row in enumerate(decfunc.T):
                _, count = np.unique(row, return_counts=True)
                if count.shape[0]==2 and count[0]==count[1]:
                    self.draws += 1
                    dec = random.randint(0, 1)
                    predict.append(dec)
                else:
                    decision = np.bincount(row)
                    predict.append(np.argmax(decision))
            y_pred = np.array(predict)
        else:
            y_pred = decfunc > 0
        return y_pred
