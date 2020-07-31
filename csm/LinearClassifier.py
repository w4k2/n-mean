import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LinearClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        p = np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])
        self.b = np.mean(p, axis=0)
        self.w = p[1] - self.b
        # innowacja
        self.foo = (p - self.b).dot(self.w)
        return self

    # def decision_function(self, X):
    #     return (X - self.b).dot(self.w)
    def decision_function(self, X):
        a = (X - self.b).dot(self.w)
        a = a / np.abs(self.foo[1])
        b = np.abs(np.abs(a) - 1)
        b[b > 1] = 1
        b = 1 - b
        b[a < 0] = -b[a < 0]
        return b

    def predict(self, X):
        return self.decision_function(X) > 0
