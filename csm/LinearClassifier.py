import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LinearClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        p = np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])
        self.b = np.mean(p, axis=0)
        self.w = p[1] - self.b
        # innowacja
        self.foo = np.abs((p - self.b).dot(self.w))[0]
        return self

    def decision_function(self, X):
        df = (X - self.b).dot(self.w)
        z = self.foo * (2/(1 + np.exp(-df))-1)
        return df

    def predict(self, X):
        return self.decision_function(X) > 0
