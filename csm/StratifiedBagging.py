from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from sklearn.utils.multiclass import unique_labels
from .LinearClassifier import LinearClassifier


KUNCHEVA = 0.000001


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


class StratifiedBagging(BaseEstimator, ClassifierMixin):

    def __init__(self, base_clf=LinearClassifier(), ensemble_size=20, decision="mean", random_state=None):
        self.ensemble_size = ensemble_size
        self.base_clf = base_clf
        self.decision = decision
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.estimators_ = []

        np.random.seed(self.random_state)

        for i in range(self.ensemble_size):
            selected_samples = [np.random.randint(0, self.X_[self.y_==label].shape[0], self.X_[self.y_==label].shape[0]) for label in self.classes_]

            X_train = np.concatenate((self.X_[self.y_==0][selected_samples[0]],self.X_[self.y_==1][selected_samples[1]]), axis=0)
            y_train = np.concatenate((self.y_[self.y_==0][selected_samples[0]],self.y_[self.y_==1][selected_samples[1]]), axis=0)

            self.estimators_.append(clone(self.base_clf).fit(X_train, y_train))

        return self

    def decfunc(self, X):
        decfuncs = np.array([clf.decision_function(X)
                             for clf in self.estimators_])

        if self.decision == "n-mean":
            decfunc = np.array([m_pad(d) for d in decfuncs.T])
        elif self.decision == "mean":
            decfunc = np.mean(decfuncs, axis=0)
        elif self.decision == "mv":
            decfunc = np.array([clf.predict(X) for clf in self.estimators_])
        return decfunc

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        decfunc = self.decfunc(X)
        if self.decision == "mvote":
            predict = []
            for i, row in enumerate(decfunc.T):
                decision = np.bincount(row)
                predict.append(np.argmax(decision))
            y_pred = np.array(predict)
        else:
            y_pred = decfunc > 0
        return y_pred
