from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from sklearn.utils.multiclass import unique_labels
from .LinearClassifier import LinearClassifier
from weles.metrics import balanced_accuracy_score

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


class StratifiedBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf=LinearClassifier(), ensemble_size=20, k=100,
                 decision="mean", random_state=None, tsp=.9):
        self.ensemble_size = ensemble_size
        self.base_clf = base_clf
        self.decision = decision
        self.random_state = random_state
        self.tsp = tsp
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, self.prior_ = np.unique(y, return_counts=True)

        # print(self.classes_, self.prior_)

        self.X_ = np.copy(X)
        self.y_ = np.copy(y)
        self.estimators_ = []

        np.random.seed(self.random_state)

        # Prepare probas
        p = np.ones(y.shape)
        idx = np.linspace(0, len(y)-1, len(y)).astype(int)
        self.weights = []

        for i in range(self.k):
            # Get and normalize probabilities
            _p = [p[y == label] for label in self.classes_]
            _p = [_ / np.sum(_) for _ in _p]

            # Select samples
            ss = [np.random.choice(idx[y == label],
                                   (self.prior_[label]*self.tsp).astype(int),
                                   p=_p[label])
                  for label in self.classes_]
            ss = np.concatenate(ss)

            # Train model
            clf = clone(self.base_clf).fit(X[ss], y[ss])
            y_pred = clf.predict(X)
            score = balanced_accuracy_score(y, y_pred)
            self.weights.append(score)

            # Get mistakes and add probability
            mistakes = y_pred != y
            # print("%.3f, %3i MISTAKES | %3i" % (score, np.sum(mistakes), i))
            p[mistakes] += i

            # Store model
            self.estimators_.append(clf)

        self.weights = np.array(self.weights)
        # print(self.weights)

        ars = np.argsort(-self.weights)
        # print(ars)

        self.weights = self.weights[ars[:self.ensemble_size]]
        # print(self.weights)

        self.estimators_ = [self.estimators_[i]
                            for i in ars[:self.ensemble_size]]

        # print(self.estimators_)
        # print(self.weights)

        #print(mask)
        #print(self.weights)

        # print(self.estimators_)
        # exit()

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
