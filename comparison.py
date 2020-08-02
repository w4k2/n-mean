import weles as ws
from strlearn.metrics import balanced_accuracy_score
from csm import StratifiedBoosting
from csm import LinearClassifier

metrics = {
    "BAC": balanced_accuracy_score
}
ensemble_size = 15
rs = 42
clfs = {
    "BA": LinearClassifier(),
    "MV": StratifiedBoosting(ensemble_size=ensemble_size, decision="mv",
                             random_state=rs),
    "AS": StratifiedBoosting(ensemble_size=ensemble_size, decision="mean",
                             random_state=rs),
    "NM": StratifiedBoosting(ensemble_size=ensemble_size, decision="n-mean",
                             random_state=rs)
}

"""
c = 12
datasets = {}
for i in range(c):
    datasets.update({'dataset%i' % i:
                     make_classification(n_samples=1000,
                                         n_features=10,
                                         n_informative=5,
                                         n_classes=2,
                                         n_redundant=0,
                                         random_state=193 + i, flip_y=.1)
                     }
                    )
"""

data = ws.utils.Data(selection=(
    "all", ["balanced", "binary"]), path="datasets/")
datasets = data.load()

eval = ws.evaluation.Evaluator(
    datasets=datasets,
    protocol=(1, 3, 1410)
)
eval.process(clfs=clfs, verbose=True)
scores = eval.score(metrics=metrics, verbose=True)


t = ws.evaluation.PairedTests(eval).process(
    't_test_rel', tablefmt="plain", std_fmt="(%.2f)"
)

[print(t[a]) for a in t]
