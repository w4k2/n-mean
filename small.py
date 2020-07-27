import weles as ws
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
)
from csm import StratifiedBoosting
from csm import LinearClassifier

metrics = {
    "BAC": balanced_accuracy_score,
    "G-mean": geometric_mean_score_1,
    "F1": f1_score,
}
ensemble_size = 20
k = 20
clfs = {
    "BC": LinearClassifier(),
    "-M": StratifiedBoosting(ensemble_size=ensemble_size, decision="mean",
                             random_state=1410, k=k),
    "NM": StratifiedBoosting(ensemble_size=ensemble_size, decision="n-mean",
                             random_state=1410, k=k),
    "MV": StratifiedBoosting(ensemble_size=ensemble_size, decision="mv",
                             random_state=1410, k=k)
}

data = ws.utils.Data(selection=("all", ["balanced", "binary"]),
                     path="datasets/")
datasets = data.load()
eval = ws.evaluation.Evaluator(
    datasets=datasets,
    protocol=(5, 5, 1410),
    store="/Users/xehivs/store/"
)
eval.process(clfs=clfs, verbose=True)
scores = eval.score(metrics=metrics, verbose=True)


t = ws.evaluation.PairedTests(eval).process(
    't_test_corrected', corr=.2, tablefmt="plain", std_fmt="(%.2f)"
)

[print(t[a]) for a in t]
