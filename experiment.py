import numpy as np
import weles as ws
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity,
)
from csm import StratifiedBagging
from csm import LinearClassifier

data_types = ["balanced", "imbalanced"]

metrics = {
    "BAC": balanced_accuracy_score,
    "G-mean": geometric_mean_score_1,
    "F1": f1_score,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity
}

for data_type in data_types:
    for ensemble_size in range(3,49,2):
        clfs = {
            "L": LinearClassifier(),
            "Mean": StratifiedBagging(ensemble_size=ensemble_size, decision="mean", random_state=1410),
            "N-mean": StratifiedBagging(ensemble_size=ensemble_size, decision="n-mean", random_state=1410),
            "MV": StratifiedBagging(ensemble_size=ensemble_size, decision="mv", random_state=1410)
        }

        data = ws.utils.Data(selection=(
            "all", [data_type, "binary"]), path="datasets/")
        datasets = data.load()

        eval = ws.evaluation.Evaluator(
            datasets=datasets,
            protocol=(5, 5, 1410),
            store="store/"
        )

        eval.process(clfs=clfs, verbose=True)

        # scores = eval.score(metrics=metrics, verbose=False)
        # np.save("scores/%s_%i" % (data_type, ensemble_size), scores)
