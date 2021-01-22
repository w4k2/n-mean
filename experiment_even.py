import numpy as np
import weles as ws
from strlearn.metrics import (
    balanced_accuracy_score,
)
from csm import StratifiedBoosting
from csm import LinearClassifier
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tabulate import tabulate


# # Set plot params
# rcParams["font.family"] = "monospace"
# colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0.9, 0, 0)]
# ls = ["-", "--", ":", "-"]
# lw = [1.5, 1.5, 1.5, 1.5]

metrics = {
    "BAC": balanced_accuracy_score,
    # "G-mean": geometric_mean_score_1,
    # "F1": f1_score,
    # "Precision": precision,
    # "Recall": recall,
    # "Specificity": specificity
}


scores_bac = []

random_state = 1410

for ensemble_size in range(2, 32, 2):
    print("%i CLASSIFIERS" % ensemble_size)
    clfs = {
        # "L": LinearClassifier(),
        "MV": StratifiedBoosting(ensemble_size=ensemble_size, decision="mv",
                                 random_state=random_state, k=ensemble_size),
        "M": StratifiedBoosting(ensemble_size=ensemble_size, decision="mean",
                                 random_state=random_state, k=ensemble_size),
        "NM": StratifiedBoosting(ensemble_size=ensemble_size, decision="n-mean",
                                 random_state=random_state, k=ensemble_size),
    }

    data = ws.utils.Data(selection=(
        "all", ["balanced", "binary"]), path="datasets/")
    datasets = data.load()

    eval = ws.evaluation.Evaluator(
        datasets=datasets,
        protocol=(5, 5, 1410),
        store="store_even/"
    )

    eval.process(clfs=clfs, verbose=True)

    scores = eval.score(metrics=metrics, verbose=True)
    stat = ws.evaluation.PairedTests(eval)
    # tables = stat.process("t_test_corrected", tablefmt="latex_booktabs")
    # [print(tables[a]) for a in tables]

    st_ranks = stat.global_ranks(name=str(ensemble_size))
    # np.save("scores/%s_%i" % (data_type, ensemble_size), scores)
    print(tabulate(st_ranks, tablefmt="latex_booktabs"))
    # print(tabulate(pool_size_stat, headers=(list(clfs.keys())), tablefmt="plain"))

    # print(eval.mean_ranks)


# plots
"""
    scores_bac.append(eval.mean_ranks[0])
scores_bac = np.array(scores_bac)

plt_data = [scores_bac]
metric_names = ["BAC"]

for p, data in enumerate(plt_data):
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.axes()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    x = np.arange(3, 55, 2)
    x_labels = [str(x[i])for i in range(len(x))]
    plt.xticks(x, x_labels)
    plt.xlim(3, 53)
    plt.ylim(0.0, 4.0)
    y = np.arange(0.0, 4.5, 0.5)
    plt.yticks(y)
    plt.title(metric_names[p])
    ax.set_xlabel('Classifier pool size')
    ax.set_ylabel('Mean rank')
    plt.grid(color=(.8,.8,.8), linestyle=':', linewidth=0.1)

    for j, (values, label) in enumerate(zip(data.T, clfs.keys())):
        plt.plot(x, data[:,j], label=label, c=colors[j], ls=ls[j], lw=lw[j])

    ax.legend(
        loc=8,
        bbox_to_anchor=(0.5, 0.05),
        fancybox=False,
        shadow=True,
        ncol=4,
        fontsize=7,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig("foo")
    plt.savefig("plots/%s" % metric_names[p])
    plt.savefig("plots/%s.eps" % metric_names[p])
    plt.close()
# """
