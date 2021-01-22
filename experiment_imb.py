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
from csm import StratifiedBoosting
from csm import LinearClassifier
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tabulate import tabulate
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score

def rrse(y_true, y_pred):
    return -np.sqrt(np.sum(np.square(y_true-y_pred))/
                   np.sum(np.square(y_true-np.mean(y_true))))

def new_wonderful_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0

# Set plot params
rcParams["font.family"] = "monospace"
colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0.9, 0, 0)]
ls = ["-", "--", ":", "-"]
lw = [1.5, 1.5, 1.5, 1.5]

random_state = 1410

metrics = {
    "BAC": balanced_accuracy_score,
    # "G-mean": geometric_mean_score_1,
    # "F1": f1_score,
    # "Precision": precision,
    # "Recall": recall,
    # "Specificity": specificity,
    "AUC": new_wonderful_auc,
    "RRSE": rrse,
}


scores_bac = []
scores_gmean = []
scores_f1 = []
scores_precision = []
scores_recall = []
scores_specificity = []

for ensemble_size in range(3, 55, 2):
    print("%i CLASSIFIERS" % ensemble_size)

    oclfs = {
        "OL": ws.classifiers.MetaPreproc(base_estimator=LinearClassifier(), preprocessing=RandomOverSampler(random_state=random_state)),
        "OMV": ws.classifiers.MetaPreproc(base_estimator=StratifiedBoosting(ensemble_size=ensemble_size, decision="mv", random_state=random_state, k=ensemble_size), preprocessing=RandomOverSampler(random_state=random_state)),
        "OM": ws.classifiers.MetaPreproc(base_estimator=StratifiedBoosting(ensemble_size=ensemble_size, decision="mean", random_state=random_state, k=ensemble_size), preprocessing=RandomOverSampler(random_state=random_state)),
        "ONM": ws.classifiers.MetaPreproc(base_estimator=StratifiedBoosting(ensemble_size=ensemble_size, decision="n-mean", random_state=random_state, k=ensemble_size), preprocessing=RandomOverSampler(random_state=random_state)),
    }

    data = ws.utils.Data(selection=(
        "all", ["imbalanced", "binary"]), path="datasets/")
    datasets = data.load()

    # check datasets
    """
    for key in datasets.keys():
        print("%s %i %i %.1f" % (key, datasets[key][0].shape[0], datasets[key][0].shape[1], len(datasets[key][1][datasets[key][1]==0])/len(datasets[key][1][datasets[key][1]==1])))
    """
    eval = ws.evaluation.Evaluator(
        datasets=datasets,
        protocol=(5, 5, 1410),
        store="store_imb/"
    )

    eval.process(clfs=oclfs, verbose=True)

    scores = eval.score(metrics=metrics, verbose=False)
    stat = ws.evaluation.PairedTests(eval)
    # tables = stat.process("t_test_corrected", tablefmt="latex_booktabs")
    # [print(tables[a]) for a in tables]

    st_ranks = stat.global_ranks(name=str(ensemble_size))
    # np.save("scores/%s_%i" % (data_type, ensemble_size), scores)
    print(tabulate(st_ranks, tablefmt="latex_booktabs"))

    # print(eval.mean_ranks)


# plots
"""
    # scores_bac.append(eval.mean_ranks[0])
    # scores_gmean.append(eval.mean_ranks[1])
    # scores_f1.append(eval.mean_ranks[2])
    # scores_precision.append(eval.mean_ranks[3])
    # scores_recall.append(eval.mean_ranks[4])
    # scores_specificity.append(eval.mean_ranks[5])

scores_bac = np.array(scores_bac)
scores_gmean = np.array(scores_gmean)
scores_f1 = np.array(scores_f1)
scores_precision = np.array(scores_precision)
scores_recall = np.array(scores_recall)
scores_specificity = np.array(scores_specificity)

plt_data = [scores_bac, scores_gmean, scores_f1,  scores_precision, scores_recall, scores_specificity]
metric_names = ["BAC", "G-mean", "F1", "Precision", "Recall", "Specificity"]

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

    for j, (values, label) in enumerate(zip(data.T, oclfs.keys())):
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
    plt.savefig("plots/imb/%s_imb" % metric_names[p])
    plt.savefig("plots/imb/%s_imb.eps" % metric_names[p])
    plt.close()
# """
