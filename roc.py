import numpy as np
import matplotlib.pyplot as plt
# from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# from sklearn import cross_validation
import argparse
import os


def draw_roc(K, cnv=True):
    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for k in range(K):  
        score = np.load(os.getcwd() + f'/results/clinic/score_{k}.npy')
        y_true = np.load(os.getcwd() + f'/results/clinic/y_true_{k}.npy')
        print()
        fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(score)[:, 1])
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.3f)' % (k, roc_auc))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = {:.3f})'.format(mean_auc, std_auc), lw=2,
            color='Red')  # alpha=.6
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tick_params(labelsize=15)
    plt.title('ROC curve')
    plt.show()
    fig.savefig(os.getcwd() + f'/results/clinic/tmbc2.svg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--cnv', type=bool, default=True)
    args = parser.parse_args()
    draw_roc(args.K, args.cnv)

    origirn_classfication_set = None
