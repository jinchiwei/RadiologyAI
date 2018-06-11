# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib
matplotlib.use('Agg') # Added for display errors
from matplotlib import pyplot as plt
import numpy as np

from sklearn import metrics


def roc_auc_metrics(y_true, y_score, n_classes, weightfile):
    y_true = np.concatenate(y_true, 0)
    y_true2 = np.zeros((y_true.shape[0], 2))
    for column in range(y_true2.shape[1]):
        y_true2[:, column] = (y_true == column)
    y_true = y_true2

    y_score = np.concatenate(y_score, 0)

    # print(y_true)
    # print(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('roc' + str(weightfile) + '.png')

    auc_score = metrics.roc_auc_score(y_true[:, 1], y_score[:, 1])
    print('auc_score: ', auc_score)