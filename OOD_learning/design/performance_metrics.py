import numpy as np
import sklearn.metrics as metrics
import math


class PerformanceMetrics:

    @staticmethod
    def investigated_n(preds, y):

        n = np.count_nonzero(y)
        maxs = preds.argsort()[::-1][:n]
        true_values = y[maxs]

        # discounted cumulative gain
        discounter = np.zeros(n, dtype=object)
        for i in range(n):
            discounter[i] = 1 / (math.log2(i + 2))  # index starts at zero, thus +2

        numerator = (2 ** true_values) - 1

        discounted_cum_gain = np.sum(np.multiply(numerator, discounter))
        positive_found = np.sum(true_values) / n

        return discounted_cum_gain, positive_found

    @staticmethod
    def roc_auc(preds, y):
        fpr, tpr, threshold = metrics.roc_curve(y, preds)
        roc_auc = metrics.auc(fpr, tpr)

        return fpr, tpr, roc_auc

    @staticmethod
    def pr_auc(preds, y):
        pr, rec, threshold = metrics.precision_recall_curve(y, preds)
        ap = metrics.average_precision_score(y, preds)

        return pr, rec, ap
