from pathlib import Path
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import combinations


def critical_difference_plotter(metrics, alpha, name, models):
    """
    Code is modified version of:

    Title: How can plot Results of the Friedman-Nemenyi test using python
    Author: ImportanceOfBeingErnest
    Date: 2017
    Availability: https://stackoverflow.com/questions/43383144/how-can-plot-results-of-the-friedman-nemenyi-test-using-python
    """

    def draw_diag(cd, dict_cd, plot_path, name, number):

        limits = (number, 1)
        fig, ax = plt.subplots(figsize=(6+number/4, 1.8 * number / 4))
        plt.subplots_adjust(left=0.2, right=0.8)

        # set up plot
        ax.set_xlim(limits)
        ax.set_ylim(0, 1)
        ax.spines['top'].set_position(('axes', 0.6))
        # ax.xaxis.tick_top()
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_visible(False)
        for pos in ["bottom", "left", "right"]:
            ax.spines[pos].set_visible(False)

        s=0
        #if number != 4:
        #    s = 0.1

        # CD bar
        ax.plot([limits[0], limits[0] - cd], [.9, .9], color="k")
        ax.plot([limits[0]-s, limits[0]-s], [.9 - 0.03, .9 + 0.03], color="k")
        ax.plot([limits[0] - cd, limits[0] - cd], [.9 - 0.03, .9 + 0.03], color="k")
        ax.text(limits[0] - cd / 2., 0.92, "CD", ha="center", va="bottom")

        # annotations
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, va="center")

        b = np.array(list(dict_cd.values()))
        list_sorted = sorted(dict_cd.items(), key=lambda x: x[1])

        if number == 3:
            ax.annotate(list_sorted[2][0], xy=(list_sorted[2][1], 0.6), xytext=(-0.1, 0), ha="right", **kw)
            ax.annotate(list_sorted[1][0], xy=(list_sorted[1][1], 0.6), xytext=(1.1, 0), ha="left", **kw)
            ax.annotate(list_sorted[0][0], xy=(list_sorted[0][1], 0.6), xytext=(1.1, 0.25), ha="left", **kw)

        if number == 4:
            ax.annotate(list_sorted[3][0], xy=(list_sorted[3][1], 0.6), xytext=(-0.1, 0.25), ha="right", **kw)
            ax.annotate(list_sorted[2][0], xy=(list_sorted[2][1], 0.6), xytext=(-0.1, 0), ha="right", **kw)
            ax.annotate(list_sorted[1][0], xy=(list_sorted[1][1], 0.6), xytext=(1.1, 0), ha="left", **kw)
            ax.annotate(list_sorted[0][0], xy=(list_sorted[0][1], 0.6), xytext=(1.1, 0.25), ha="left", **kw)

        if number == 5:
            ax.annotate(list_sorted[4][0], xy=(list_sorted[4][1], 0.6), xytext=(-0.1, 0), ha="right", **kw)
            ax.annotate(list_sorted[3][0], xy=(list_sorted[3][1], 0.6), xytext=(-0.1, -0.25), ha="right", **kw)
            ax.annotate(list_sorted[2][0], xy=(list_sorted[2][1], 0.6), xytext=(1.1, -0.25), ha="left", **kw)
            ax.annotate(list_sorted[1][0], xy=(list_sorted[1][1], 0.6), xytext=(1.1, 0), ha="left", **kw)
            ax.annotate(list_sorted[0][0], xy=(list_sorted[0][1], 0.6), xytext=(1.1, 0.25), ha="left", **kw)

        if number == 6:
            ax.annotate(list_sorted[5][0], xy=(list_sorted[5][1], 0.6), xytext=(-0.1, 0.25), ha="right", **kw)
            ax.annotate(list_sorted[4][0], xy=(list_sorted[4][1], 0.6), xytext=(-0.1, 0), ha="right", **kw)
            ax.annotate(list_sorted[3][0], xy=(list_sorted[3][1], 0.6), xytext=(-0.1, -0.25), ha="right", **kw)
            ax.annotate(list_sorted[2][0], xy=(list_sorted[2][1], 0.6), xytext=(1.1, -0.25), ha="left", **kw)
            ax.annotate(list_sorted[1][0], xy=(list_sorted[1][1], 0.6), xytext=(1.1, 0), ha="left", **kw)
            ax.annotate(list_sorted[0][0], xy=(list_sorted[0][1], 0.6), xytext=(1.1, 0.25), ha="left", **kw)

        if number == 7:
            #ax.annotate(list_sorted[7][0], xy=(list_sorted[7][1], 0.6), xytext=(-0.1, 0.25), ha="right", **kw)
            ax.annotate(list_sorted[6][0], xy=(list_sorted[6][1], 0.6), xytext=(-0.1, 0), ha="right", **kw)
            ax.annotate(list_sorted[5][0], xy=(list_sorted[5][1], 0.6), xytext=(-0.1, -0.25), ha="right", **kw)
            ax.annotate(list_sorted[4][0], xy=(list_sorted[4][1], 0.6), xytext=(-0.1, -0.5), ha="right", **kw)
            ax.annotate(list_sorted[3][0], xy=(list_sorted[3][1], 0.6), xytext=(1.1, -0.5), ha="left", **kw)
            ax.annotate(list_sorted[2][0], xy=(list_sorted[2][1], 0.6), xytext=(1.1, -0.25), ha="left", **kw)
            ax.annotate(list_sorted[1][0], xy=(list_sorted[1][1], 0.6), xytext=(1.1, 0), ha="left", **kw)
            ax.annotate(list_sorted[0][0], xy=(list_sorted[0][1], 0.6), xytext=(1.1, 0.25), ha="left", **kw)

        if number == 8:
            ax.annotate(list_sorted[7][0], xy=(list_sorted[7][1], 0.6), xytext=(-0.1, 0.25), ha="right", **kw)
            ax.annotate(list_sorted[6][0], xy=(list_sorted[6][1], 0.6), xytext=(-0.1, 0), ha="right", **kw)
            ax.annotate(list_sorted[5][0], xy=(list_sorted[5][1], 0.6), xytext=(-0.1, -0.25), ha="right", **kw)
            ax.annotate(list_sorted[4][0], xy=(list_sorted[4][1], 0.6), xytext=(-0.1, -0.5), ha="right", **kw)
            ax.annotate(list_sorted[3][0], xy=(list_sorted[3][1], 0.6), xytext=(1.1, -0.5), ha="left", **kw)
            ax.annotate(list_sorted[2][0], xy=(list_sorted[2][1], 0.6), xytext=(1.1, -0.25), ha="left", **kw)
            ax.annotate(list_sorted[1][0], xy=(list_sorted[1][1], 0.6), xytext=(1.1, 0), ha="left", **kw)
            ax.annotate(list_sorted[0][0], xy=(list_sorted[0][1], 0.6), xytext=(1.1, 0.25), ha="left", **kw)

        if number == 9:
            ax.annotate(list_sorted[8][0], xy=(list_sorted[8][1], 0.6), xytext=(-0.1, 0.25), ha="right", **kw)
            ax.annotate(list_sorted[7][0], xy=(list_sorted[7][1], 0.6), xytext=(-0.1, 0), ha="right", **kw)
            ax.annotate(list_sorted[6][0], xy=(list_sorted[6][1], 0.6), xytext=(-0.1, -0.25), ha="right", **kw)
            ax.annotate(list_sorted[5][0], xy=(list_sorted[5][1], 0.6), xytext=(-0.1, -0.5), ha="right", **kw)
            ax.annotate(list_sorted[4][0], xy=(list_sorted[4][1], 0.6), xytext=(-0.1, -0.75), ha="right", **kw)
            ax.annotate(list_sorted[3][0], xy=(list_sorted[3][1], 0.6), xytext=(1.1, -0.5), ha="left", **kw)
            ax.annotate(list_sorted[2][0], xy=(list_sorted[2][1], 0.6), xytext=(1.1, -0.25), ha="left", **kw)
            ax.annotate(list_sorted[1][0], xy=(list_sorted[1][1], 0.6), xytext=(1.1, 0), ha="left", **kw)
            ax.annotate(list_sorted[0][0], xy=(list_sorted[0][1], 0.6), xytext=(1.1, 0.25), ha="left", **kw)

        # bars
        constant = 0.07
        c = 0
        list_sor = np.sort(b)

        diff = list(combinations(list_sor, 2))
        list_diff = []
        list_di = list(np.array((pd.DataFrame(diff).diff(axis=1).dropna(axis=1))))
        for i in range(len(list_di)):
            list_diff.append(float(list_di[i]))

        diff_ind = list(combinations(range(0, number), 2))
        df_all = np.array(pd.DataFrame(diff_ind))
        list_all = []
        for i in range(np.shape(df_all)[0]):
            list_all.append(list(range(df_all[i, 0], df_all[i, 1] + 1)))
        new_dict = dict(zip(list_diff, list_all))

        dict_1 = {k: new_dict[k] for k in new_dict if k < cd}
        dict_2 = dict(
            [i for i in dict_1.items() if not any(set(j).issuperset(set(i[1])) and j != i[1] for j in dict_1.values())])
        list_sorted = sorted(dict_2.items(), key=lambda x: len(x[1]))
        for value in list_sorted:
            list_number = value[1]
            firstn = list_number[0]
            lastn = list_number[-1]

            ax.plot([list_sor[firstn], list_sor[lastn]], [0.55 - c, 0.55 - c], color="k", lw=3)
            c = c + constant

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    for metric in metrics:

        base_path = Path(__file__).parent
        plot_name = "critical_difference_" + metric + '_' + name + ".png"
        plot_path = (base_path / "../../plots/plots critical difference" / plot_name).resolve()

        table_name = 'all' + '_' + metric + ".xlsx"
        tables_path = (base_path / "../../tables/tables performance" / table_name).resolve()
        tab = pd.read_excel(tables_path, index_col=0)
        tab = tab.rename(columns={'Logit(1, 1, 0, 0)': 'Logit','XGBoost(1, 1, 0, 0)': 'XGBoost',
                                  'XGBoost(1, 1, 1, 1)_BROOD_ROT': 'XGB_BR(MIN_ROT)','XGBoost(1, 1, 1, 1)_BROOD_KNN': 'XGB_BR(MIN_KNN)',
                                  'XGBoost(1, 1, 0, 0)_rose': 'XGB_ROSE','XGBoost(1, 1, 1, 1)_BROOD_ROT_rose': 'XGB_ROSE_BR(MIN_ROT)',
                                  'XGBoost(1, 1, 0, 0)_smote': 'XGB_SMOTE','XGBoost(1, 1, 0, 0)_adasyn': 'XGB_ADASYN',
                                  'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose': 'XGB_ROSE_BR(MIN_KNN)'})
        tab = tab[tab.columns.drop(list(tab.filter(regex='Logit\(1, 1, 1, 1\)|Logit\(1, 1, 0, 0\)_|XGBoost\(1, 1, 0, 0\)_ABROOD')))]
        #tab = tab[tab.columns.drop(list(tab.filter(regex='ADASYN|ROSE|SMOTE')))]
        tab = tab.rank(ascending=False,axis=1)

        mean = tab.mean(axis=0)
        dict_cd = mean.to_dict()


        CRITICAL_VALUES = [
            # p   0.01   0.05   0.10  Models
            [2.576, 1.960, 1.645],  # 2
            [2.913, 2.344, 2.052],  # 3
            [3.113, 2.569, 2.291],  # 4
            [3.255, 2.728, 2.460],  # 5
            [3.364, 2.850, 2.589],  # 6
            [3.452, 2.948, 2.693],  # 7
            [3.526, 3.031, 2.780],  # 8
            [3.590, 3.102, 2.855],  # 9
            [3.646, 3.164, 2.920],  # 10
            [3.696, 3.219, 2.978],  # 11
            [3.741, 3.268, 3.030],  # 12
            [3.781, 3.313, 3.077],  # 13
            [3.818, 3.354, 3.120],  # 14
            [3.853, 3.391, 3.159],  # 15
            [3.884, 3.426, 3.196],  # 16
            [3.914, 3.458, 3.230],  # 17
            [3.941, 3.489, 3.261],  # 18
            [3.967, 3.517, 3.291],  # 19
            [3.992, 3.544, 3.319],  # 20
            [4.015, 3.569, 3.346],  # 21
            [4.037, 3.593, 3.371],  # 22
            [4.057, 3.616, 3.394],  # 23
            [4.077, 3.637, 3.417],  # 24
            [4.096, 3.658, 3.439],  # 25
            [4.114, 3.678, 3.459],  # 26
            [4.132, 3.696, 3.479],  # 27
            [4.148, 3.714, 3.498],  # 28
            [4.164, 3.732, 3.516],  # 29
            [4.179, 3.749, 3.533],  # 30
            [4.194, 3.765, 3.550],  # 31
            [4.208, 3.780, 3.567],  # 32
            [4.222, 3.795, 3.582],  # 33
            [4.236, 3.810, 3.597],  # 34
            [4.249, 3.824, 3.612],  # 35
            [4.261, 3.837, 3.626],  # 36
            [4.273, 3.850, 3.640],  # 37
            [4.285, 3.863, 3.653],  # 38
            [4.296, 3.876, 3.666],  # 39
            [4.307, 3.888, 3.679],  # 40
            [4.318, 3.899, 3.691],  # 41
            [4.329, 3.911, 3.703],  # 42
            [4.339, 3.922, 3.714],  # 43
            [4.349, 3.933, 3.726],  # 44
            [4.359, 3.943, 3.737],  # 45
            [4.368, 3.954, 3.747],  # 46
            [4.378, 3.964, 3.758],  # 47
            [4.387, 3.973, 3.768],  # 48
            [4.395, 3.983, 3.778],  # 49
            [4.404, 3.992, 3.788],  # 50
        ]

        def critical_value(pvalue, models):
            """
            Returns the critical value for the two-tailed Nemenyi test for a given
            p-value and number of models being compared.
            """
            if pvalue == 0.01:
                col_idx = 0
            elif pvalue == 0.05:
                col_idx = 1
            elif pvalue == 0.10:
                col_idx = 2
            else:
                raise ValueError('p-value must be one of 0.01, 0.05, or 0.10')

            if not (2 <= models and models <= 50):
                raise ValueError('number of models must be in range [2, 50]')
            else:
                row_idx = models - 2

            return CRITICAL_VALUES[row_idx][col_idx]

        #models = np.shape(mean)[0]
        n = np.shape(tab)[0]
        cv = critical_value(alpha, models)
        cd = cv * math.sqrt((models * (models + 1)) / (6.0 * n))

        #groupby alphabetical order

        # if models == 4:
        #     dict_cd = {'Logit': float(mean[0]), 'Logit_Brood': float(mean[1]),
        #                'XGBoost': float(mean[2]), 'XGBoost_Brood': float(mean[3])}
        #     draw_diag(cd, dict_cd, plot_path=plot_path, name=metric, number=models)
        #
        # if models == 6:
        #     dict_cd = {'Logit': float(mean[0]), 'Logit_Brood': float(mean[1]),
        #                'Logit_Rose': float(mean[2]), 'XGBoost': float(mean[3]),
        #                'XGBoost_Brood': float(mean[4]), 'XGBoost_Rose': float(mean[5]),
        #                }
        #     draw_diag(cd, dict_cd, plot_path=plot_path, name=metric, number=models)
        #
        # if models == 8:
        #     dict_cd = {'Logit': float(mean[0]), 'Logit_Brood': float(mean[1]),
        #                'Logit_Rose': float(mean[2]), 'Logit_Rose_Brood': float(mean[3]),
        #                'XGBoost': float(mean[4]), 'XGBoost_Brood': float(mean[5]),
        #                'XGBoost_Rose': float(mean[6]), 'XGBoost_Rose_Brood': float(mean[7])
        #                }

        draw_diag(cd, dict_cd, plot_path=plot_path, name=metric, number=models)
