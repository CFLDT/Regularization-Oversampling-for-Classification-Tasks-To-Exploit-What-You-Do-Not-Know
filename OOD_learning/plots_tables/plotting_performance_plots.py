import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_performance_plots(name, fpr_dict, tpr_dict, roc_auc_df, precis_dict, recall_dict,
                           ap_df, positive_found_df, disc_cum_gain_df):
    base_path = Path(__file__).parent

    plot_name_roc_auc = "ROC_AUC_" + name + ".png"
    plot_name_ap = "Average_precision_" + name + ".png"
    plot_name_fraud_n_chos = "Percent_positive_" + name + ".png"
    plot_name_disc_cum_gain = "Disc_cum_gain_" + name + ".png"

    plt.close()
    ax = sns.boxplot(data=roc_auc_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right", fontsize=7)
    plt.xlabel('Methods')
    plt.ylabel('ROC_AUC')
    plt.title("ROC_AUC_" + name)
    plt.tight_layout()
    plt.savefig((base_path / "../../plots/plots performance/boxplots" / plot_name_roc_auc).resolve())
    plt.close()

    ax = sns.boxplot(data=ap_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right", fontsize=7)
    plt.xlabel('Methods')
    plt.ylabel('Average Precision')
    plt.title("Average_precision_" + name)
    plt.tight_layout()
    plt.savefig((base_path / "../../plots/plots performance/boxplots" / plot_name_ap).resolve())
    plt.close()

    ax = sns.boxplot(data=disc_cum_gain_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right", fontsize=7)
    plt.xlabel('Methods')
    plt.ylabel('Discounted Cumulative Gain')
    plt.title("Discounted_cumulative_gain_" + name)
    plt.tight_layout()
    plt.savefig((base_path / "../../plots/plots performance/boxplots" / plot_name_disc_cum_gain).resolve())
    plt.close()

    ax = sns.boxplot(data=positive_found_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right", fontsize=7)
    plt.xlabel('Methods')
    plt.ylabel('Percent positive')
    plt.title("Percent_positive_" + name)
    plt.tight_layout()
    plt.savefig((base_path / "../../plots/plots performance/boxplots" / plot_name_fraud_n_chos).resolve())
    plt.close()

    for i in fpr_dict.keys():

        plot_name_roc = "R0C_" + name + i + ".png"

        for j in fpr_dict[i].keys():
            fpr = fpr_dict[i][j]
            tpr = tpr_dict[i][j]
            plt.title("R0C_" + name + i)
            plt.plot(fpr, tpr, 'b')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

        plt.tight_layout()
        plt.savefig((base_path / "../../plots/plots performance/curves" / plot_name_roc).resolve())
        plt.close()

    for i in fpr_dict.keys():

        plot_name_pr = "PR_" + name + i + ".png"

        for j in fpr_dict[i].keys():
            pr = precis_dict[i][j]
            rec = recall_dict[i][j]
            # Baseline PR curve plot is a horizontal line with height equal to the number
            # of positive examples P over the total number of training data N.
            plt.title("PR_" + name + i)
            plt.plot(rec, pr, 'b')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Precision')
            plt.xlabel('Recall')

        plt.tight_layout()
        plt.savefig((base_path / "../../plots/plots performance/curves" / plot_name_pr).resolve())
        plt.close()
