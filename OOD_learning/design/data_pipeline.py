import numpy as np
from OOD_learning.design import DataHandler
from OOD_learning.design import WoeEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


class DataPipeline:

    def __init__(self):
        self.dummy_cols = None
        self.median = None
        self.woeencoder = None
        self.colnames = None

    def pipeline_fit_trans(self, X_train, X_test, y_train):
        """
        Dummy encoding for na values for numerical features
        """

        X_train_dummy = X_train.copy()
        X_test_dummy = X_test.copy()
        self.dummy_cols = X_train_dummy.select_dtypes(include=np.number).columns.tolist()
        X_train_dummy, X_test_dummy = \
            DataHandler.create_dummies(X_train_dummy,
                                       X_test_dummy, self.dummy_cols)

        self.num_col_names = self.dummy_cols
        all_col = X_train_dummy.columns.tolist()
        self.cat_col_names = list(set(all_col) ^ set(self.num_col_names))

        """
        Inpute the median for continuous features
        """

        X_train_median = X_train_dummy.copy()
        X_test_median = X_test_dummy.copy()
        X_train_median, X_test_median, self.median = DataHandler. \
            impute_median(X_train_median, X_test_median,
                          self.dummy_cols)
        """
        WOE encoding:
        Clean data by using IV value
        Keep WOE encoding categorical features
        """

        X_train_woe = X_train_median.copy()
        X_test_woe = X_test_median.copy()
        self.woeencoder = WoeEncoder(fine_classer=20)
        self.woeencoder.fit(X_train_woe, y_train)
        X_train_woe = self.woeencoder.transform(X_train_woe)
        X_test_woe = self.woeencoder.transform(X_test_woe)
        info_value = self.woeencoder.iv_df

        # f, ax = plt.subplots(figsize=(8, 6))
        # base_path = Path(__file__).parent
        # plot_path = (base_path / "../../plots/plots feature importance" / "iv.png").resolve()
        # sns.barplot(x=info_value['Variable_Name'], y=info_value['Information_Value'], ax=ax)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
        # plt.title('Feature Importance')
        # plt.ylabel('Importance')
        # plt.tight_layout()
        # plt.savefig(plot_path)
        # plt.close()

        """
        Feature extraction
        """

        X_train_ext = X_train_woe.copy()
        X_test_ext = X_test_woe.copy()

        # iv > 0.1 , max 30 features
        info_value_feature_extracted = info_value[info_value['Information_Value'].gt(0.1)]
        info_value_feature_extracted = info_value_feature_extracted.nlargest(30, 'Information_Value')
        self.colnames = info_value_feature_extracted['Variable_Name'].tolist()
        X_train_ext = X_train_ext[X_train_ext.columns[X_train_ext.columns.isin(self.colnames)]]
        X_test_ext = X_test_ext[X_test_ext.columns[X_test_ext.columns.isin(self.colnames)]]

        # X_train_ext = X_train_ext[self.colnames]
        # X_test_ext = X_test_ext[self.colnames]

        self.num_col_names = list(set(self.colnames).intersection(set(self.num_col_names)))
        self.cat_col_names = list(set(self.colnames).intersection(set(self.cat_col_names)))

        X_train_ext = X_train_ext.reindex(columns=self.colnames)
        X_test_ext = X_test_ext.reindex(columns=self.colnames)

        return X_train_ext, X_test_ext

    def pipeline_trans(self, X):
        """
        Dummy encoding for na values for numerical features
        """

        X_dummy = X.copy()

        X_dummy, X_test_dummy = \
            DataHandler.create_dummies(X_dummy, None, self.dummy_cols)

        """
        Inpute the median for continuous features
        """

        X_median = X_dummy.copy()
        X_median, X_test_median = DataHandler. \
            impute_values(X_median, None,
                          self.dummy_cols, self.median)

        """
        WOE encoding for categorical features
        """

        X_woe = X_median.copy()
        X_woe = self.woeencoder.transform(X_woe)

        # info_value = self.woeencoder.iv_df
        # base_path = Path(__file__).parent
        # plot_path = (base_path / "../../plots/plots feature importance" / "iv.png").resolve()
        # ax = sns.barplot(x=info_value['Variable_Name'], y=info_value['Information_Value'])
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
        # plt.title('Feature Importance')
        # plt.ylabel('Importance')
        # plt.tight_layout()
        # plt.savefig(plot_path)

        """
        Features 
        """

        X_ext = X_woe.copy()

        X_ext = X_ext[X_ext.columns[X_ext.columns.isin(self.colnames)]]
        # X_ext = X_ext[self.colnames]

        X_ext = X_ext.reindex(columns=self.colnames)

        return X_ext
