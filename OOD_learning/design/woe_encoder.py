import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


class WoeEncoder:
    """
    Code is based on the package

    Title: XuniVerse
    Author: Sundar Krishnan
    Date: 2020
    Availability: https://github.com/Sundar0989/XuniVerse
    """

    def __init__(self, woe_bins=None, woe_df=None, fine_classer=None, coarse_classer=None):

        self.woe_bins = woe_bins
        self.woe_df = woe_df
        self.fine_classer = fine_classer
        self.coarse_classer = coarse_classer

    def fit(self, X, y):

        X = X.fillna('NA')  # treat NaN's as own class
        X.apply(lambda x: self.fitting(x, y), axis=0)

        self.woenew = 0  # log(1/1)

        self.iv_df = pd.DataFrame({'Information_Value': self.woe_df.groupby('Variable_Name').Information_Value.max()})
        self.iv_df = self.iv_df.reset_index()
        self.iv_df = self.iv_df.sort_values('Information_Value', ascending=False)

    def fitting(self, X, y):

        woe_mapping = {}
        temp_woe = pd.DataFrame({}, index=[])

        if (is_numeric_dtype(X) == True) and (self.fine_classer is not None):
            colname = X.name
            X_binned = pd.qcut(X, q=self.fine_classer, duplicates='drop')

            if X_binned.nunique() == 1:
                bins = pd.IntervalIndex.from_tuples([(X.min()-0.01, X.min()), (X.min(), X.max()+0.01)])
                X_binned = pd.cut(X, bins=bins)

            X = pd.Series(X_binned, name=colname)

        temp_df = pd.DataFrame({'X': X, "Y": y})
        grouped_df = temp_df.groupby('X', as_index=True)

        # calculate stats for variable and store it in temp_woe
        target_sum = grouped_df.Y.sum()
        temp_woe['Count'] = grouped_df.Y.count()
        temp_woe['Category'] = target_sum.index
        temp_woe['Event'] = target_sum
        temp_woe['Non_Event'] = temp_woe['Count'] - temp_woe['Event']
        temp_woe['Event_Rate'] = temp_woe['Event'] / temp_woe['Count']
        temp_woe['Non_Event_Rate'] = temp_woe['Non_Event'] / temp_woe['Count']

        # calculate distributions and woe
        total_event = temp_woe['Event'].sum()
        total_non_event = temp_woe['Non_Event'].sum()
        temp_woe['Event_Distribution'] = (temp_woe['Event'] + 0.5) / total_event
        temp_woe['Non_Event_Distribution'] = (temp_woe['Non_Event'] + 0.5) / total_non_event
        temp_woe['WOE'] = np.log(temp_woe['Event_Distribution'] / temp_woe['Non_Event_Distribution'])
        temp_woe['Information_Value'] = (temp_woe['Event_Distribution'] -
                                         temp_woe['Non_Event_Distribution']) * temp_woe['WOE']
        temp_woe['Variable_Name'] = X.name
        temp_woe = temp_woe[['Variable_Name', 'Category', 'Count', 'Event', 'Non_Event',
                             'Event_Rate', 'Non_Event_Rate', 'Event_Distribution', 'Non_Event_Distribution',
                             'WOE', 'Information_Value']]

        temp_woe['Information_Value'] = temp_woe['Information_Value'].sum()
        temp_woe = temp_woe.reset_index(drop=True)
        woe_mapping[str(X.name)] = dict(zip(temp_woe['Category'], temp_woe['WOE']))

        try:
            self.woe_df = self.woe_df.append(temp_woe, ignore_index=True)
            self.woe_bins.update(woe_mapping)
        except:
            self.woe_df = temp_woe
            self.woe_bins = woe_mapping

    def transform(self, X, y=None):

        X = X.fillna('NA')
        outX = X.copy(deep=True)
        transform_features = list(self.woe_bins.keys())
        transform_features = list(set(transform_features) & set(outX.columns))  # intersection

        for i in transform_features:

            tempX = outX[i]
            original_column_name = str(i)

            new_column_name = original_column_name

            if (is_numeric_dtype(tempX) == True):
                outX[new_column_name] = tempX
            else:
                outX[new_column_name] = tempX.map(self.woe_bins[original_column_name]).fillna(self.woenew)

        return outX
