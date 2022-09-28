import pandas as pd
import numpy as np


class DataHandler:

    @staticmethod
    def create_dummies(data_train, data_test, columns):

        data_train = data_train.copy()
        if data_test is not None:
            data_test = data_test.copy()
        na_string = "_nan"
        for col in columns:
            column = col + na_string
            data_train[column] = data_train[col].isnull().map({True: 'T', False: 'F'})
            if data_test is not None:
                data_test[column] = data_test[col].isnull().map({True: 'T', False: 'F'})
        return data_train, data_test

    @staticmethod
    def impute_median(data_train, data_test, columns):

        data_train = data_train.copy()
        if data_test is not None:
            data_test = data_test.copy()
        array_median = np.empty(len(columns))
        for i, col in enumerate(columns):
            median = data_train[col].median()
            array_median[i] = median
            data_train[col].fillna(median, inplace=True)
            if data_test is not None:
                data_test[col].fillna(median, inplace=True)
        return data_train, data_test, array_median

    @staticmethod
    def impute_values(data_train, data_test, columns, values):

        data_train = data_train.copy()
        if data_test is not None:
            data_test = data_test.copy()
        for i, col in enumerate(columns):
            data_train[col].fillna(values[i], inplace=True)
            if data_test is not None:
                data_test[col].fillna(values[i], inplace=True)
        return data_train, data_test

    @staticmethod
    def remove_na(data, columns):

        data = data.copy()
        if columns in data.columns:
            data = data.dropna(subset=[columns])
        print('The shape of the dataset not including empty values equals ' + str(data.shape))
        return data

    @staticmethod
    def deleter(data, columns):
        data = data.copy()
        data = data.drop(columns, axis=1)
        return data

    @staticmethod
    def date_ymd_converter(data, columns):
        data = data.copy()
        for col in columns:
            data[col] = pd.to_datetime(data[col], errors='ignore', format='%Y%m%d')

        return data

    @staticmethod
    def date_ym_converter(data, columns):
        data = data.copy()
        for col in columns:
            data[col] = pd.to_datetime(data[col], errors='ignore', format='%Y%m')

        return data

    @staticmethod
    def date_month_differencer(data, column1, column2):

        data = data.copy()
        data[column1 + '_' + column2 + '_diff'] = data[column1].sub(data[column2], axis=0).dt.days

        return data
