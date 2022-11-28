import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from ..design import data_pipeline
from ..un_gen import BROOD



def divide_clean_sample(X, y, train_index, test_index, id_samp_dict, ood_samp_dic, id_samp, ood_samp):
    X_train, X_test, y_train, y_test = \
        divider(X, y, train_index, test_index)

    X_train, X_test, y_train, \
    y_test, datapipeline = cleaner(X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, ood_train, id_samp_train, st_scaler = \
        sampler(X_train, X_test, y_train,
                id_samp_dict, ood_samp_dic, id_samp, ood_samp)

    alpha, beta = constant_calculator(y_train, ood_train, id_samp_train)

    return ood_train, id_samp_train, alpha, beta, X_train, \
           y_train, X_test, y_test, datapipeline, st_scaler


def sampler(X_train_or, X_test_or, y_train, id_samp_dic,
            ood_samp_dic, id_samp=False, ood_samp=True):
    st_scaler = StandardScaler()
    st_scaler.fit(X_train_or)
    X_train = st_scaler.transform(X_train_or)
    X_test = st_scaler.transform(X_test_or)

    ood_train = np.zeros(y_train.shape[0])
    id_samp_train = np.zeros(y_train.shape[0])

    if id_samp == True:

        smote = id_samp_dic.get("smote")
        adasyn = id_samp_dic.get("adasyn")
        rose = id_samp_dic.get("rose")
        float = id_samp_dic.get("float")
        number = X_train.shape[0]

        if smote == True:
            ov = SMOTE(sampling_strategy=float, random_state=2290)

        elif adasyn == True:
            ov = ADASYN(sampling_strategy=float, random_state=2290)

        elif rose == True:
            #shrinkage is 1 (scale the rule of thumb selector)
            ov = RandomOverSampler(sampling_strategy=float, random_state=2290, shrinkage=1)
        try:
            X_id, y_id = ov.fit_resample(X_train, y_train)
        except ValueError:
            X_id = X_train
            y_id = y_train

        X_id = X_id[number:]
        y_id = y_id[number:]

        ood_train = np.append(ood_train, np.zeros(X_id.shape[0]))
        id_samp_train = np.append(id_samp_train, np.ones(X_id.shape[0]))

        X_train = np.append(X_train, X_id, axis=0)
        y_train = np.append(y_train, y_id, axis=0)

    if ood_samp == True:
        dist_id_ood = ood_samp_dic.get("dist_id_ood")
        query_strategy = ood_samp_dic.get("query_strategy")
        number_of_dir_m = ood_samp_dic.get("number_of_dir_m")
        max_ood = ood_samp_dic.get('max_ood')
        simple = ood_samp_dic.get("simple")
        h_strategy = ood_samp_dic.get("h_strategy")
        equal = ood_samp_dic.get("equal")

        ood_sampler = BROOD(number_of_dir_m=number_of_dir_m, dist_id_ood=dist_id_ood,
                            query_strategy=query_strategy, max_ood=max_ood, simple=simple,
                            h_strategy=h_strategy, equal = equal)
        X_ood, y_ood = ood_sampler.fit_resample(X_train, y_train, seed=2290)

        ood_train = np.append(ood_train, np.ones(X_ood.shape[0]))
        id_samp_train = np.append(id_samp_train, np.zeros(X_ood.shape[0]))

        X_train = np.append(X_train, X_ood, axis=0)
        y_train = np.append(y_train, y_ood, axis=0)

    return X_train, X_test, y_train, ood_train, id_samp_train, st_scaler


def divider(X, y, train_index, test_index):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test


def cleaner(X_train, X_test, y_train, y_test):
    datapipeline = data_pipeline.DataPipeline()
    X_train, X_test = datapipeline.pipeline_fit_trans(X_train, X_test, y_train)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test, datapipeline


def constant_calculator(array_y_train, array_ood_train, id_samp_train):
    array_y_train_ood = array_y_train[(array_ood_train == 1) & (id_samp_train == 0)]
    array_y_train_true_id = array_y_train[(array_ood_train == 0) & (id_samp_train == 0)]

    alpha = np.count_nonzero(array_y_train_true_id) / float(array_y_train_true_id.size)
    beta = float(array_y_train_ood.size) / float(array_y_train.size)

    return alpha, beta
