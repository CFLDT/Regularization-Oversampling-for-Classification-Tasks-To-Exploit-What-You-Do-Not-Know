import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold
import copy

from ..design import MethodLearner
from ..design import PerformanceMetrics
from ..plots_tables import plot_performance_plots, performance_tables
from ..design import divide_clean_sample


def performance_check(methods, par_dict_init, X, y, cost_dic, id_samp_dic, ood_samp_dic, task_dict,
                      fold, repeats, cross_val=False, cross_val_perf_ind='ROC'):

    id_samp = task_dict['id_samp']
    ood_samp = task_dict['ood_samp']
    name = task_dict['name']

    cart_prod_cost = list(product(*cost_dic.values()))

    rskf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeats, random_state=2290)

    ood_list = []
    id_samp_list = []
    alpha_list = []
    beta_list = []
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    par_dict_list = []

    counter = 0
    for train_index, test_index in rskf.split(X, y):
        print('performance number ' + str(counter + 1))

        if cross_val == True:

            train_indexing = train_index

            y_train = y.iloc[train_index]
            rskf_2 = RepeatedStratifiedKFold(n_splits=int(fold - 1), n_repeats=1, random_state=2290)

            for train_index, validation_index in rskf_2.split(train_indexing, y_train):
                train_index = train_indexing[train_index]
                validation_index = train_indexing[validation_index]

                par_dict, ood_train, id_samp_train, alpha, beta, X_train, \
                y_train, X_val, y_val, datapipeline, st_scaler = cross_validation_t_v(
                    methods=methods, train_index=train_index, validation_index=validation_index,
                    par_dic=par_dict_init, X=X,
                    y=y,
                    cart_prod_cost=cart_prod_cost, cost_dic=cost_dic, id_samp_dic=id_samp_dic,
                    ood_samp_dic=ood_samp_dic, task_dict=task_dict,
                    perf_ind=cross_val_perf_ind)

                X_test = st_scaler.transform(np.array(datapipeline.pipeline_trans(X.iloc[test_index])))
                y_test = np.array(y.iloc[test_index])

                break

        if cross_val == False:
            par_dict = par_dict_init

            ood_train, id_samp_train, alpha, beta, X_train, \
            y_train, X_test, y_test, datapipeline, st_scaler = \
                divide_clean_sample(X, y, train_index, test_index,
                                    id_samp_dic, ood_samp_dic, id_samp, ood_samp)

        ood_list.append(ood_train)
        id_samp_list.append(id_samp_train)
        alpha_list.append(alpha)
        beta_list.append(beta)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        par_dict_list.append(par_dict)

        counter = counter + 1

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_df = pd.DataFrame()
    precis_dict = {}
    recall_dict = {}
    ap_df = pd.DataFrame()
    positive_found_df = pd.DataFrame()
    disc_cum_gain_df = pd.DataFrame()
    pred_prob_dict = {}
    true_outcome_dict = {}

    for ind, value in enumerate(cart_prod_cost):

        if (value[2] != value[3]):
            continue

        dict_cost = {}
        for i, key in enumerate(cost_dic):
            dict_cost[key] = value[i]

        if 'Logit' in methods:
            fpr_dict['Logit' + str(value)] = {}
            tpr_dict['Logit' + str(value)] = {}
            roc_auc_df['Logit' + str(value)] = ""
            precis_dict['Logit' + str(value)] = {}
            recall_dict['Logit' + str(value)] = {}
            ap_df['Logit' + str(value)] = ""
            positive_found_df['Logit' + str(value)] = ""
            disc_cum_gain_df['Logit' + str(value)] = ""
            pred_prob_dict['Logit' + str(value)] = {}
            true_outcome_dict['Logit' + str(value)] = {}

        if 'XGBoost' in methods:
            fpr_dict['XGBoost' + str(value)] = {}
            tpr_dict['XGBoost' + str(value)] = {}
            roc_auc_df['XGBoost' + str(value)] = ""
            precis_dict['XGBoost' + str(value)] = {}
            recall_dict['XGBoost' + str(value)] = {}
            ap_df['XGBoost' + str(value)] = ""
            positive_found_df['XGBoost' + str(value)] = ""
            disc_cum_gain_df['XGBoost' + str(value)] = ""
            pred_prob_dict['XGBoost' + str(value)] = {}
            true_outcome_dict['XGBoost' + str(value)] = {}

        if 'NeuralNet' in methods:

            fpr_dict['NeuralNet'+str(value)] = {}
            tpr_dict['NeuralNet'+str(value)] = {}
            roc_auc_df['NeuralNet'+str(value)] = ""
            precis_dict['NeuralNet'+str(value)] = {}
            recall_dict['NeuralNet'+str(value)] = {}
            ap_df['NeuralNet'+str(value)] = ""
            positive_found_df['NeuralNet'+str(value)] = ""
            disc_cum_gain_df['NeuralNet'+str(value)] = ""
            pred_prob_dict['NeuralNet'+str(value)] = {}
            true_outcome_dict['NeuralNet'+str(value)] = {}

        for i in range(int(fold * repeats)):

            ood_train = ood_list[i]
            id_samp_train = id_samp_list[i]
            alpha = alpha_list[i]
            beta = beta_list[i]
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_test = X_test_list[i]
            y_test = y_test_list[i]

            try:
                par_dict_opt = par_dict_list[i][ind]
            except:
                par_dict_opt = par_dict

            if 'Logit' in methods:
                model = MethodLearner.logit(par_dict_opt.get('Logit'), alpha, beta, dict_cost,
                                             X_train, y_train, ood_train, id_samp_train)

                predict_prob = model.predict_proba(X_test)
                fpr, tpr, roc, precis, recall, ap, \
                positive_found_n, disc_cum_gain_n = performances(predict_prob, y_test, alpha)

                fpr_dict['Logit' + str(value)][i] = fpr
                tpr_dict['Logit' + str(value)][i] = tpr
                roc_auc_df.loc[i, 'Logit' + str(value)] = roc
                precis_dict['Logit' + str(value)][i] = precis
                recall_dict['Logit' + str(value)][i] = recall
                ap_df.loc[i, 'Logit' + str(value)] = ap
                positive_found_df.loc[i, 'Logit' + str(value)] = positive_found_n
                disc_cum_gain_df.loc[i, 'Logit' + str(value)] = disc_cum_gain_n
                pred_prob_dict['Logit' + str(value)][i] = predict_prob
                true_outcome_dict['Logit' + str(value)][i] = y_test

            if 'XGBoost' in methods:
                xgboost, model = MethodLearner.xgboost(par_dict_opt.get('XGBoost'), alpha, beta, dict_cost,
                                                    X_train, y_train, ood_train, id_samp_train)

                predict_prob = xgboost.predict_proba(model, X_test)
                fpr, tpr, roc, precis, recall, ap, \
                positive_found_n, disc_cum_gain_n = performances(predict_prob, y_test, alpha)

                fpr_dict['XGBoost' + str(value)][i] = fpr
                tpr_dict['XGBoost' + str(value)][i] = tpr
                roc_auc_df.loc[i, 'XGBoost' + str(value)] = roc
                precis_dict['XGBoost' + str(value)][i] = precis
                recall_dict['XGBoost' + str(value)][i] = recall
                ap_df.loc[i, 'XGBoost' + str(value)] = ap
                positive_found_df.loc[i, 'XGBoost' + str(value)] = positive_found_n
                disc_cum_gain_df.loc[i, 'XGBoost' + str(value)] = disc_cum_gain_n
                pred_prob_dict['XGBoost' + str(value)][i] = predict_prob
                true_outcome_dict['XGBoost' + str(value)][i] = y_test

            if 'NeuralNet' in methods:
                neuralnet, model = MethodLearner.nn(par_dict_opt.get('NeuralNet'),alpha, beta, dict_cost,
                                                X_train, y_train, ood_train, id_samp_train)

                predict_prob = neuralnet.predict_proba(model, X_test)
                fpr, tpr, roc, precis, recall, ap, \
                positive_found_n, disc_cum_gain_n = performances( predict_prob, y_test,alpha)

                fpr_dict['NeuralNet'+str(value)][i] = fpr
                tpr_dict['NeuralNet'+str(value)][i] = tpr
                roc_auc_df.loc[i, 'NeuralNet'+str(value)] = roc
                precis_dict['NeuralNet'+str(value)][i] = precis
                recall_dict['NeuralNet'+str(value)][i] = recall
                ap_df.loc[i, 'NeuralNet'+str(value)] = ap
                positive_found_df.loc[i, 'NeuralNet'+str(value)] = positive_found_n
                disc_cum_gain_df.loc[i, 'NeuralNet'+str(value)] = disc_cum_gain_n
                pred_prob_dict['NeuralNet'+str(value)][i] = predict_prob
                true_outcome_dict['NeuralNet'+str(value)][i] = y_test

    ood_counter = 0
    minority_counter = 0
    majority_counter = 0
    for ood_array,y_train_array in zip(ood_list,y_train_list):
        ood_counter = ood_counter + np.count_nonzero(ood_array == 1)
        cond = ood_array == 1

        minority_counter = minority_counter+  np.count_nonzero(np.extract(cond, y_train_array) == 1)
        majority_counter = majority_counter + np.count_nonzero(np.extract(cond, y_train_array) == 0)

    print('the number of (1) ood samples equals ' + str(minority_counter/4))
    print('the number of (0) ood samples equals ' + str(majority_counter/4))
    print('average number of OOD samples equals '+ str(ood_counter/4))

    if roc_auc_df.empty == False:
        plot_performance_plots(name, fpr_dict, tpr_dict, roc_auc_df, precis_dict, recall_dict,
                               ap_df, positive_found_df, disc_cum_gain_df)

        dict_per = defaultdict(list)
        dict_per['roc_auc'] = []
        dict_per['ap'] = []
        dict_per['positive_found'] = []
        dict_per['disc_cum_gain'] = []

        performance_tables(name, roc_auc_df,
                           ap_df, positive_found_df, disc_cum_gain_df, dict_per)


def performances(predict_prob, y_test, alpha):
    try:
        disc_cum_gain_n, positive_found_n = PerformanceMetrics.investigated_n(predict_prob, y_test)
        fpr, tpr, roc = PerformanceMetrics.roc_auc(predict_prob, y_test)
        precis, recall, ap = PerformanceMetrics.pr_auc(predict_prob, y_test)
    except:
        predict_prob = alpha * np.ones(len(y_test))
        disc_cum_gain_n, positive_found_n = PerformanceMetrics.investigated_n(predict_prob, y_test)
        fpr, tpr, roc = PerformanceMetrics.roc_auc(predict_prob, y_test)
        precis, recall, ap = PerformanceMetrics.pr_auc(predict_prob, y_test)

    return fpr, tpr, roc, precis, recall, ap, positive_found_n, disc_cum_gain_n


def cross_validation_t_v(methods, par_dic, X, y, train_index, validation_index,
                         cart_prod_cost, cost_dic, id_samp_dic, ood_samp_dic, task_dict, perf_ind='AP'):
    id_samp = task_dict['id_samp']
    ood_samp = task_dict['ood_samp']
    name = task_dict['name']

    ood_train, id_samp, alpha, beta, X_train, \
    y_train, X_val, y_val, datapipeline, st_scaler = \
        divide_clean_sample(X, y, train_index, validation_index,
                            id_samp_dic, ood_samp_dic, id_samp, ood_samp)

    dict_array = [{} for x in range(len(cart_prod_cost))]

    for ind, value in enumerate(cart_prod_cost):

        par_dict = copy.deepcopy(par_dic)

        if (value[2] != value[3]):
            continue

        if (value[2] == value[3] == 0):
            par_dict['Logit']['logit_ood_1'] = [0]
            par_dict['Logit']['logit_ood_0'] = [0]
            par_dict['Logit']['logit_beta_t'] = [0]
            par_dict['XGBoost']['xg_ood_1'] = [0]
            par_dict['XGBoost']['xg_ood_0'] = [0]
            par_dict['XGBoost']['xg_beta_t'] = [0]

        dict_cost = {}
        for i, key in enumerate(cost_dic):
            dict_cost[key] = value[i]

        print('cross validation for ' + str(value))

        dict = copy.deepcopy(par_dict)
        for k in dict:
            dict[k].clear()

        if 'Logit' in methods:
            cart_prod_logs = list(product(*par_dict.get('Logit').values()))
            cart_prod_log = [i for i in cart_prod_logs if i[-2] >= i[-1]]  # ood_t_1 >= ood_t_0
            keys_log = par_dict.get('Logit').keys()
            par_dict_log = par_dict.fromkeys(keys_log)

            per_matrix_log = np.zeros(len(cart_prod_log))

        if 'XGBoost' in methods:
            cart_prod_xgs = list(product(*par_dict.get('XGBoost').values()))
            cart_prod_xg = [i for i in cart_prod_xgs if i[-2] >= i[-1]]  # ood_t_1 >= ood_t_0
            keys_xg = par_dict.get('XGBoost').keys()
            par_dict_xg = par_dict.fromkeys(keys_xg)

            per_matrix_xg = np.zeros(len(cart_prod_xg))

        if 'NeuralNet' in methods:
            cart_prod_nn = list(product(*par_dict.get('NeuralNet').values()))
            keys_nn = par_dict.get('NeuralNet').keys()
            par_dict_nn = par_dict.fromkeys(keys_nn)

            per_matrix_nn = np.zeros(len(cart_prod_nn))

        if 'Logit' in methods:

            for j, value in enumerate(cart_prod_log):
                for i, key in enumerate(keys_log):
                    par_dict_log.update({key: value[i]})

                model = MethodLearner.logit(par_dict_log, alpha, beta, dict_cost,
                                             X_train, y_train, ood_train, id_samp)

                predict_prob = model.predict_proba(X_val)
                fpr, tpr, roc, precis, recall, ap, \
                positive_found_n, disc_cum_gain_n = performances(predict_prob, y_val, alpha)

                if perf_ind == 'AP':
                    per_matrix_log[j] += ap

                if perf_ind == 'ROC':
                    per_matrix_log[j] += roc

                if perf_ind == 'DCG':
                    per_matrix_log[j] += disc_cum_gain_n

        if 'XGBoost' in methods:

            for j, value in enumerate(cart_prod_xg):
                for i, key in enumerate(keys_xg):
                    par_dict_xg.update({key: value[i]})

                xgboost, model = MethodLearner.xgboost(par_dict_xg, alpha, beta, dict_cost,
                                                    X_train, y_train, ood_train, id_samp)

                predict_prob = xgboost.predict_proba(model, X_val)
                fpr, tpr, roc, precis, recall, ap, \
                positive_found_n, disc_cum_gain_n = performances(predict_prob, y_val, alpha)

                if perf_ind == 'AP':
                    per_matrix_xg[j] += ap

                if perf_ind == 'ROC':
                    per_matrix_xg[j] += roc

                if perf_ind == 'DCG':
                    per_matrix_xg[j] += disc_cum_gain_n

        if 'NeuralNet' in methods:

            for j, value in enumerate(cart_prod_nn):
                for i, key in enumerate(keys_nn):
                    par_dict_nn.update({key: value[i]})

                neuralnet, model = MethodLearner.nn(par_dict_nn, alpha, beta, dict_cost,
                                                    X_train, y_train, ood_train, id_samp)

                predict_prob = neuralnet.predict_proba(model, X_val)
                fpr, tpr, roc, precis, recall, ap, \
                positive_found_n, disc_cum_gain_n = performances(predict_prob, y_val, alpha)

                if perf_ind == 'AP':
                    per_matrix_nn[j] += ap

                if perf_ind == 'ROC':
                    per_matrix_nn[j] += roc

                if perf_ind == 'DCG':
                    per_matrix_nn[j] += disc_cum_gain_n

        if 'Logit' in methods:

            max_ind = np.argmax(per_matrix_log)
            max_values = cart_prod_log[max_ind]

            for i, key in enumerate(keys_log):
                dict['Logit'][key] = max_values[i]

        if 'XGBoost' in methods:

            max_ind = np.argmax(per_matrix_xg)
            max_values = cart_prod_xg[max_ind]

            for i, key in enumerate(keys_xg):
                dict['XGBoost'][key] = max_values[i]

        if 'NeuralNet' in methods:

            max_ind = np.argmax(per_matrix_nn)
            max_values = cart_prod_nn[max_ind]

            for i, key in enumerate(keys_nn):
                dict['NeuralNet'][key] = max_values[i]

        dict_array[ind] = dict

    print(name + ' The optimal hyperparameters are' + str(dict_array))

    return dict_array, ood_train, id_samp, alpha, beta, X_train, \
           y_train, X_val, y_val, datapipeline, st_scaler
