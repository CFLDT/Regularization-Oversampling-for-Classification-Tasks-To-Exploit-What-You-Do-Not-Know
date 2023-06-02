from itertools import product
import numpy as np

from ..plots_tables import dec_boundary_plotter
from ..design import MethodLearner
from ..design import divide_clean_sample
from OOD_learning.plots_tables import plotting_2d_plots


def decision_boundary(methods, par_dict, X, y, cost_dic, id_samp_dic, ood_samp_dic, task_dict):

    id_samp = task_dict['id_samp']
    ood_samp = task_dict['ood_samp']
    name = task_dict['name']

    train_index = list(np.linspace(0, len(y) - 1, num=len(y), dtype=int))
    test_index = list(np.linspace(0, len(y) - 1, num=len(y), dtype=int))

    ood_train, id_samp_train, alpha, beta, X_train, \
    y_train, X_test, y_test, datapipeline, st_scaler = \
        divide_clean_sample(X, y, train_index, test_index,
                            id_samp_dic, ood_samp_dic, id_samp, ood_samp)

    cart_prod_cost = list(product(*cost_dic.values()))

    plotting_2d_plots.twod_plots_plotter('toy', y_train, X_train, datapipeline,
                                         orig_space=False, training_space=True, directions=False, max_combos=None)

    for ind, value in enumerate(cart_prod_cost):

        if (value[2] != value[3]):
            continue

        dict_cost = {}
        for i, key in enumerate(cost_dic):
            dict_cost[key] = value[i]

        try:
            par_dict_cost = par_dict[ind]
        except:
            par_dict_cost = par_dict

        xx, yy = np.mgrid[X_train[:, 0].min() - 2:X_train[:, 0].max() + 2:.02,
                 X_train[:, 1].min() - 2:X_train[:, 1].max() + 2:.02]

        grid = np.c_[xx.ravel(), yy.ravel()]

        for method in methods:

            if method == 'Logit':

                model = MethodLearner.logit(par_dict_cost.get('Logit'), alpha, beta, dict_cost,
                                                   X_train, y_train, ood_train,id_samp_train)

                grid_probs_fr = model.predict_proba(grid)

                if ood_samp == True:
                    model = MethodLearner.logit(par_dict_cost.get('Logit'), alpha, beta, dict_cost,
                                                       X_train, ood_train, np.zeros(np.shape(ood_train)),id_samp_train)

                    grid_probs_ood = model.predict_proba(grid)


            elif method == 'XGBoost':

                xgboost, model = MethodLearner.xgboost(par_dict_cost.get('XGBoost'), alpha, beta, dict_cost,
                                                          X_train, y_train, ood_train,id_samp_train)

                grid_probs_fr = xgboost.predict_proba(model, grid)

                if ood_samp == True:
                    xgboost, model = MethodLearner.xgboost(par_dict_cost.get('XGBoost'), alpha, beta, dict_cost,
                                                              X_train, ood_train, np.zeros(np.shape(ood_train)),id_samp_train)

                    grid_probs_ood = xgboost.predict_proba(model, grid)

            elif method == 'NeuralNet':

                neuralnet,model = MethodLearner.nn(par_dict_cost.get('NeuralNet'), alpha, beta, dict_cost,
                                                          X_train, y_train, ood_train,id_samp_train)

                grid_probs_fr = neuralnet.predict_proba(model, grid)

                if ood_samp == True:

                    neuralnet,model = MethodLearner.nn(par_dict_cost.get('NeuralNet'), alpha, beta, dict_cost,
                                                              X_train, ood_train, np.zeros(np.shape(ood_train)),id_samp_train)

                    grid_probs_ood = neuralnet.predict_proba(model, grid)

            else:

                continue

            if ood_samp == True:

                dec_boundary_plotter(name + str(value), method, X_train,
                                     y_train, xx, yy, grid_probs_fr, ood_train,
                                     grid_probs_ood=grid_probs_ood, ood_samp=ood_samp)

            else:

                dec_boundary_plotter(name + str(value), method, X_train,
                                     y_train, xx, yy, grid_probs_fr, ood_train,
                                     grid_probs_ood=None, ood_samp=ood_samp)

