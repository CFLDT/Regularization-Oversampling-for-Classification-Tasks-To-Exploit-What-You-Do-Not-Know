import numpy as np
from ..logit import Logit
from ..boost import Xgbst
from ..neural_net import NeuralNetwork
from sklearn.linear_model import LogisticRegression

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC


class MethodLearner:

    @staticmethod
    def logit(opt_par_dict, alpha, beta, dict_cost, X_train, y_train, ood_train,id_samp):

        lambd = opt_par_dict.get("logit_lambd")
        beta_t = opt_par_dict.get("logit_beta_t")
        alpha_t = opt_par_dict.get("alpha_t")
        dict_ood = {1: opt_par_dict.get("logit_ood_1"), 0: opt_par_dict.get("logit_ood_0")}

        if ((dict_cost.get(str(0) + str(1)) == dict_cost.get(str(1) + str(1)) == 0) | (beta_t == 0)):
            X_train = X_train[ood_train == 0]
            y_train = y_train[ood_train == 0]
            id_samp = id_samp[ood_train == 0]
            ood_train = ood_train[ood_train == 0]


        if (alpha_t == 0):

            X_train = X_train[id_samp == 0]
            y_train = y_train[id_samp == 0]
            ood_train = ood_train[id_samp == 0]
            id_samp = id_samp[id_samp == 0]

        if lambd == 0:
            lgtr = LogisticRegression(penalty='l2', random_state=2290)
        else:
            lgtr = LogisticRegression(penalty='l2', C=1 / lambd, random_state=2290)

        clf = lgtr.fit(X_train, y_train)

        init_theta = np.insert(clf.coef_, 0, values=clf.intercept_)
        # init_theta = np.zeros(X_train.shape[1]+1)

        for key in dict_ood:
            dict_ood[key] = min(dict_ood[key] * alpha, 1)

        logist = Logit(lambd=lambd, dict_ood=dict_ood,
                       beta_t=beta_t, dict_cost=dict_cost)
        logist.fitting(X_train, y_train, ood_train, init_theta)

        # kernel = 1.0 * RBF(1.0)
        # logist = GaussianProcessClassifier(kernel=kernel, random_state = 0).fit(X_train, y_train)

        #logist = SVC(gamma=2, C=1,probability=True).fit(X_train, y_train)

        return logist

    @staticmethod
    def xgboost(opt_par_dict, alpha, beta, dict_cost, X_train, y_train, ood_train,id_samp):

        opt_max_depth = opt_par_dict.get("xg_max_depth")
        opt_n_estimators = opt_par_dict.get("xg_n_estimators")
        opt_lambd_xg = opt_par_dict.get("xg_lambd")
        opt_colsample_bytree_xg = opt_par_dict.get("xg_colsample_bytree")
        opt_learning_rate_xg = opt_par_dict.get("xg_learning_rate")
        opt_subsample_xg = opt_par_dict.get("xg_subsample")
        alpha_t = opt_par_dict.get("xg_alpha_t")
        beta_t = opt_par_dict.get("xg_beta_t")
        dict_ood = {1: opt_par_dict.get("xg_ood_1"), 0: opt_par_dict.get("xg_ood_0")}

        if ((dict_cost.get(str(0) + str(1)) == dict_cost.get(str(1) + str(1)) == 0) | (beta_t == 0)):
            X_train = X_train[ood_train == 0]
            y_train = y_train[ood_train == 0]
            id_samp = id_samp[ood_train == 0]
            ood_train = ood_train[ood_train == 0]


        if (alpha_t == 0):

            X_train = X_train[id_samp == 0]
            y_train = y_train[id_samp == 0]
            ood_train = ood_train[id_samp == 0]
            id_samp = id_samp[id_samp == 0]

        for key in dict_ood:
            dict_ood[key] = min(dict_ood[key] * alpha, 1)

        xgboost = Xgbst(n_estimators=opt_n_estimators, max_depth=opt_max_depth,
                        lambd=opt_lambd_xg, colsample_bytree=opt_colsample_bytree_xg, learning_rate=opt_learning_rate_xg,
                        subsample=opt_subsample_xg,
                        dict_cost=dict_cost,
                        beta_t=beta_t, dict_ood=dict_ood)

        model, time = xgboost.fitting(X_train, y_train, ood_train)

        return xgboost, model

    @staticmethod
    def nn(opt_par_dict, alpha, beta, dict_cost, X_train, y_train, ood_train,id_samp):

        opt_batch_size_nn = opt_par_dict.get("nn_batch_size")
        opt_epochs_nn = opt_par_dict.get("nn_epochs")
        opt_learning_rate_nn = opt_par_dict.get("nn_learning_rate")
        opt_depth_nn = opt_par_dict.get("nn_depth")
        opt_alpha_dropout_nn = opt_par_dict.get("nn_alpha_dropout")
        opt_nodes_mult_nn = opt_par_dict.get("nn_nodes_mult")
        opt_lamb_nn = opt_par_dict.get("nn_lambd")


        alpha_t = opt_par_dict.get("nn_alpha_t")
        beta_t = opt_par_dict.get("nn_beta_t")
        dict_ood = {1: opt_par_dict.get("nn_ood_1"), 0: opt_par_dict.get("nn_ood_0")}

        if ((dict_cost.get(str(0) + str(1)) == dict_cost.get(str(1) + str(1)) == 0) | (beta_t == 0)):
            X_train = X_train[ood_train == 0]
            y_train = y_train[ood_train == 0]
            id_samp = id_samp[ood_train == 0]
            ood_train = ood_train[ood_train == 0]

        if (alpha_t == 0):

            X_train = X_train[id_samp == 0]
            y_train = y_train[id_samp == 0]
            ood_train = ood_train[id_samp == 0]
            id_samp = id_samp[id_samp == 0]

        for key in dict_ood:
            dict_ood[key] = min(dict_ood[key] * alpha, 1)

        neuralnet = NeuralNetwork(batch_size=opt_batch_size_nn, epochs=opt_epochs_nn,
                                  learning_rate=opt_learning_rate_nn, depth=opt_depth_nn,
                                  alpha_dropout=opt_alpha_dropout_nn,
                                  nodes_mult=opt_nodes_mult_nn,lambd=opt_lamb_nn, dict_cost=dict_cost,
                                  beta_t=beta_t, dict_ood=dict_ood)

        model, time = neuralnet.training(X_train, y_train, ood_train)

        return neuralnet, model



