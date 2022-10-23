from timeit import default_timer as timer
import xgboost as xgb
from .xgbst import Xgboost
import numpy as np
import random


class Xgbst(Xgboost):

    def __init__(self, n_estimators, max_depth, lambd, colsample_bytree, learning_rate, subsample,
                 beta_t, dict_ood, dict_cost):

        super().__init__(n_estimators, max_depth, lambd, colsample_bytree, learning_rate, subsample,
                         dict_ood, beta_t, dict_cost)

    def fitting(self, X, y, ood_train):

        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        dict_weights_id = {}
        dict_weights_ood = {}

        for y_label in np.unique(y):
            dict_weights_id[y_label] = np.where(((y == y_label) & (ood_train == 0)),
                                                self.dict_cost.get(str(int(y_label)) + str(0)), 0)

            dict_weights_ood[y_label] = np.where(((y == y_label) & (ood_train == 1)),
                                                 self.dict_cost.get(str(int(y_label)) + str(1)), 0) * self.beta_t

        self.weight_1_1 = dict_weights_ood.get(1)
        self.weight_0_1 = dict_weights_ood.get(0)
        self.weight_1_0 = dict_weights_id.get(1)
        self.weight_0_0 = dict_weights_id.get(0)

        self.ood_1_t = self.dict_ood.get(1)
        self.ood_0_t = self.dict_ood.get(0)

        dtrain = xgb.DMatrix(X, label=self.weight_1_0)

        param = {'max_depth': self.max_depth, 'colsample_bytree': self.colsample_bytree,
                 'eta': self.learning_rate, 'lambda': self.lambd, 'subsample': self.subsample,
                 'min_child_weight': 0}

        n_estimators = self.n_estimators
        if (((self.dict_cost.get(str(0) + str(1)) == 0) & (self.dict_cost.get(str(1) + str(1)) == 0)) | (
                self.beta_t == 0)):
            model = xgb.train(param, dtrain=dtrain, num_boost_round=n_estimators, obj=self.objective_function_id)
        else:
            model = xgb.train(param, dtrain=dtrain, num_boost_round=n_estimators, obj=self.objective_function_ood)

        endtimer = timer()

        return model, endtimer - starttimer
