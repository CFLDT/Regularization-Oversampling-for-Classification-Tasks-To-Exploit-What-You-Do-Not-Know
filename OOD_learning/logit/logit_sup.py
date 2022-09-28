from timeit import default_timer as timer
import numpy as np
import random
from .logit import Lgt


class Logit(Lgt):

    def __init__(self, lambd, dict_ood, beta_t, dict_cost, theta=None):

        super().__init__(lambd, dict_ood, beta_t, dict_cost, theta)

    def fitting(self, X, y, ood_train, init_theta):

        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        div = 1 / X.shape[0]

        dict_weights_id = {}
        dict_weights_ood = {}

        for y_label in np.unique(y):
            dict_weights_id[y_label] = np.where(((y == y_label) & (ood_train == 0)),
                                                self.dict_cost.get(str(int(y_label)) + str(0)), 0)

            dict_weights_ood[y_label] = np.where(((y == y_label) & (ood_train == 1)),
                                                 self.dict_cost.get(str(int(y_label)) + str(1)), 0) * self.beta_t

        weight_1_1 = dict_weights_ood.get(1)
        weight_0_1 = dict_weights_ood.get(0)
        weight_1_0 = dict_weights_id.get(1)
        weight_0_0 = dict_weights_id.get(0)

        self.ood_1_t = self.dict_ood.get(1)
        self.ood_0_t = self.dict_ood.get(0)

        if (((self.dict_cost.get(str(0) + str(1)) == 0) & (self.dict_cost.get(str(1) + str(1)) == 0)) | (
                self.beta_t == 0)):
            def obj_func(theta):
                return self.objective_function_id(theta, X, div, weight_1_0, weight_0_0)

            self.theta, func_min = self.optimization(obj_func, init_theta)
        else:
            def obj_func(theta):
                return self.objective_function_ood(theta, X, div, weight_1_1, weight_0_1, weight_1_0, weight_0_0)

            self.theta, func_min = self.optimization(obj_func, init_theta)

        endtimer = timer()

        return func_min, endtimer - starttimer
