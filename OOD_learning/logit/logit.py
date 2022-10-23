import numpy as np
import scipy.optimize
import warnings

warnings.filterwarnings('ignore')


class Lgt:

    def __init__(self, lambd, dict_ood, beta_t, dict_cost, theta=None):
        self.lambd = lambd
        self.theta = theta
        self.beta_t = beta_t
        self.dict_ood = dict_ood
        self.dict_cost = dict_cost
        self.ood_1_t = 0
        self.ood_0_t = 0

    def predict(self, X_predict, treshhold):
        scores = 1 / (1 + np.exp(-self.theta[0] - X_predict.dot(self.theta[1:])))
        predictions = (scores > treshhold).astype(int)

        return predictions

    def predict_proba(self, X_predict):
        scores = 1 / (1 + np.exp(-self.theta[0] - X_predict.dot(self.theta[1:])))
        return scores

    def optimization(self, obj_func, initial_theta):
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B",
                                          options={'disp': False, 'maxiter': 100000})
        theta_opt, func_min = opt_res.x, opt_res.fun
        return theta_opt, func_min

    def objective_function_ood(self, theta, X, div, weight_1_1, weight_0_1, weight_1_0, weight_0_0):
        scores = theta[0] + X.dot(theta[1:])

        loss_1_0 = np.log(1 + np.exp(-scores))
        loss_0_0 = np.log(1 + np.exp(scores))

        p = 1 / (1 + np.exp(-scores))

        loss_1_1 = self.ood_1_t * np.log(self.ood_1_t / p) + \
                   (1 - self.ood_1_t) * np.log((1 - self.ood_1_t) / ((1 - p)))

        loss_0_1 = self.ood_0_t * np.log(self.ood_0_t / p) + \
                   (1 - self.ood_0_t) * np.log((1 - self.ood_0_t) / ((1 - p)))

        objective = div * (weight_1_0.dot(loss_1_0) + weight_0_0.dot(loss_0_0)
                           + weight_0_1.dot(loss_0_1) + weight_1_1.dot(loss_1_1)) \
                    + self.lambd * np.sum(theta[1:] ** 2)

        return objective

    def objective_function_id(self, theta, X, div, weight_1_0, weight_0_0):
        scores = theta[0] + X.dot(theta[1:])

        loss_1_0 = np.log(1 + np.exp(-scores))
        loss_0_0 = np.log(1 + np.exp(scores))

        objective = div * (weight_1_0.dot(loss_1_0) + weight_0_0.dot(loss_0_0)) + self.lambd * np.sum(theta[1:] ** 2)

        return objective
