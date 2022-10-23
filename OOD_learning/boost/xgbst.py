import numpy as np
import xgboost as xgb


class Xgboost:

    def __init__(self, n_estimators, max_depth, lambd, colsample_bytree, learning_rate, subsample,
                 dict_ood, beta_t, dict_cost):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lambd = lambd
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.beta_t = beta_t
        self.dict_ood = dict_ood
        self.dict_cost = dict_cost
        self.ood_1_t = 0
        self.ood_0_t = 0
        self.weight_0_0 = 0
        self.weight_1_0 = 0
        self.weight_0_1 = 0
        self.weight_1_1 = 0

    def predict(self, model, X_test, treshhold):
        dtest = xgb.DMatrix(X_test)
        scores = model.predict(dtest)

        scores = 1 / (1 + np.exp(-scores))
        predictions = (scores > treshhold).astype(int)

        return predictions

    def predict_proba(self, model, X_test):
        dtest = xgb.DMatrix(X_test)
        scores = model.predict(dtest)

        scores = 1 / (1 + np.exp(-scores))

        return scores

    def objective_function_ood(self, predt, dtrain):

        scores_1_0 = -1 / (1 + np.exp(predt))
        scores_0_0 = 1 / (1 + np.exp(-predt))

        scores_1_1 = scores_0_0 - self.ood_1_t
        scores_0_1 = scores_0_0 - self.ood_0_t

        sec_der_1_0 = np.multiply(-scores_1_0, 1 + scores_1_0)
        sec_der_0_0 = np.multiply(scores_0_0, 1 - scores_0_0)

        sec_der_1_1 = np.multiply(-scores_1_0, 1 + scores_1_0)
        sec_der_0_1 = np.multiply(-scores_1_0, 1 + scores_1_0)

        grad = np.multiply(dtrain.get_label(), scores_1_0) \
               + np.multiply(self.weight_0_0, scores_0_0) + np.multiply(self.weight_1_1, scores_1_1) \
               + np.multiply(self.weight_0_1, scores_0_1)
        hess = np.abs(np.multiply(dtrain.get_label(), sec_der_1_0)
                      + np.multiply(self.weight_0_0, sec_der_0_0) + np.multiply(self.weight_1_1, sec_der_1_1)
                      + np.multiply(self.weight_0_1, sec_der_0_1))

        return grad, hess

    def objective_function_id(self, predt, dtrain):
        scores_1_0 = -1 / (1 + np.exp(predt))
        scores_0_0 = 1 / (1 + np.exp(-predt))

        sec_der_1_0 = np.multiply(-scores_1_0, 1 + scores_1_0)
        sec_der_0_0 = np.multiply(scores_0_0, 1 - scores_0_0)

        grad = np.multiply(dtrain.get_label(), scores_1_0) \
               + np.multiply(self.weight_0_0, scores_0_0)
        hess = np.abs(np.multiply(dtrain.get_label(), sec_der_1_0)
                      + np.multiply(self.weight_0_0, sec_der_0_0))

        return grad, hess
