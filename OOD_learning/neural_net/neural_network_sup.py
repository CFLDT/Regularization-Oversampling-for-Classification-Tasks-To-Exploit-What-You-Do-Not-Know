from timeit import default_timer as timer
from .neural_network import NN
import numpy as np
import random


class NeuralNetwork(NN):

    def __init__(self, batch_size, epochs, learning_rate, depth, nodes_mult, alpha_dropout, lambd,
                 beta_t, dict_ood, dict_cost):
        super().__init__(batch_size, epochs, learning_rate, depth, nodes_mult, alpha_dropout, lambd,
                         dict_ood, beta_t, dict_cost)

    def predict(self, model, X_test, treshhold):
        scores = model.predict(X_test, verbose=0)
        scores = scores.reshape(-1, )
        predictions = (scores > treshhold).astype(int)
        return predictions

    def training(self, X, y, ood_train):
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

        weights = np.stack((self.weight_0_0, self.weight_1_0, self.weight_0_1, self.weight_1_1), axis=1)

        model = self.fitting(X, weights)

        endtimer = timer()

        return model, endtimer - starttimer
