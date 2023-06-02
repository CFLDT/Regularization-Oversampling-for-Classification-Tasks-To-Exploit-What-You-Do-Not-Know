from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AlphaDropout, BatchNormalization, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers


class NN:

    def __init__(self, batch_size, epochs, learning_rate, depth, nodes_mult, alpha_dropout, lambd,
                 dict_ood, beta_t, dict_cost):

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.depth = depth
        self.alpha_dropout = alpha_dropout
        self.nodes_mult = nodes_mult
        self.lambd = lambd
        self.beta_t = beta_t
        self.dict_ood = dict_ood
        self.dict_cost = dict_cost

    def predict(self, model, X_test, treshhold):

        scores = model.predict(X_test, verbose=0)

        scores = 1 / (1 + np.exp(-scores))

        predictions = (scores > treshhold).astype(int)
        return predictions

    def predict_proba(self, model, X_test):

        scores = np.squeeze(model.predict(X_test, verbose=0))

        scores = 1 / (1 + np.exp(-scores))

        return scores

    def fitting(self, X, y):

        data_dimension = X.shape[1]
        data_amount = X.shape[0]

        model = Sequential()

        if self.depth >= 1:
            if self.depth != 1:
                model.add(tf.keras.layers.Dense(int(self.nodes_mult), kernel_initializer='lecun_normal',
                                                activation='selu'))
            if self.depth == 1:
                model.add(
                    tf.keras.layers.Dense(int(self.nodes_mult), kernel_initializer='lecun_normal',
                                          activation='selu', kernel_regularizer=regularizers.L2(self.lambd)))
            model.add(AlphaDropout(self.alpha_dropout))

        if self.depth >= 2:
            if self.depth != 2:
                model.add(
                    tf.keras.layers.Dense(int(self.nodes_mult / 2), kernel_initializer='lecun_normal',
                                          activation='selu'))
            if self.depth == 2:
                model.add(tf.keras.layers.Dense(int(self.nodes_mult / 2), kernel_initializer='lecun_normal',
                                                activation='selu', kernel_regularizer=regularizers.L2(self.lambd)))
            model.add(AlphaDropout(self.alpha_dropout))

        if self.depth >= 3:

            if self.depth != 3:
                model.add(
                    tf.keras.layers.Dense(int(self.nodes_mult / 3), kernel_initializer='lecun_normal',
                                          activation='selu'))
            if self.depth == 3:
                model.add(tf.keras.layers.Dense(int(self.nodes_mult / 3), kernel_initializer='lecun_normal',
                                                activation='selu', kernel_regularizer=regularizers.L2(self.lambd)))
            model.add(AlphaDropout(self.alpha_dropout))

        model.add(Dense(1, kernel_initializer='lecun_normal', activation='linear'))

        def loss_function(dtrain, predt):

            index_0 = [0]
            w_0_0 = tf.gather(dtrain, index_0, axis=1)

            index_1 = [1]
            w_1_0 = tf.gather(dtrain, index_1, axis=1)

            index_2 = [2]
            w_0_1 = tf.gather(dtrain, index_2, axis=1)

            index_3 = [3]
            w_1_1 = tf.gather(dtrain, index_3, axis=1)

            p = 1 / (1 + K.exp(-predt))

            loss_1_1 = self.ood_1_t * K.log(self.ood_1_t / p) + (1 - self.ood_1_t) * K.log(
                ((1 - self.ood_1_t) / (1 - p)))
            loss_0_1 = self.ood_0_t * K.log(self.ood_0_t / p) + (1 - self.ood_0_t) * K.log(
                ((1 - self.ood_0_t) / (1 - p)))
            loss_1_0 = K.log(1 + K.exp(-predt))
            loss_0_0 = K.log(1 + K.exp(predt))

            objective = K.dot(tf.transpose(w_1_1), loss_1_1) + \
                        K.dot(tf.transpose(w_0_1), loss_0_1) + \
                        K.dot(tf.transpose(w_1_0), loss_1_0) + \
                        K.dot(tf.transpose(w_0_0), loss_0_0)

            return objective

        def loss_function_1(dtrain, predt):

            index_0 = [0]
            w_0 = tf.gather(dtrain, index_0, axis=1)

            index_1 = [1]
            w_1 = tf.gather(dtrain, index_1, axis=1)

            loss_1 = K.log(1 + K.exp(-predt))
            loss_0 = K.log(1 + K.exp(predt))

            objective = K.dot(tf.transpose(w_1), loss_1) + \
                        K.dot(tf.transpose(w_0), loss_0)

            return objective

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if np.any(self.weight_1_1):
            model.compile(loss=loss_function, optimizer=opt)

        else:
            model.compile(loss=loss_function_1, optimizer=opt)

        y = y.astype('float64')
        X = X.astype('float64')
        model.fit(X, y, batch_size=int(self.batch_size * data_amount), epochs=self.epochs,
                  verbose=0, validation_data=None, use_multiprocessing=False)

        return model
