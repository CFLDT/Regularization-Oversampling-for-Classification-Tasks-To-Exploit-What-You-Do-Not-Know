import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import random
from OOD_learning.testing import decision_boundary
import logging

logging.getLogger('matplotlib.font_manager').disabled = True

random.seed(2290)
np.random.seed(2290)
base_path = Path(__file__).parent

n_n = 200
n_fe = 10
n_fc = 60

x_n_1, x_n_2 = np.random.multivariate_normal([6, 5], [[1, 0], [0, 2]], size=n_n).T
x_fe_1, x_fe_2 = np.random.multivariate_normal([11.5, 1.5], [[0.3, 0], [0, 0.3]], size=n_fe).T
x_fc_1, x_fc_2 = np.random.multivariate_normal([8, 10.5], [[0.8, 0], [0, 0.5]], size=n_fc).T


# n_n = 50
# n_fe = 100
# n_fc = 600
#
# outer_radius = 1
# inner_radius = 0.7
# rho = np.sqrt(np.random.uniform(inner_radius**2,outer_radius**2,n_n))
# theta = np.random.uniform(0, 2*np.pi,n_n)
# x_n_1 = rho * np.cos(theta)
# x_n_2 = rho * np.sin(theta)
#
# outer_radius = 0.5
# inner_radius = 0
# rho = np.sqrt(np.random.uniform(inner_radius**2,outer_radius**2,n_fe))
# theta = np.random.uniform(0, 2*np.pi,n_fe)
# x_fe_1 = rho * np.cos(theta)
# x_fe_2 = rho * np.sin(theta)
# rho = np.sqrt(np.random.uniform(inner_radius**2,outer_radius**2,n_fc))
# theta = np.random.uniform(0, 2*np.pi,n_fc)
# x_fc_1 = rho * np.cos(theta)
# x_fc_2 = rho * np.sin(theta)

x_n_1 = x_n_1.reshape(-1, 1)
x_n_2 = x_n_2.reshape(-1, 1)
x_fe_1 = x_fe_1.reshape(-1, 1)
x_fe_2 = x_fe_2.reshape(-1, 1)
x_fc_1 = x_fc_1.reshape(-1, 1)
x_fc_2 = x_fc_2.reshape(-1, 1)

y_n = np.zeros(n_n)
y_fc = np.ones(n_fc)
y_fe = np.ones(n_fe)

y_n = np.zeros(n_n)
y_fc = np.ones(n_fc)
y_fe = np.ones(n_fe)

y_fc = y_fc.copy()
idx = np.linspace(0, len(y_fc)-1, len(y_fc)).astype(int)
#y_fc[np.random.choice(idx, size=int(0.3*len(idx)), replace=False)] = 2

y_fe = y_fe.copy()
idx = np.linspace(0, len(y_fe)-1, len(y_fe)).astype(int)
#y_fe[np.random.choice(idx, size=int(0*len(idx)), replace=False)] = 1

x_n = np.concatenate((x_n_1, x_n_2), axis=1)
x_fe = np.concatenate((x_fe_1, x_fe_2), axis=1)
x_fc = np.concatenate((x_fc_1, x_fc_2), axis=1)

X = np.concatenate((x_n, x_fe), axis=0)
X = np.concatenate((X, x_fc), axis=0)

y = np.concatenate((y_n, y_fe), axis=0)
y = np.concatenate((y, y_fc), axis=0)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit((np.array(X)))
X = min_max_scaler.transform(np.array(X)) * 0.7 + 0.15

dataset = pd.DataFrame({'X1': X[:, 1], 'X2': X[:, 0], 'y': y})

covariates = dataset[['X1', 'X2']]
y = dataset['y']

column_headers = list(dataset.columns.values)
row_amount = len(covariates.index)

id_samp_dic = {"smote": False, "adasyn": False, 'rose': True, 'float': 0.5}

ood_samp_dic = {'number_of_dir_m': 50, 'query_strategy': ['outlying', 1],
                'simple': True, 'h_strategy': 2, 'dist_id_ood': 1, 'equal': False}

cost_dic = {'00': [1], '10': [1], '01': [0, 1], '11': [0, 1]}

task_dict = {'id_samp': False, 'ood_samp': True, 'name': 'Toy_Data_1'}

opt_par_dict_fr_samp = {'Logit': {'logit_lambd': 0,
                                  'logit_alpha_t': 1,
                                  'logit_beta_t': 1,
                                  'logit_ood_1': 1,
                                  'logit_ood_0': 1},
                        'XGBoost': {"xg_max_depth": 10,
                                    "xg_n_estimators": 50,
                                    "xg_lambd": 25,
                                    "xg_colsample_bytree": 1,
                                    "xg_learning_rate": 0.1,
                                    "xg_subsample": 0.5,
                                    'xg_alpha_t': 1,
                                    "xg_beta_t": 1,
                                    'xg_ood_1': 2,
                                    'xg_ood_0': 1},
                        'NeuralNet' : {'nn_batch_size': 1,
                                     'nn_epochs': 500,
                                     'nn_learning_rate': 0.01,
                                     'nn_depth': 2,
                                     'nn_alpha_dropout':0.5,
                                     'nn_nodes_mult': 100,
                                     'nn_lambd':5,
                                     'nn_alpha_t': 1,
                                     'nn_beta_t': 1,
                                     'nn_ood_1': 2,
                                     'nn_ood_0': 1}}

dec_bound = True
if dec_bound:
    decision_boundary([ 'XGBoost'], opt_par_dict_fr_samp,
                      covariates, y, cost_dic=cost_dic, id_samp_dic=id_samp_dic,
                      ood_samp_dic=ood_samp_dic, task_dict=task_dict)



