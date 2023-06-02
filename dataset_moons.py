import numpy as np
import pandas as pd
from sklearn import datasets
from pathlib import Path
import random
from OOD_learning.testing import decision_boundary
import logging
from OOD_learning.testing import performance_check

logging.getLogger('matplotlib.font_manager').disabled = True

random.seed(2290)
np.random.seed(2290)
base_path = Path(__file__).parent

covariates, y = datasets.make_moons(n_samples=1000, noise=0.1)
covariates = pd.DataFrame(covariates)
covariates.columns = covariates.columns.astype(str)
y = pd.Series(y)

id_samp_dic = {"smote": False, "adasyn": False, 'rose': False, 'float': 0.5}

ood_samp_dic = {'number_of_dir_m': 40, 'query_strategy': ['outlying', 1],
                'simple': False, 'h_strategy': 2, 'dist_id_ood': 0.75,'equal': False}

cost_dic = {'00': [1], '10': [1], '01': [0, 1], '11': [0, 1]}

task_dict = {'id_samp': False, 'ood_samp': True, 'name': 'Moons'}

opt_par_dict_fr_samp = {'Logit': {'logit_lambd': 0,
                                  'logit_alpha_t': 0,
                                  'logit_beta_t': 1,
                                  'logit_ood_1': 1,
                                  'logit_ood_0': 0.2},
                        'XGBoost': {"xg_max_depth": 5,
                                    "xg_n_estimators": 50,
                                    "xg_lambd": 100,
                                    "xg_colsample_bytree": 0,
                                    "xg_subsample": 0.3,
                                    "xg_learning_rate": 0.1,
                                    'xg_alpha_t': 0,
                                    "xg_beta_t": 1,
                                    'xg_ood_1': 1,
                                    'xg_ood_0': 0.2}}

dec_bound = True
if dec_bound == True:
    decision_boundary(['Logit', 'XGBoost'], opt_par_dict_fr_samp,
                      covariates, y, cost_dic=cost_dic, id_samp_dic=id_samp_dic,
                      ood_samp_dic=ood_samp_dic, task_dict=task_dict)

