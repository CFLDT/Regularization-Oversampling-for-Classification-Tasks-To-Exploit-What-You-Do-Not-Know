import numpy as np
import pandas as pd
from OOD_learning.testing import performance_check
from pathlib import Path
from OOD_learning.plots_tables import plotting_critical_difference_plots

id_sampling = False

if id_sampling:
    id_sample = True
    ood_sample = True
    id_samp_dic = {"smote": False, "adasyn": False, 'rose': True, 'float': 0.5}
    name = "id_samp"
else:
    id_sample = False
    ood_sample = True
    id_samp_dic = {"smote": False, "adasyn": False, 'rose': False, 'float': 0.5}
    name = ""

base_path = Path(__file__).parent

# https://archive.ics.uci.edu/ml/datasets/ecoli
path = (base_path / "data/bal/UCI_Ecoli.csv").resolve()
df_ecoli = pd.read_csv(path, header=None)
df_ecoli.columns = df_ecoli.columns.map(str)
df_ecoli.loc[:, '7'].replace({'pp': 1, 'cp': 0, 'im': 0, 'imS': 0, 'imL': 0, 'imU': 0, 'om': 0, "omL": 0}, inplace=True)

X_ecoli = df_ecoli.drop('7', axis=1)
y_ecoli = df_ecoli.loc[:, '7']
ratio_ecoli = np.count_nonzero(y_ecoli) / len(y_ecoli)  # 0.182


# http://archive.ics.uci.edu/ml/datasets/mammographic+mass
# Column 0 (BI-RADS assessment) non predictive
path = (base_path / "data/bal/UCI_Mammographic_Masses.csv").resolve()
df_mammo = pd.read_csv(path, header=None)
df_mammo.columns = df_mammo.columns.astype(str)

# column with label 1 are ages
df_mammo['1'] = pd.to_numeric(df_mammo['1'], errors='coerce')
df_mammo['1'] = df_mammo['1'].astype('float64')

X_mammo = df_mammo.drop(['0', '5'], axis=1)
y_mammo = df_mammo.loc[:, '5']
ratio_mammo = np.count_nonzero(y_mammo) / len(y_mammo)  # 0.463


# https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
path = (base_path / "data/bal/UCI_Sonar.csv").resolve()
df_sonar = pd.read_csv(path, sep=",", header=None)
df_sonar.columns = df_sonar.columns.astype(str)

df_sonar.loc[:, '60'].replace({'R': 0, 'M': 1}, inplace=True)
X_sonar = df_sonar.drop('60', axis=1)
y_sonar = df_sonar.loc[:, '60']
ratio_sonar = np.count_nonzero(y_sonar) / len(y_sonar)  # 0.536


# https://archive.ics.uci.edu/ml/datasets/ionosphere
path = (base_path / "data/bal/UCI_Ionopshere.csv").resolve()
df_ionopshere = pd.read_csv(path, sep=",")

df_ionopshere.loc[:, 'label'].replace({'g': 0, 'b': 1}, inplace=True)
X_ionopshere = df_ionopshere.drop('label', axis=1)
y_ionopshere = df_ionopshere.loc[:, 'label']
ratio_ionopshere = np.count_nonzero(y_ionopshere) / len(y_ionopshere)  # 0.359


# https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
path = (base_path / "data/bal/UCI_Breast_Cancer_Wisconsin.csv").resolve()
df_breast_cancer_wis = pd.read_csv(path)

df_breast_cancer_wis.loc[:, 'diagnosis'].replace({'B': 0, 'M': 1}, inplace=True)
X_breast_cancer_wis = df_breast_cancer_wis.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y_breast_cancer_wis = df_breast_cancer_wis.loc[:, 'diagnosis']
ratio_breast_cancer_wis = np.count_nonzero(y_breast_cancer_wis) / len(y_breast_cancer_wis)  # 0.373


# https://archive.ics.uci.edu/ml/datasets/statlog+(heart)
path = (base_path / "data/bal/UCI_Heart.csv").resolve()
df_heart = pd.read_csv(path, sep=",")

df_heart.loc[:, 'presence'].replace({2: 1, 1: 0}, inplace=True)
X_heart = df_heart.drop('presence', axis=1)
y_heart = df_heart.loc[:, 'presence']
ratio_heart = np.count_nonzero(y_heart) / len(y_heart)  # 0.444


# https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
path = (base_path / "data/bal/UCI_Indian_Liver.csv").resolve()
df_indian_liver = pd.read_csv(path, sep=",", header=None)
df_indian_liver.columns = df_indian_liver.columns.astype(str)

df_indian_liver.loc[:, '10'].replace({2: 1, 1: 0}, inplace=True)
X_indian_liver = df_indian_liver.drop('10', axis=1)
y_indian_liver = df_indian_liver.loc[:, '10']
ratio_indian_liver = np.count_nonzero(y_indian_liver) / len(y_indian_liver)  # 0.286


# https://archive.ics.uci.edu/ml/datasets/liver+disorders
# column 5 represents alcohol consumption <= 0.5 --> 0, else 1
# column 6 is a train/test split indicator --> delete
path = (base_path / "data/bal/UCI_Liver_Disorder.csv").resolve()
df_liver_disorder = pd.read_csv(path, sep=",", header=None)
df_liver_disorder.columns = df_liver_disorder.columns.astype(str)

df_liver_disorder['5'].values[df_liver_disorder['5'].values > 0.5] = 1
df_liver_disorder['5'].values[df_liver_disorder['5'].values <= 0.5] = 0
df_liver_disorder.loc[:, '5'].replace({1: 0, 0: 1}, inplace=True)

X_liver_disorder = df_liver_disorder.drop(['5', '6'], axis=1)
y_liver_disorder = df_liver_disorder.loc[:, '5']
ratio_liver_disorder = np.count_nonzero(y_liver_disorder) / len(y_liver_disorder)  # 0.339


# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer?ref=datanews.io
path = (base_path / "data/bal/UCI_Breast_Cancer.csv").resolve()
df_breast_cancer = pd.read_csv(path, header=None, sep=",")
df_breast_cancer.columns = df_breast_cancer.columns.astype(str)

df_breast_cancer.loc[:, '0'].replace({'no-recurrence-events': 0, 'recurrence-events': 1}, inplace=True)

X_breast_cancer = df_breast_cancer.drop('0', axis=1)
y_breast_cancer = df_breast_cancer.loc[:, '0']
ratio_breast_cancer = np.count_nonzero(y_breast_cancer) / len(y_breast_cancer)  # 0.297


# https://archive.ics.uci.edu/ml/datasets/Statlog+%28Vehicle+Silhouettes%29
# van is 1
path = (base_path / "data/bal/UCI_Vehicle.csv").resolve()
df_vehicle = pd.read_csv(path, sep=",")

df_vehicle['class'] = np.where(df_vehicle['class'] == 'van', 1, 0)

X_vehicle = df_vehicle.drop('class', axis=1)
y_vehicle = df_vehicle.loc[:, 'class']
ratio_vehicle = np.count_nonzero(y_vehicle) / len(y_vehicle)  # 0.235


# https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
# long term is 1
path = (base_path / "data/bal/UCI_Contraceptive_Method.csv").resolve()
df_con_meth = pd.read_csv(path, sep=",")

df_con_meth['Contraceptive_method_used'] = np.where(df_con_meth['Contraceptive_method_used'] == 2, 1, 0)

df_con_meth['Wifes_education'] = df_con_meth['Wifes_education'].astype('string')
df_con_meth['Husbands_education'] = df_con_meth['Husbands_education'].astype('string')
df_con_meth['Number_of_children_ever_born'] = df_con_meth['Number_of_children_ever_born'].astype('string')
df_con_meth['Wifes_religion'] = df_con_meth['Wifes_religion'].astype('string')
df_con_meth['Wifes_now_working'] = df_con_meth['Wifes_now_working'].astype('string')
df_con_meth['Husbands_occupation'] = df_con_meth['Husbands_occupation'].astype('string')
df_con_meth['Standard-of-living_index'] = df_con_meth['Standard-of-living_index'].astype('string')
df_con_meth['Media_exposure'] = df_con_meth['Media_exposure'].astype('string')

X_con_meth = df_con_meth.drop('Contraceptive_method_used', axis=1)
y_con_meth = df_con_meth.loc[:, 'Contraceptive_method_used']
ratio_con_meth = np.count_nonzero(y_con_meth) / len(y_con_meth)  # 0.226


# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
path = (base_path / "data/bal/UCI_German_Credit.csv").resolve()
df_german_credit = pd.read_csv(path, sep=",", header=None)
df_german_credit.columns = df_german_credit.columns.astype(str)

df_german_credit.loc[:, '20'].replace({2: 1, 1: 0}, inplace=True)

X_german_credit = df_german_credit.drop('20', axis=1)
y_german_credit = df_german_credit.loc[:, '20']
ratio_german_credit = np.count_nonzero(y_german_credit) / len(y_german_credit)  # 0.300


# https://archive.ics.uci.edu/ml/datasets/credit+approval
path = (base_path / "data/bal/UCI_Credit_Approval.csv").resolve()
df_credit_approval = pd.read_csv(path, sep=",", header=None)
df_credit_approval.columns = df_credit_approval.columns.astype(str)

df_credit_approval['1'] = pd.to_numeric(df_credit_approval['1'], errors='coerce')
df_credit_approval['1'] = df_credit_approval['1'].astype('float64')
df_credit_approval['13'] = pd.to_numeric(df_credit_approval['13'], errors='coerce')
df_credit_approval['13'] = df_credit_approval['13'].astype('float64')

df_credit_approval.loc[:, '15'].replace({'+': 1, '-': 0}, inplace=True)
X_credit_approval = df_credit_approval.drop('15', axis=1)
y_credit_approval = df_credit_approval.loc[:, '15']
ratio_approval = np.count_nonzero(y_credit_approval) / len(y_credit_approval)  # 0.445


# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
path = (base_path / "data/bal/UCI_Bank_Marketing.csv").resolve()
df_bank_marketing = pd.read_csv(path, sep=";")
df_bank_marketing.columns = df_bank_marketing.columns.astype(str)

df_bank_marketing.loc[:, 'y'].replace({"yes": 1, 'no': 0}, inplace=True)

X_bank_marketing = df_bank_marketing.drop('y', axis=1)
y_bank_marketing = df_bank_marketing.loc[:, 'y']
ratio_bank_marketing = np.count_nonzero(y_bank_marketing) / len(y_bank_marketing)  # 0.109


# https://archive.ics.uci.edu/ml/datasets/madelon
path = (base_path / "data/bal/UCI_Madelon.csv").resolve()
df_madelon = pd.read_csv(path, sep=",")
df_madelon.columns = df_madelon.columns.astype(str)
df_madelon.loc[:, 'Class'].replace({2: 1, 1: 0}, inplace=True)

X_madelon = df_madelon.drop('Class', axis=1)
y_madelon = df_madelon.loc[:, 'Class']
ratio_madelon = np.count_nonzero(y_madelon) / len(y_madelon)

par_dict_samp = {'Logit': {'logit_lambd': [0, 0.1, 1, 10],
                           'logit_alpha_t': [0],
                           'logit_beta_t': [1],
                           'logit_ood_1': [0, 0.5, 1, 2, 10],
                           'logit_ood_0': [0, 0.5, 1]},
                 'XGBoost': {"xg_max_depth": [5, 10],
                             "xg_n_estimators": [50, 200],
                             "xg_lambd": [0, 10, 100],
                             "xg_colsample_bytree": [1],
                             "xg_learning_rate": [0.1],
                             "xg_subsample": [0.25, 0.5, 0.75],
                             'xg_alpha_t': [0],
                             "xg_beta_t": [1],
                             'xg_ood_1': [0, 0.5, 1, 2, 10],
                             'xg_ood_0': [0, 0.5, 1]},
                 'NeuralNet': {'nn_batch_size': [1],
                               'nn_epochs': [500],
                               'nn_learning_rate': [0.01],
                               'nn_depth': [2],
                               'nn_alpha_dropout': [0.1, 0.25,0.5],
                               'nn_nodes_mult': [50, 100],
                               'nn_lambd': [0,1,5],
                               'nn_alpha_t': [0],
                               'nn_beta_t': [1],
                               'nn_ood_1': [0, 0.5, 1, 2, 10],
                               'nn_ood_0': [0, 0.5, 1]}}


# name = name + '_BROOD_ROT'
# ood_samp_dic = {'number_of_dir_m': 30, 'query_strategy': ['outlying', 1], 'max_ood': 10000,
#                 'simple': True, 'h_strategy': 0, 'dist_id_ood': 4,'equal': False}
#
# cost_dic = {'00': [1], '10': [1], '01': [0], '11': [0]}
# cost_dic = {'00': [1], '10': [1], '01': [0, 1], '11': [0, 1]}

name = name + '_BROOD_KNN'
ood_samp_dic = {'number_of_dir_m': 30, 'query_strategy': ['outlying', 1], 'max_ood': 10000,
                'simple': True, 'h_strategy': 2, 'dist_id_ood': 1.5,'equal': False}

cost_dic = {'00': [1], '10': [1], '01': [1], '11': [1]}

meth = ['Logit', 'XGBoost']
#meth = []

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_ecoli' + name}


performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_ecoli, y=y_ecoli,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_mammo' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_mammo, y=y_mammo,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_sonar' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_sonar, y=y_sonar,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_ionopshere' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_ionopshere, y=y_ionopshere,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_breast_cancer_wisconsin' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_breast_cancer_wis, y=y_breast_cancer_wis,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_heart' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_heart, y=y_heart,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_indian_liver' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_indian_liver, y=y_indian_liver,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_liver_disorder' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_liver_disorder, y=y_liver_disorder,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_breast_cancer' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_breast_cancer, y=y_breast_cancer,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_vehicle' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_vehicle, y=y_vehicle,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_contraceptive_method'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_con_meth, y=y_con_meth,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_german_credit' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_german_credit, y=y_german_credit,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_credit_approval'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_credit_approval, y=y_credit_approval,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_bank_marketing'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_bank_marketing, y=y_bank_marketing,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')


task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'UCI_madelon'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_madelon, y=y_madelon,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='ROC')



plotting_critical_difference_plots.critical_difference_plotter(metrics=['ROC_AUC', 'AP', 'Disc_Cum_Gain'],
                                                               alpha=0.1, name='benchmark',models=4)



