import numpy as np
import pandas as pd
from pathlib import Path
from OOD_learning.plots_tables import plotting_critical_difference_plots
from OOD_learning.testing import performance_check

base_path = Path(__file__).parent

id_sampling = False

if id_sampling:
    id_sample = True
    ood_sample = True
    id_samp_dic = {"smote": False, "adasyn": False, 'rose': True, 'float': 0.1}
    name = "_rose"

else:
    id_sample = False
    ood_sample = True
    id_samp_dic = {"smote": False, "adasyn": False, 'rose': False, 'float': 0.1}
    name = ""


# def clean_imb_datasets():
#     # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
#     path = (base_path / "data/bal/UCI_German_Credit.csv").resolve()
#     df_german_credit = pd.read_csv(path, sep=",", header=None)
#     df_german_credit.columns = df_german_credit.columns.astype(str)
#
#     df_german_credit.loc[:, '20'].replace({2: 1, 1: 0}, inplace=True)
#     df_german_credit = df_german_credit.rename(columns={"20": "y"})
#
#     df_german_credit = df_german_credit.drop(
#         df_german_credit.query('y == 1').sample(frac=.8, random_state=2290).index)
#
#     path = (base_path / "data/imb/cleaned/UCI_German_Credit.csv").resolve()
#     df_german_credit.to_csv(path, index=True)
#
#     # https://archive.ics.uci.edu/ml/datasets/credit+approval
#     path = (base_path / "data/bal/UCI_Credit_Approval.csv").resolve()
#     df_credit_approval = pd.read_csv(path, sep=",", header=None)
#     df_credit_approval.columns = df_credit_approval.columns.astype(str)
#
#     df_credit_approval = df_credit_approval.rename(columns={"15": "y"})
#
#     df_credit_approval.loc[:, 'y'].replace({'+': 1, '-': 0}, inplace=True)
#
#     df_credit_approval = df_credit_approval.drop(
#         df_credit_approval.query('y == 1').sample(frac=.9, random_state=2290).index)
#
#     path = (base_path / "data/imb/cleaned/UCI_Credit_Approval.csv").resolve()
#     df_credit_approval.to_csv(path, index=True)
#
#     # https://www.kaggle.com/datasets/shebrahimi/financial-distress
#
#     path = (base_path / "data/imb/original/Financial_Distress.csv").resolve()
#     df_financial_distress = pd.read_csv(path, sep=",", index_col=0)
#
#     df_financial_distress.loc[df_financial_distress['Financial Distress']>=-0.5, 'Financial Distress'] = 0
#     df_financial_distress.loc[df_financial_distress['Financial Distress']<-0.5, 'Financial Distress'] = 1
#
#     path = (base_path / "data/imb/cleaned/Financial_Distress.csv").resolve()
#     df_financial_distress.to_csv(path, index=True)
#
#
#     # https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
#     # Ensemble boosted trees with synthetic features generation in
#     # application to bankruptcy prediction
#
#     path = (base_path / "data/imb/original/Bankruptcy.csv").resolve()
#     df_bankruptcy = pd.read_csv(path, sep=",", index_col=0)
#
#     df_bankruptcy.loc[:, 'class'] = df_bankruptcy.loc[:, 'class'].str.decode('utf-8')
#
#     path = (base_path / "data/imb/cleaned/Bankruptcy.csv").resolve()
#     df_bankruptcy.to_csv(path, index=True)
#
#     # https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html
#
#     path = (base_path / "data/imb/original/KDD_Cup.csv").resolve()
#     df_kdd = pd.read_csv(path, sep=",", index_col=0)
#
#     df_kdd = df_kdd.sample(frac=.15, random_state=2290)
#     df_kdd['ODATEDW'] = pd.to_datetime(df_kdd['ODATEDW'])
#     df_kdd['month_o'] = df_kdd['ODATEDW'].dt.month
#     df_kdd['year_o'] = df_kdd['ODATEDW'].dt.year
#     df_kdd = df_kdd.drop(['ODATEDW'], axis=1)
#
#     df_kdd['DOB'] = pd.to_datetime(df_kdd['DOB'])
#     df_kdd['month_b'] = df_kdd['DOB'].dt.month
#     df_kdd['year_b'] = df_kdd['DOB'].dt.year
#     df_kdd = df_kdd.drop(['DOB'], axis=1)
#
#     df_kdd['MAXADATE'] = pd.to_datetime(df_kdd['MAXADATE'])
#     df_kdd['month_max'] = df_kdd['MAXADATE'].dt.month
#     df_kdd['year_max'] = df_kdd['MAXADATE'].dt.year
#     df_kdd = df_kdd.drop(['MAXADATE'], axis=1)
#
#     df_kdd['MINRDATE'] = pd.to_datetime(df_kdd['MINRDATE'])
#     df_kdd['month_min'] = df_kdd['MINRDATE'].dt.month
#     df_kdd['year_min'] = df_kdd['MINRDATE'].dt.year
#     df_kdd = df_kdd.drop(['MINRDATE'], axis=1)
#
#     df_kdd['LASTDATE'] = pd.to_datetime(df_kdd['LASTDATE'])
#     df_kdd['month_las'] = df_kdd['LASTDATE'].dt.month
#     df_kdd['year_las'] = df_kdd['LASTDATE'].dt.year
#     df_kdd = df_kdd.drop(['LASTDATE'], axis=1)
#
#     df_kdd['FISTDATE'] = pd.to_datetime(df_kdd['FISTDATE'])
#     df_kdd['month_fir'] = df_kdd['FISTDATE'].dt.month
#     df_kdd['year_fir'] = df_kdd['FISTDATE'].dt.year
#     df_kdd = df_kdd.drop(['FISTDATE'], axis=1)
#
#     df_kdd['NEXTDATE'] = pd.to_datetime(df_kdd['NEXTDATE'])
#     df_kdd['month_nex'] = df_kdd['NEXTDATE'].dt.month
#     df_kdd['year_nex'] = df_kdd['NEXTDATE'].dt.year
#     df_kdd = df_kdd.drop(['NEXTDATE'], axis=1)
#
#     df_kdd = df_kdd[df_kdd.columns.drop(list(df_kdd.filter(regex='ADATE_')))]
#     for i in range(23):
#         df_kdd = df_kdd.drop(['RFA_' + str(i + 2)], axis=1)
#     df_kdd = df_kdd[df_kdd.columns.drop(list(df_kdd.filter(regex='RDATE_')))]
#     df_kdd = df_kdd[df_kdd.columns.drop(list(df_kdd.filter(regex='RAMNT_')))]
#
#     path = (base_path / "data/imb/cleaned/KDD_Cup.csv").resolve()
#     df_kdd.to_csv(path, index=True)
#
#     # https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#     path = (base_path / "data/imb/original/UCI_Bank_Marketing.csv").resolve()
#     df_bank_marketing = pd.read_csv(path, sep=";")
#     df_bank_marketing.columns = df_bank_marketing.columns.astype(str)
#
#     df_bank_marketing.loc[:, 'y'].replace({"yes": 1, 'no': 0}, inplace=True)
#
#     df_bank_marketing = df_bank_marketing.drop(
#         df_bank_marketing.query('y == 1').sample(frac=.8, random_state=2290).index)
#
#     path = (base_path / "data/imb/cleaned/UCI_Bank_Marketing.csv").resolve()
#     df_bank_marketing.to_csv(path, index=True)
#
#     # Credit Card Approval
#     # https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
#     # https://www.kaggle.com/code/stevenwhang/credit-card-acceptance-status-people-with-no-debt/data
#
#     path = (base_path / "data/imb/original/Credit_Card_Approval.csv").resolve()
#     df_credit_card_approval = pd.read_csv(path)
#
#     df_credit_card_approval = df_credit_card_approval.sample(frac=.5, random_state=2290)
#
#     path = (base_path / "data/imb/cleaned/Credit_Card_Approval.csv").resolve()
#     df_credit_card_approval.to_csv(path, index=True)
#
#     # Telco Customer Churn
#     # https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
#
#     path = (base_path / "data/imb/original/Telco_Customer_Churn.csv").resolve()
#     df_telco_cust_churn = pd.read_csv(path)
#
#     df_telco_cust_churn.loc[:, 'Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)
#
#     df_telco_cust_churn = df_telco_cust_churn.drop(
#         df_telco_cust_churn.query('Churn == 1').sample(frac=.8, random_state=2290).index)
#
#     path = (base_path / "data/imb/cleaned/Telco_Customer_Churn.csv").resolve()
#     df_telco_cust_churn.to_csv(path, index=True)
#
#     # Cost-sensitive learning for profit-driven credit-scoring
#     # https://www.tandfonline.com/doi/abs/10.1080/01605682.2020.1843975?journalCode=tjor20
#
#     path = (base_path / "data/imb/original/Credit_Scoring.xlsx").resolve()
#     df_credit_scoring = pd.read_excel(path)
#     df_credit_scoring = df_credit_scoring.drop('Days_late',
#                                                1)  # remove variable days late (if days late > 45 --> response 1)
#     df_credit_scoring = df_credit_scoring.iloc[1:, :]  # first row days late input error
#     df_credit_scoring = df_credit_scoring.drop(
#         df_credit_scoring.query('Default_45 == 1').sample(frac=.8, random_state=2290).index)
#
#     path = (base_path / "data/imb/cleaned/Credit_Scoring.csv").resolve()
#     df_credit_scoring.to_csv(path, index=True)
#
#     # private
#     # path = (base_path / "data/imb/original/Car_Insurance_Fraud.csv").resolve()
#     # df_car_insurance_fraud = pd.read_csv(path)
#     #
#     # df_car_insurance_fraud = df_car_insurance_fraud.sample(frac=.8, random_state=2290)
#     # df_car_insurance_fraud = df_car_insurance_fraud.drop(
#     #     df_car_insurance_fraud.query('y1 == 0').sample(frac=.9, random_state=2290).index)
#     #
#     # path = (base_path / "data/imb/cleaned/Car_Insurance_Fraud.csv").resolve()
#     # df_car_insurance_fraud.to_csv(path, index=True)
#
#     # https://github.com/JarFraud/FraudDetection
#     path = (base_path / "data/imb/original/Financial_Statement_Fraud.csv").resolve()
#     df_accounting_fraud = pd.read_csv(path)
#
#     fraud = df_accounting_fraud[df_accounting_fraud['misstate'] == 1]
#     fraud = fraud.drop_duplicates(['misstate', 'gvkey'], keep='first')
#     no_fraud = df_accounting_fraud[df_accounting_fraud['misstate'] == 0]
#     df_accounting_fraud = no_fraud.append(fraud)
#
#     df_accounting_fraud = df_accounting_fraud.drop(
#         df_accounting_fraud.query('misstate == 0').sample(frac=.9, random_state=2290).index)
#
#     path = (base_path / "data/imb/cleaned/Financial_Statement_Fraud.csv").resolve()
#     df_accounting_fraud.to_csv(path, index=True)
#
#     # private
#     # path = (base_path / "data/imb/original/APATA_Credit_Card_Fraud.csv").resolve()
#     # df_apata_credit_card_fraud = pd.read_csv(path)
#     #
#     # df_apata_credit_card_fraud.loc[:, 'fraud'].replace({'False': 0, 'True': 1}, inplace=True)
#     # df_apata_credit_card_fraud = df_apata_credit_card_fraud.sample(frac=.002, random_state=2290)
#     #
#     # path = (base_path / "data/imb/cleaned/APATA_Credit_Card_Fraud.csv").resolve()
#     # df_apata_credit_card_fraud.to_csv(path, index=True)
#
#     # https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#     # analysis on the training set, only using transaction information
#     path = (base_path / "data/imb/original/Customer_Trans_Fraud_Detection.csv").resolve()
#     df_customer_trans_fraud = pd.read_csv(path)
#
#     # df_customer_trans_fraud = df_customer_trans_fraud.drop(df_customer_trans_fraud.query('isFraud == 0').sample(frac=.9).index)
#     df_customer_trans_fraud = df_customer_trans_fraud.sample(frac=.02, random_state=2290)
#
#     path = (base_path / "data/imb/cleaned/Customer_Trans_Fraud_Detection.csv").resolve()
#     df_customer_trans_fraud.to_csv(path, index=True)
#
#     # https://www.kaggle.com/datasets/ealaxi/paysim1
#     # step represents unit of time in the real world, we take the first 8 time units
#     path = (base_path / "data/imb/original/Synthetic_Mobile_Money_Transactions_Fraud.csv").resolve()
#     df_synth_mobile_money_fraud = pd.read_csv(path)
#     df_synth_mobile_money_fraud = df_synth_mobile_money_fraud[(df_synth_mobile_money_fraud['step'] <= 8)]
#
#     def count_val_orig(x):
#         sub_df = df_synth_mobile_money_fraud.loc[((df_synth_mobile_money_fraud['nameOrig'] == x['nameOrig']) &
#                                                   (df_synth_mobile_money_fraud['step'] == x['step']) &
#                                                   (df_synth_mobile_money_fraud['index'] < x['index']))]
#         number_of_trans = len(sub_df)
#         return number_of_trans
#
#     def sum_val_orig(x):
#         sub_df = df_synth_mobile_money_fraud.loc[((df_synth_mobile_money_fraud['nameOrig'] == x['nameOrig']) &
#                                                   (df_synth_mobile_money_fraud['step'] == x['step']) &
#                                                   (df_synth_mobile_money_fraud['index'] < x['index']))]
#         total_sum = sub_df['amount'].sum()
#         return total_sum
#
#     def count_val_dest(x):
#         sub_df = df_synth_mobile_money_fraud.loc[((df_synth_mobile_money_fraud['nameDest'] == x['nameDest']) &
#                                                   (df_synth_mobile_money_fraud['step'] == x['step']) &
#                                                   (df_synth_mobile_money_fraud['index'] < x['index']))]
#         number_of_trans = len(sub_df)
#         return number_of_trans
#
#     def sum_val_dest(x):
#         sub_df = df_synth_mobile_money_fraud.loc[((df_synth_mobile_money_fraud['nameDest'] == x['nameDest']) &
#                                                   (df_synth_mobile_money_fraud['step'] == x['step']) &
#                                                   (df_synth_mobile_money_fraud['index'] < x['index']))]
#         total_sum = sub_df['amount'].sum()
#         return total_sum
#
#     df_synth_mobile_money_fraud['index'] = df_synth_mobile_money_fraud.index
#     df_synth_mobile_money_fraud['count_trans'] = df_synth_mobile_money_fraud.apply(count_val_orig, axis=1)
#     df_synth_mobile_money_fraud['count_dest'] = df_synth_mobile_money_fraud.apply(count_val_dest, axis=1)
#     df_synth_mobile_money_fraud['sum_trans'] = df_synth_mobile_money_fraud.apply(sum_val_orig, axis=1)
#     df_synth_mobile_money_fraud['sum_dest'] = df_synth_mobile_money_fraud.apply(sum_val_dest, axis=1)
#     df_synth_mobile_money_fraud['diff_orig'] = df_synth_mobile_money_fraud['oldbalanceOrg'] - \
#                                                df_synth_mobile_money_fraud[
#                                                    'newbalanceOrig']
#     df_synth_mobile_money_fraud['diff_dest'] = df_synth_mobile_money_fraud['oldbalanceDest'] - \
#                                                df_synth_mobile_money_fraud[
#                                                    'newbalanceDest']
#
#     df_synth_mobile_money_fraud = df_synth_mobile_money_fraud.drop(['index'], axis=1)
#     df_synth_mobile_money_fraud = df_synth_mobile_money_fraud.drop(
#         df_synth_mobile_money_fraud.query('isFraud == 0').sample(frac=.9).index)
#     path = (base_path / "data/imb/cleaned/Synthetic_Mobile_Money_Transactions_Fraud.csv").resolve()
#     df_synth_mobile_money_fraud.to_csv(path, index=True)
#
#     # https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions
#     path = (base_path / "data/imb/original/Synthetic_Credit_Card_Fraud.csv").resolve()
#     df_synth_credit_card = pd.read_csv(path)
#     df_synth_credit_card = df_synth_credit_card[(df_synth_credit_card['User'] <= 3)]
#
#     df_synth_credit_card = df_synth_credit_card.rename(columns={'Is Fraud?': 'Fraud'})
#     df_synth_credit_card = df_synth_credit_card.drop(
#         df_synth_credit_card[df_synth_credit_card['Fraud'] == 'No'].sample(frac=.93).index)
#
#     df_synth_credit_card['Amount'] = df_synth_credit_card['Amount'].replace('\$', '', regex=True)
#     df_synth_credit_card['Amount'] = df_synth_credit_card['Amount'].astype(float)
#     df_synth_credit_card['Date1'] = pd.to_datetime(df_synth_credit_card[['Year', 'Month', 'Day']])
#     df_synth_credit_card['Date'] = pd.to_datetime(df_synth_credit_card['Date1'].astype("string") + ' '
#                                                   + df_synth_credit_card['Time'])
#     df_synth_credit_card['Hour'] = df_synth_credit_card['Date'].dt.hour
#     df_synth_credit_card = df_synth_credit_card.drop(['Date1'], axis=1)
#
#     def count_val_orig_c(x):
#         sub_df = df_synth_credit_card.loc[((df_synth_credit_card['User'] == x['User']) &
#                                            (df_synth_credit_card['Date'] > x['Date'] - timedelta(hours=24)) &
#                                            (df_synth_credit_card['Date'] < x['Date']))]
#         number_of_trans = len(sub_df)
#         return number_of_trans
#
#     def sum_val_orig_c(x):
#         sub_df = df_synth_credit_card.loc[((df_synth_credit_card['User'] == x['User']) &
#                                            (df_synth_credit_card['Date'] > x['Date'] - timedelta(hours=24)) &
#                                            (df_synth_credit_card['Date'] < x['Date']))]
#         total_sum = sub_df['Amount'].sum()
#         return total_sum
#
#     def count_val_dest_c(x):
#         sub_df = df_synth_credit_card.loc[((df_synth_credit_card['Merchant Name'] == x['Merchant Name']) &
#                                            (df_synth_credit_card['Date'] > x['Date'] - timedelta(hours=24)) &
#                                            (df_synth_credit_card['Date'] < x['Date']))]
#         number_of_trans = len(sub_df)
#         return number_of_trans
#
#     def sum_val_dest_c(x):
#         sub_df = df_synth_credit_card.loc[((df_synth_credit_card['Merchant Name'] == x['Merchant Name']) &
#                                            (df_synth_credit_card['Date'] == x['Date']) &
#                                            (df_synth_credit_card['Date'] < x['Date']))]
#         total_sum = sub_df['Amount'].sum()
#         return total_sum
#
#     df_synth_credit_card['count_trans'] = df_synth_credit_card.apply(count_val_orig_c, axis=1)
#     df_synth_credit_card['count_dest'] = df_synth_credit_card.apply(count_val_dest_c, axis=1)
#     df_synth_credit_card['sum_trans'] = df_synth_credit_card.apply(sum_val_orig_c, axis=1)
#     df_synth_credit_card['sum_dest'] = df_synth_credit_card.apply(sum_val_dest_c, axis=1)
#
#     path = (base_path / "data/imb/cleaned/Synthetic_Credit_Card_Fraud.csv").resolve()
#     df_synth_credit_card.to_csv(path, index=True)


#clean_imb_datasets()

path = (base_path / "data/imb/cleaned/UCI_German_Credit.csv").resolve()
df_german_credit = pd.read_csv(path, sep=",", index_col=0)

X_german_credit = df_german_credit.drop('y', axis=1)
y_german_credit = df_german_credit.loc[:, 'y']
ratio_german_credit = np.count_nonzero(y_german_credit) / len(y_german_credit)  # 0.041


path = (base_path / "data/imb/cleaned/UCI_Credit_Approval.csv").resolve()
df_credit_approval = pd.read_csv(path, sep=",", index_col=0)

df_credit_approval['1'] = pd.to_numeric(df_credit_approval['1'], errors='coerce')
df_credit_approval['1'] = df_credit_approval['1'].astype('float64')
df_credit_approval['13'] = pd.to_numeric(df_credit_approval['13'], errors='coerce')
df_credit_approval['13'] = df_credit_approval['13'].astype('float64')

X_credit_approval = df_credit_approval.drop('y', axis=1)
y_credit_approval = df_credit_approval.loc[:, 'y']
ratio_approval = np.count_nonzero(y_credit_approval) / len(y_credit_approval)  # 0.038


path = (base_path / "data/imb/cleaned/Financial_Distress.csv").resolve()
df_financial_distress = pd.read_csv(path, index_col=0)
df_financial_distress.columns = df_financial_distress.columns.astype(str)

X_financial_distress = df_financial_distress.drop(['Financial Distress','Time'], axis=1)
y_financial_distress = df_financial_distress.loc[:, 'Financial Distress']
ratio_financial_distress = np.count_nonzero(y_financial_distress) / len(y_financial_distress)  # 0.0370


path = (base_path / "data/imb/cleaned/Bankruptcy.csv").resolve()
df_bankrupt = pd.read_csv(path, index_col=0)
df_bankrupt.columns = df_bankrupt.columns.astype(str)

X_bankrupt = df_bankrupt.drop('class', axis=1)
y_bankrupt = df_bankrupt.loc[:, 'class']
ratio_bankrupt = np.count_nonzero(y_bankrupt) / len(y_bankrupt)  # 0.0386


path = (base_path / "data/imb/cleaned/KDD_Cup.csv").resolve()
df_kdd = pd.read_csv(path, index_col=0)
df_kdd.columns = df_kdd.columns.astype(str)

df_kdd['TCODE'] = df_kdd['TCODE'].astype('string')
df_kdd['MSA'] = df_kdd['MSA'].astype('string')
df_kdd['ADI'] = df_kdd['ADI'].astype('string')
df_kdd['ZIP'] = df_kdd['ZIP'].astype('string')
df_kdd['DMA'] = df_kdd['DMA'].astype('string')
df_kdd['RFA_2F'] = df_kdd['RFA_2F'].astype('string')
df_kdd['CLUSTER2'] = df_kdd['CLUSTER2'].astype('string')
df_kdd[df_kdd.filter(like='month').columns] = df_kdd[df_kdd.filter(like='month').columns].astype('string')

X_kdd = df_kdd.drop(['TARGET_B', 'CONTROLN', 'TARGET_D', 'HPHONE_D'], axis=1)
y_kdd = df_kdd.loc[:, 'TARGET_B']
ratio_kdd = np.count_nonzero(y_kdd) / len(y_kdd)  #


path = (base_path / "data/imb/cleaned/UCI_Bank_Marketing.csv").resolve()
df_bank_marketing = pd.read_csv(path, index_col=0)
df_bank_marketing.columns = df_bank_marketing.columns.astype(str)

X_bank_marketing = df_bank_marketing.drop('y', axis=1)
y_bank_marketing = df_bank_marketing.loc[:, 'y']
ratio_bank_marketing = np.count_nonzero(y_bank_marketing) / len(y_bank_marketing)  # 0.0239


path = (base_path / "data/imb/cleaned/Credit_Card_Approval.csv").resolve()
df_credit_card_appr = pd.read_csv(path, index_col=0)

df_credit_card_appr.loc[:, 'Status'].replace({1: 0, 0: 1}, inplace=True)

X_credit_card_appr = df_credit_card_appr.drop(['Status', 'Applicant_ID', 'Total_Bad_Debt', 'Total_Good_Debt'], axis=1)
y_credit_card_appr = df_credit_card_appr['Status']
ratio_credit_card_appr = np.count_nonzero(y_credit_card_appr) / len(y_credit_card_appr)


path = (base_path / "data/imb/cleaned/Telco_Customer_Churn.csv").resolve()
df_telco_customer_churn = pd.read_csv(path, index_col=0)

X_telco_customer_churn = df_telco_customer_churn.drop(['Churn', 'customerID'], axis=1)
y_telco_customer_churn = df_telco_customer_churn['Churn']
ratio_telco_customer_churn = np.count_nonzero(y_telco_customer_churn) / len(y_telco_customer_churn)


path = (base_path / "data/imb/cleaned/Credit_Scoring.csv").resolve()
df_credit_scoring = pd.read_csv(path, index_col=0)

df_credit_scoring['Business_channel'] = df_credit_scoring['Business_channel'].astype('string')

X_credit_scoring = df_credit_scoring.drop(['Default_45', 'Test_set1', 'Test_set2', 'Test_set3', 'ID'], axis=1)
y_credit_scoring = df_credit_scoring['Default_45']
ratio_credit_scoring = np.count_nonzero(y_credit_scoring) / len(y_credit_scoring)


# path = (base_path / "data/imb/cleaned/Car_Insurance_Fraud.csv").resolve()
# df_car_insurance_fraud = pd.read_csv(path, index_col=0)
#
# df_car_insurance_fraud['DAY(SI01_D_DCL)'] = df_car_insurance_fraud['DAY(SI01_D_DCL)'].astype('string')
# df_car_insurance_fraud['DAY(SI01_D_SURV_SIN)'] = df_car_insurance_fraud['DAY(SI01_D_SURV_SIN)'].astype('string')
# df_car_insurance_fraud['MONTH(SI01_D_DCL)'] = df_car_insurance_fraud['MONTH(SI01_D_DCL)'].astype('string')
# df_car_insurance_fraud['MONTH(SI01_D_SURV_SIN)'] = df_car_insurance_fraud['MONTH(SI01_D_SURV_SIN)'].astype('string')
# df_car_insurance_fraud['WEEKDAY(SI01_D_DCL)'] = df_car_insurance_fraud['WEEKDAY(SI01_D_DCL)'].astype('string')
# df_car_insurance_fraud['WEEKDAY(SI01_D_SURV_SIN)'] = df_car_insurance_fraud['WEEKDAY(SI01_D_SURV_SIN)'].astype('string')
# df_car_insurance_fraud['SI01_CPOST'] = df_car_insurance_fraud['SI01_CPOST'].astype('string')
# df_car_insurance_fraud['CPOST_1'] = df_car_insurance_fraud['CPOST_1'].astype('string')
# df_car_insurance_fraud['CPOST_2'] = df_car_insurance_fraud['CPOST_2'].astype('string')
# df_car_insurance_fraud['CPOST_3'] = df_car_insurance_fraud['CPOST_3'].astype('string')
#
# y_car_insurance_fraud = df_car_insurance_fraud['y1']
# X_car_insurance_fraud = df_car_insurance_fraud.drop(['SI04_NO_SIN', 'y1', 'y2'], axis=1)
# ratio_car_insurance_fraud = np.count_nonzero(y_car_insurance_fraud) / len(y_car_insurance_fraud)


path = (base_path / "data/imb/cleaned/Financial_Statement_Fraud.csv").resolve()
df_accounting_fraud = pd.read_csv(path, index_col=0)
y_accounting_fraud = df_accounting_fraud['misstate']
X_accounting_fraud = df_accounting_fraud.drop(['fyear', 'gvkey', 'sich', 'insbnk', 'understatement', 'option',
                                               'p_aaer', 'new_p_aaer', 'misstate'], axis=1)
ratio_accounting_fraud = np.count_nonzero(y_accounting_fraud) / len(y_accounting_fraud)


# path = (base_path / "data/imb/cleaned/APATA_Credit_Card_Fraud.csv").resolve()
# df_apata_credit_card_fraud = pd.read_csv(path, index_col=0)
#
# df_apata_credit_card_fraud['categorical1'] = df_apata_credit_card_fraud['categorical1'].astype('string')
# df_apata_credit_card_fraud['categorical2'] = df_apata_credit_card_fraud['categorical2'].astype('string')
# df_apata_credit_card_fraud['categorical3'] = df_apata_credit_card_fraud['categorical3'].astype('string')
# df_apata_credit_card_fraud['categorical4'] = df_apata_credit_card_fraud['categorical4'].astype('string')
# df_apata_credit_card_fraud['categorical5'] = df_apata_credit_card_fraud['categorical5'].astype('string')
# df_apata_credit_card_fraud['categorical6'] = df_apata_credit_card_fraud['categorical6'].astype('string')
#
# y_apata_credit_card_fraud = df_apata_credit_card_fraud['fraud']
# X_apata_credit_card_fraud = df_apata_credit_card_fraud.drop('fraud', axis=1)
# ratio_apata_credit_card_fraud = np.count_nonzero(y_apata_credit_card_fraud) / len(
#     y_apata_credit_card_fraud)


path = (base_path / "data/imb/cleaned/Customer_Trans_Fraud_Detection.csv").resolve()
df_customer_trans_fraud = pd.read_csv(path, index_col=0)

df_customer_trans_fraud['card1'] = df_customer_trans_fraud['card1'].astype('string')
df_customer_trans_fraud['card2'] = df_customer_trans_fraud['card2'].astype('string')
df_customer_trans_fraud['card3'] = df_customer_trans_fraud['card3'].astype('string')
df_customer_trans_fraud['card5'] = df_customer_trans_fraud['card5'].astype('string')
df_customer_trans_fraud['addr1'] = df_customer_trans_fraud['card5'].astype('string')
df_customer_trans_fraud['addr2'] = df_customer_trans_fraud['card5'].astype('string')

# TransactionDT is a timedelta from a reference datetime (we delete this feature)
y_customer_trans_fraud = df_customer_trans_fraud['isFraud']
X_customer_trans_fraud = df_customer_trans_fraud.drop(['TransactionID', 'TransactionDT', 'isFraud'], axis=1)
ratio_customer_trans_fraud = np.count_nonzero(y_customer_trans_fraud) / len(y_customer_trans_fraud)


path = (base_path / "data/imb/cleaned/Synthetic_Mobile_Money_Transactions_Fraud.csv").resolve()
df_synth_mobile_money_fraud = pd.read_csv(path, index_col=0)

y_synth_mobile_money_fraud = df_synth_mobile_money_fraud['isFraud']
X_synth_mobile_money_fraud = df_synth_mobile_money_fraud.drop(['step', 'nameOrig', 'nameDest', 'isFraud'], axis=1)
ratio_synth_mobile_money_fraud = np.count_nonzero(y_synth_mobile_money_fraud) / len(
    y_synth_mobile_money_fraud)


path = (base_path / "data/imb/cleaned/Synthetic_Credit_Card_Fraud.csv").resolve()
df_synth_credit_card_fraud = pd.read_csv(path, index_col=0)

df_synth_credit_card_fraud.loc[:, 'Fraud'].replace({'No': 0, 'Yes': 1}, inplace=True)

df_synth_credit_card_fraud['Card'] = df_synth_credit_card_fraud['Card'].astype('string')
df_synth_credit_card_fraud['Year'] = df_synth_credit_card_fraud['Year'].astype('string')
df_synth_credit_card_fraud['Month'] = df_synth_credit_card_fraud['Month'].astype('string')
df_synth_credit_card_fraud['Day'] = df_synth_credit_card_fraud['Day'].astype('string')
df_synth_credit_card_fraud['Merchant Name'] = df_synth_credit_card_fraud['Merchant Name'].astype('string')
df_synth_credit_card_fraud['Zip'] = df_synth_credit_card_fraud['Zip'].astype('string')
df_synth_credit_card_fraud['MCC'] = df_synth_credit_card_fraud['MCC'].astype('string')

y_synth_credit_card_fraud = df_synth_credit_card_fraud['Fraud']
X_synth_credit_card_fraud = df_synth_credit_card_fraud.drop(['User', 'Fraud', 'Card', 'Date', 'Year', 'Time'], axis=1)
ratio_synth_credit_card_fraud = np.count_nonzero(y_synth_credit_card_fraud) / len(
    y_synth_credit_card_fraud)

par_dict_samp = {'Logit': {'logit_lambd': [0, 0.1, 1, 10],
                           'logit_alpha_t': [1],
                           'logit_beta_t': [1],
                           'logit_ood_1': [0, 1, 2, 5, 10, 100],
                           'logit_ood_0': [0]},
                 'XGBoost': {"xg_max_depth": [5, 10],
                             "xg_n_estimators": [50, 200],
                             "xg_lambd": [0, 10, 100],
                             "xg_colsample_bytree": [1],
                             "xg_learning_rate": [0.1],
                             "xg_subsample": [0.25, 0.5, 0.75],
                             'xg_alpha_t': [1],
                             "xg_beta_t": [1],
                             'xg_ood_1': [0, 1, 2, 5, 10, 100],
                             'xg_ood_0': [0]},
                 'NeuralNet': {'nn_batch_size': [1],
                               'nn_epochs': [500],
                               'nn_learning_rate': [0.01],
                               'nn_depth': [2],
                               'nn_alpha_dropout': [0.1, 0.25, 0.5],
                               'nn_nodes_mult': [50, 100],
                               'nn_lambd': [0, 1, 5],
                               'nn_alpha_t': [1],
                               'nn_beta_t': [1],
                               'nn_ood_1': [0, 1, 5, 10, 100, 1000],
                               'nn_ood_0': [0]}}

# name = name + '_BROOD_ROT'
# ood_samp_dic = {'number_of_dir_m': 30, 'query_strategy': ['label', 1], 'max_ood': 50000,
#                 'simple': True, 'h_strategy': 0, 'dist_id_ood': 4, 'sparse': False}

# cost_dic = {'00': [1], '10': [1], '01': [0], '11': [0]}
# cost_dic = {'00': [1], '10': [1], '01': [0, 1], '11': [0, 1]}

name = name + '_BROOD_KNN'
ood_samp_dic = {'number_of_dir_m': 30, 'query_strategy': ['label', 1], 'max_ood': 50000,
                'simple': True, 'h_strategy': 2, 'dist_id_ood': 1.5, 'equal': False}

cost_dic = {'00': [1], '10': [1], '01': [1], '11': [1]}

meth = ['Logit', 'XGBoost']
#meth = []

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'german_credit' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_german_credit, y=y_german_credit,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')


task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'credit_approval' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_credit_approval, y=y_credit_approval,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'financial_distress' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_financial_distress, y=y_financial_distress,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'bankruptcy' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_bankrupt, y=y_bankrupt,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'kdd' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_kdd, y=y_kdd,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'bank_marketing'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_bank_marketing, y=y_bank_marketing,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'telco_customer_churn'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_telco_customer_churn, y=y_telco_customer_churn,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'credit_card_approval'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_credit_card_appr, y=y_credit_card_appr,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'vub_credit_scoring'+name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_credit_scoring, y=y_credit_scoring,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'financial_statement_fraud_label' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_accounting_fraud, y=y_accounting_fraud,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

# task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'apata_credit_card_fraud_label' + name}
#
# performance_check(methods=meth,
#                   par_dict_init=par_dict_samp,
#                   X=X_apata_credit_card_fraud, y=y_apata_credit_card_fraud,
#                   cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
#                   task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

# task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'car_insurance_fraud_label' + name}
#
# performance_check(methods=meth,
#                   par_dict_init=par_dict_samp,
#                   X=X_car_insurance_fraud, y=y_car_insurance_fraud,
#                   cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
#                   task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')
#
task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'customer_trans_fraud_detection_label' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_customer_trans_fraud, y=y_customer_trans_fraud,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')


task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'synth_mobile_money_fraud_label' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_synth_mobile_money_fraud, y=y_synth_mobile_money_fraud,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

task_dict = {'id_samp': id_sample, 'ood_samp': ood_sample, 'name': 'synth_credit_card_fraud_label' + name}

performance_check(methods=meth,
                  par_dict_init=par_dict_samp,
                  X=X_synth_credit_card_fraud, y=y_synth_credit_card_fraud,
                  cost_dic=cost_dic, id_samp_dic=id_samp_dic, ood_samp_dic=ood_samp_dic,
                  task_dict=task_dict, fold=4, repeats=1, cross_val=True, cross_val_perf_ind='DCG')

plotting_critical_difference_plots.critical_difference_plotter(metrics=['ROC_AUC', 'AP', 'Disc_Cum_Gain'],
                                                               alpha=0.1, name='imb', models=9)
