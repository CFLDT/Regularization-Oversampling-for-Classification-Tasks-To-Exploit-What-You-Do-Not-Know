import pandas as pd
import pathlib
from pathlib import Path
import os
import glob

base_path = Path(__file__).parent

imb = False

path = (base_path / "../tables/database results/balanced all").resolve()
all_files = glob.glob(os.path.join(path , "*.csv"))

roc_auc = []
ap = []
dcg = []

for filename in all_files:
    df = pd.read_csv(filename,index_col=0)
    df = df.rename(columns={"0": pathlib.PurePath(filename).name})
    df= df.T
    if (('ROC_AUC' in filename) and ('cv' not in filename)):
        if ('BROOD_ROT' in filename):
            df.index = df.index.str.replace('_BROOD_ROT','')
            df = df.rename(columns={c: c + '_BROOD_ROT' for c in df.columns if '(1, 1, 1, 1)' in c})
        if ('BROOD_KNN' in filename):
            df.index = df.index.str.replace('_BROOD_KNN','')
            df = df.rename(columns={c: c + '_BROOD_KNN' for c in df.columns if '(1, 1, 1, 1)' in c})
        if ('smote' in filename):
            df.index = df.index.str.replace('_smote','')
            df = df.add_suffix('_smote')
        if ('adasyn' in filename):
            df.index = df.index.str.replace('_adasyn','')
            df = df.add_suffix('_adasyn')
        if ('rose' in filename):
            df.index = df.index.str.replace('_rose','')
            df = df.add_suffix('_rose')
        roc_auc.append(df)
    if (('AP' in filename ) and ('cv' not in filename)):
        if ('BROOD_ROT' in filename):
            df.index = df.index.str.replace('_BROOD_ROT','')
            df = df.rename(columns={c: c + '_BROOD_ROT' for c in df.columns if '(1, 1, 1, 1)' in c})
        if ('BROOD_KNN' in filename):
            df.index = df.index.str.replace('_BROOD_KNN','')
            df = df.rename(columns={c: c + '_BROOD_KNN' for c in df.columns if '(1, 1, 1, 1)' in c})
        if ('smote' in filename):
            df.index = df.index.str.replace('_smote','')
            df = df.add_suffix('_smote')
        if ('adasyn' in filename):
            df.index = df.index.str.replace('_adasyn','')
            df = df.add_suffix('_adasyn')
        if ('rose' in filename):
            df.index = df.index.str.replace('_rose','')
            df = df.add_suffix('_rose')
        ap.append(df)
    if (('Disc_Cum_Gain' in filename) and ('cv' not in filename)):
        if ('BROOD_ROT' in filename):
            df.index = df.index.str.replace('_BROOD_ROT','')
            df = df.rename(columns={c: c + '_BROOD_ROT' for c in df.columns if '(1, 1, 1, 1)' in c})
        if ('BROOD_KNN' in filename):
            df.index = df.index.str.replace('_BROOD_KNN','')
            df = df.rename(columns={c: c + '_BROOD_KNN' for c in df.columns if '(1, 1, 1, 1)' in c})
        if ('smote' in filename):
            df.index = df.index.str.replace('_smote','')
            df = df.add_suffix('_smote')
        if ('adasyn' in filename):
            df.index = df.index.str.replace('_adasyn','')
            df = df.add_suffix('_adasyn')
        if ('rose' in filename):
            df.index = df.index.str.replace('_rose','')
            df = df.add_suffix('_rose')
        dcg.append(df)

roc_auc = pd.concat(roc_auc, axis=0)
ap = pd.concat(ap, axis=0)
dcg = pd.concat(dcg, axis=0)

roc_auc = roc_auc.groupby(level=0).sum()
ap = ap.groupby(level=0).sum()
dcg = dcg.groupby(level=0).sum()

roc_auc = roc_auc[ roc_auc.columns.drop(list(roc_auc.filter(regex='Logit\(1, 1, 1, 1\)|Logit\(1, 1, 0, 0\)_|XGBoost\(1, 1, 0, 0\)_ABROOD')))]
ap = ap[ ap.columns.drop(list(ap.filter(regex='Logit\(1, 1, 1, 1\)|Logit\(1, 1, 0, 0\)_|XGBoost\(1, 1, 0, 0\)_ABROOD')))]
dcg = dcg[ dcg.columns.drop(list(dcg.filter(regex='Logit\(1, 1, 1, 1\)|Logit\(1, 1, 0, 0\)_|XGBoost\(1, 1, 0, 0\)_ABROOD')))]

def add_hline(latex: str, index: int) -> str:

    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines).replace('NaN', '')

if imb == False:
    roc_auc = roc_auc.reindex(['UCI_ecoli_ROC_AUC.csv',
                               'UCI_mammo_ROC_AUC.csv',
                               'UCI_sonar_ROC_AUC.csv',
                               'UCI_ionopshere_ROC_AUC.csv',
                               'UCI_breast_cancer_wisconsin_ROC_AUC.csv',
                               'UCI_heart_ROC_AUC.csv',
                               'UCI_indian_liver_ROC_AUC.csv',
                               'UCI_liver_disorder_ROC_AUC.csv',
                               'UCI_breast_cancer_ROC_AUC.csv',
                               'UCI_vehicle_ROC_AUC.csv',
                               'UCI_contraceptive_method_ROC_AUC.csv',
                               'UCI_german_credit_ROC_AUC.csv',
                               'UCI_credit_approval_ROC_AUC.csv',
                               'UCI_bank_marketing_ROC_AUC.csv',
                               'UCI_madelon_ROC_AUC.csv'])

    roc_auc = roc_auc.reindex(columns=['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)',
                                       'XGBoost(1, 1, 1, 1)_BROOD_ROT',
                                       'XGBoost(1, 1, 1, 1)_BROOD_KNN'])

    roc_auc_c = roc_auc.copy()
    roc_auc_c = roc_auc_c.rename(index={'UCI_ecoli_ROC_AUC.csv':'UCI E.coli',
                               'UCI_mammo_ROC_AUC.csv':'UCI Mammographic Masses',
                               'UCI_sonar_ROC_AUC.csv':'UCI Sonar',
                               'UCI_ionopshere_ROC_AUC.csv':'UCI Ionosphere',
                               'UCI_breast_cancer_wisconsin_ROC_AUC.csv':'UCI Breast Cancer Wisconsin',
                               'UCI_heart_ROC_AUC.csv':'UCI Heart',
                               'UCI_indian_liver_ROC_AUC.csv':'UCI Indian Liver Patient',
                               'UCI_liver_disorder_ROC_AUC.csv':'UCI Liver Disorders',
                               'UCI_breast_cancer_ROC_AUC.csv':'UCI Breast Cancer ',
                               'UCI_vehicle_ROC_AUC.csv':'UCI Vehicle',
                               'UCI_contraceptive_method_ROC_AUC.csv':'UCI Contraceptive method',
                               'UCI_german_credit_ROC_AUC.csv':'UCI German Credit Card',
                               'UCI_credit_approval_ROC_AUC.csv':'UCI Credit Approval',
                               'UCI_bank_marketing_ROC_AUC.csv':'UCI Bank Marketing',
                               'UCI_madelon_ROC_AUC.csv':'UCI Madelon'})

    roc_auc_c = roc_auc_c.applymap('{:,.3f}'.format)

    latex =  roc_auc_c.to_latex()
    for i in range(14):
        latex = add_hline(latex= latex, index= 2*i+1)
    print(latex)

    ap = ap.reindex(['UCI_ecoli_AP.csv',
                               'UCI_mammo_AP.csv',
                               'UCI_sonar_AP.csv',
                               'UCI_ionopshere_AP.csv',
                               'UCI_breast_cancer_wisconsin_AP.csv',
                               'UCI_heart_AP.csv',
                               'UCI_indian_liver_AP.csv',
                               'UCI_liver_disorder_AP.csv',
                               'UCI_breast_cancer_AP.csv',
                               'UCI_vehicle_AP.csv',
                               'UCI_contraceptive_method_AP.csv',
                               'UCI_german_credit_AP.csv',
                               'UCI_credit_approval_AP.csv',
                               'UCI_bank_marketing_AP.csv',
                               'UCI_madelon_AP.csv'])

    ap = ap.reindex(columns=['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
                             'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose', 'XGBoost(1, 1, 1, 1)_BROOD_ROT',
                             'XGBoost(1, 1, 1, 1)_BROOD_KNN', 'XGBoost(1, 1, 1, 1)_BROOD_ROT_rose',
                             'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose'])

    ap_c = ap.copy()
    ap_c = ap_c.rename(index={'UCI_ecoli_AP.csv':'UCI E.coli',
                               'UCI_mammo_AP.csv':'UCI Mammographic Masses',
                               'UCI_sonar_AP.csv':'UCI Sonar',
                               'UCI_ionopshere_AP.csv':'UCI Ionosphere',
                               'UCI_breast_cancer_wisconsin_AP.csv':'UCI Breast Cancer Wisconsin',
                               'UCI_heart_AP.csv':'UCI Heart',
                               'UCI_indian_liver_AP.csv':'UCI Indian Liver Patient',
                               'UCI_liver_disorder_AP.csv':'UCI Liver Disorders',
                               'UCI_breast_cancer_AP.csv':'UCI Breast Cancer ',
                               'UCI_vehicle_AP.csv':'UCI Vehicle',
                               'UCI_contraceptive_method_AP.csv':'UCI Contraceptive method',
                               'UCI_german_credit_AP.csv':'UCI German Credit Card',
                               'UCI_credit_approval_AP.csv':'UCI Credit Approval',
                               'UCI_bank_marketing_AP.csv':'UCI Bank Marketing',
                               'UCI_madelon_AP.csv':'UCI Madelon'})

    ap_c = ap_c.reindex(columns=['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)',
                                       'XGBoost(1, 1, 1, 1)_BROOD_ROT',
                                       'XGBoost(1, 1, 1, 1)_BROOD_KNN'])


    ap_c = ap_c.applymap('{:,.3f}'.format)

    latex =  ap_c.to_latex()
    for i in range(14):
        latex = add_hline(latex= latex, index= 2*i+1)
    print(latex)

    dcg = dcg.reindex(['UCI_ecoli_Disc_Cum_Gain.csv',
                               'UCI_mammo_Disc_Cum_Gain.csv',
                               'UCI_sonar_Disc_Cum_Gain.csv',
                               'UCI_ionopshere_Disc_Cum_Gain.csv',
                               'UCI_breast_cancer_wisconsin_Disc_Cum_Gain.csv',
                               'UCI_heart_Disc_Cum_Gain.csv',
                               'UCI_indian_liver_Disc_Cum_Gain.csv',
                               'UCI_liver_disorder_Disc_Cum_Gain.csv',
                               'UCI_breast_cancer_Disc_Cum_Gain.csv',
                               'UCI_vehicle_Disc_Cum_Gain.csv',
                               'UCI_contraceptive_method_Disc_Cum_Gain.csv',
                               'UCI_german_credit_Disc_Cum_Gain.csv',
                               'UCI_credit_approval_Disc_Cum_Gain.csv',
                               'UCI_bank_marketing_Disc_Cum_Gain.csv',
                               'UCI_madelon_Disc_Cum_Gain.csv'])

    dcg = dcg.reindex(columns=['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
                               'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose',
                               'XGBoost(1, 1, 1, 1)_BROOD_ROT',
                               'XGBoost(1, 1, 1, 1)_BROOD_KNN', 'XGBoost(1, 1, 1, 1)_BROOD_ROT_rose',
                               'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose'])

    dcg_c = dcg.copy()
    dcg_c = dcg_c.rename(index={'UCI_ecoli_Disc_Cum_Gain.csv':'UCI E.coli',
                               'UCI_mammo_Disc_Cum_Gain.csv':'UCI Mammographic Masses',
                               'UCI_sonar_Disc_Cum_Gain.csv':'UCI Sonar',
                               'UCI_ionopshere_Disc_Cum_Gain.csv':'UCI Ionosphere',
                               'UCI_breast_cancer_wisconsin_Disc_Cum_Gain.csv':'UCI Breast Cancer Wisconsin',
                               'UCI_heart_Disc_Cum_Gain.csv':'UCI Heart',
                               'UCI_indian_liver_Disc_Cum_Gain.csv':'UCI Indian Liver Patient',
                               'UCI_liver_disorder_Disc_Cum_Gain.csv':'UCI Liver Disorders',
                               'UCI_breast_cancer_Disc_Cum_Gain.csv':'UCI Breast Cancer ',
                               'UCI_vehicle_Disc_Cum_Gain.csv':'UCI Vehicle',
                               'UCI_contraceptive_method_Disc_Cum_Gain.csv':'UCI Contraceptive method',
                               'UCI_german_credit_Disc_Cum_Gain.csv':'UCI German Credit Card',
                               'UCI_credit_approval_Disc_Cum_Gain.csv':'UCI Credit Approval',
                               'UCI_bank_marketing_Disc_Cum_Gain.csv':'UCI Bank Marketing',
                               'UCI_madelon_Disc_Cum_Gain.csv':'UCI Madelon'})

    dcg_c = dcg_c.reindex(columns=['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)',
                                       'XGBoost(1, 1, 1, 1)_BROOD_ROT',
                                       'XGBoost(1, 1, 1, 1)_BROOD_KNN'])


    dcg_c = dcg_c.applymap('{:,.3f}'.format)

    latex = dcg_c.to_latex()
    for i in range(14):
        latex = add_hline(latex=latex, index=2 * i + 1)
    print(latex)


if imb == True:

    roc_auc = roc_auc.reindex(['bankruptcy_ROC_AUC.csv',
           'financial_distress_ROC_AUC.csv',
           'german_credit_ROC_AUC.csv',
           'credit_approval_ROC_AUC.csv',
           'bank_marketing_ROC_AUC.csv',
           'telco_customer_churn_ROC_AUC.csv',
           'credit_card_approval_ROC_AUC.csv',
           'vub_credit_scoring_ROC_AUC.csv',
           'kdd_ROC_AUC.csv',
           'car_insurance_fraud_label_ROC_AUC.csv',
           'apata_credit_card_fraud_label_ROC_AUC.csv',
           'financial_statement_fraud_label_ROC_AUC.csv',
           'customer_trans_fraud_detection_label_ROC_AUC.csv',
           'synth_mobile_money_fraud_label_ROC_AUC.csv',
           'synth_credit_card_fraud_label_ROC_AUC.csv'])

    roc_auc = roc_auc.reindex(columns =['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
        'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose', 'XGBoost(1, 1, 1, 1)_BROOD_ROT',
        'XGBoost(1, 1, 1, 1)_BROOD_KNN','XGBoost(1, 1, 1, 1)_BROOD_ROT_rose', 'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose'])

    roc_auc_c = roc_auc.copy()
    roc_auc_c = roc_auc_c.rename(index={'bankruptcy_ROC_AUC.csv':'ESA Polish Bankruptcy',
           'financial_distress_ROC_AUC.csv':'Financial Distress',
           'german_credit_ROC_AUC.csv':'UCI German Credit Card IMB',
           'credit_approval_ROC_AUC.csv':'UCI Credit Approval IMB',
           'bank_marketing_ROC_AUC.csv':'UCI Bank Marketing IMB',
           'telco_customer_churn_ROC_AUC.csv':'IBM Telco Customer Churn ',
           'credit_card_approval_ROC_AUC.csv':'ETL Credit Card Approval',
           'vub_credit_scoring_ROC_AUC.csv':'VUB Credit Scoring' ,
           'kdd_ROC_AUC.csv':'KDD Cup 1998',
           'car_insurance_fraud_label_ROC_AUC.csv':'(Priv.) Car Insurance Fraud ',
           'apata_credit_card_fraud_label_ROC_AUC.csv':'(Priv.) APATA Credit Card Fraud ',
           'financial_statement_fraud_label_ROC_AUC.csv':'JAR Financial Statement Fraud',
           'customer_trans_fraud_detection_label_ROC_AUC.csv':'IEEE-CIS Cus. Trans. Fraud',
           'synth_mobile_money_fraud_label_ROC_AUC.csv':'Synt. Mobile Trans. Fraud',
           'synth_credit_card_fraud_label_ROC_AUC.csv':'Synthetic Credit Card Fraud'})

    roc_auc_c_1 = roc_auc_c[['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
        'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose']]

    roc_auc_c_1 = roc_auc_c_1.applymap('{:,.3f}'.format)

    latex = roc_auc_c_1.to_latex()
    for i in range(14):
        latex = add_hline(latex=latex, index=2 * i + 1)
    print(latex)

    roc_auc_c_2 = roc_auc_c[['XGBoost(1, 1, 1, 1)_BROOD_ROT',
        'XGBoost(1, 1, 1, 1)_BROOD_KNN','XGBoost(1, 1, 1, 1)_BROOD_ROT_rose', 'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose']]

    roc_auc_c_2 = roc_auc_c_2.applymap('{:,.3f}'.format)

    latex = roc_auc_c_2.to_latex()
    for i in range(14):
        latex = add_hline(latex=latex, index=2 * i + 1)
    print(latex)

    ap = ap.reindex(['bankruptcy_AP.csv',
           'financial_distress_AP.csv',
           'german_credit_AP.csv',
           'credit_approval_AP.csv',
           'bank_marketing_AP.csv',
           'telco_customer_churn_AP.csv',
           'credit_card_approval_AP.csv',
           'vub_credit_scoring_AP.csv',
           'kdd_AP.csv',
           'car_insurance_fraud_label_AP.csv',
           'apata_credit_card_fraud_label_AP.csv',
           'financial_statement_fraud_label_AP.csv',
           'customer_trans_fraud_detection_label_AP.csv',
           'synth_mobile_money_fraud_label_AP.csv',
           'synth_credit_card_fraud_label_AP.csv'])

    ap = ap.reindex(columns =['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
        'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose', 'XGBoost(1, 1, 1, 1)_BROOD_ROT',
        'XGBoost(1, 1, 1, 1)_BROOD_KNN','XGBoost(1, 1, 1, 1)_BROOD_ROT_rose', 'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose'])


    ap_c = ap.copy()
    ap_c = ap_c.rename(index={'bankruptcy_AP.csv':'ESA Polish Bankruptcy',
           'financial_distress_AP.csv':'Financial Distress',
           'german_credit_AP.csv':'UCI German Credit Card IMB',
           'credit_approval_AP.csv':'UCI Credit Approval IMB',
           'bank_marketing_AP.csv':'UCI Bank Marketing IMB',
           'telco_customer_churn_AP.csv':'IBM Telco Customer Churn ',
           'credit_card_approval_AP.csv':'ETL Credit Card Approval',
           'vub_credit_scoring_AP.csv':'VUB Credit Scoring' ,
           'kdd_AP.csv':'KDD Cup 1998',
           'car_insurance_fraud_label_AP.csv':'(Priv.) Car Insurance Fraud ',
           'apata_credit_card_fraud_label_AP.csv':'(Priv.) APATA Credit Card Fraud ',
           'financial_statement_fraud_label_AP.csv':'JAR Financial Statement Fraud',
           'customer_trans_fraud_detection_label_AP.csv':'IEEE-CIS Cus. Trans. Fraud',
           'synth_mobile_money_fraud_label_AP.csv':'Synt. Mobile Trans. Fraud',
           'synth_credit_card_fraud_label_AP.csv':'Synthetic Credit Card Fraud'})

    ap_c_1 = ap_c[['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
        'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose']]
    
    ap_c_1 = ap_c_1.applymap('{:,.3f}'.format)

    latex = ap_c_1.to_latex()
    for i in range(14):
        latex = add_hline(latex=latex, index=2 * i + 1)
    print(latex)

    ap_c_2 = ap_c[['XGBoost(1, 1, 1, 1)_BROOD_ROT',
        'XGBoost(1, 1, 1, 1)_BROOD_KNN','XGBoost(1, 1, 1, 1)_BROOD_ROT_rose', 'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose']]

    ap_c_2 = ap_c_2.applymap('{:,.3f}'.format)


    latex = ap_c_2.to_latex()
    for i in range(14):
        latex = add_hline(latex=latex, index=2 * i + 1)
    print(latex)

    dcg = dcg.reindex(['bankruptcy_Disc_Cum_Gain.csv',
           'financial_distress_Disc_Cum_Gain.csv',
           'german_credit_Disc_Cum_Gain.csv',
           'credit_approval_Disc_Cum_Gain.csv',
           'bank_marketing_Disc_Cum_Gain.csv',
           'telco_customer_churn_Disc_Cum_Gain.csv',
           'credit_card_approval_Disc_Cum_Gain.csv',
           'vub_credit_scoring_Disc_Cum_Gain.csv',
           'kdd_Disc_Cum_Gain.csv',
           'car_insurance_fraud_label_Disc_Cum_Gain.csv',
           'apata_credit_card_fraud_label_Disc_Cum_Gain.csv',
           'financial_statement_fraud_label_Disc_Cum_Gain.csv',
           'customer_trans_fraud_detection_label_Disc_Cum_Gain.csv',
           'synth_mobile_money_fraud_label_Disc_Cum_Gain.csv',
           'synth_credit_card_fraud_label_Disc_Cum_Gain.csv'])

    dcg = dcg.reindex(columns =['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
        'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose', 'XGBoost(1, 1, 1, 1)_BROOD_ROT',
        'XGBoost(1, 1, 1, 1)_BROOD_KNN','XGBoost(1, 1, 1, 1)_BROOD_ROT_rose', 'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose'])

    dcg_c = dcg.copy()
    dcg_c = dcg_c.rename(index={'bankruptcy_Disc_Cum_Gain.csv':'ESA Polish Bankruptcy',
           'financial_distress_Disc_Cum_Gain.csv':'Financial Distress',
           'german_credit_Disc_Cum_Gain.csv':'UCI German Credit Card IMB',
           'credit_approval_Disc_Cum_Gain.csv':'UCI Credit Approval IMB',
           'bank_marketing_Disc_Cum_Gain.csv':'UCI Bank Marketing IMB',
           'telco_customer_churn_Disc_Cum_Gain.csv':'IBM Telco Customer Churn ',
           'credit_card_approval_Disc_Cum_Gain.csv':'ETL Credit Card Approval',
           'vub_credit_scoring_Disc_Cum_Gain.csv':'VUB Credit Scoring' ,
           'kdd_Disc_Cum_Gain.csv':'KDD Cup 1998',
           'car_insurance_fraud_label_Disc_Cum_Gain.csv':'(Priv.) Car Insurance Fraud ',
           'apata_credit_card_fraud_label_Disc_Cum_Gain.csv':'(Priv.) APATA Credit Card Fraud ',
           'financial_statement_fraud_label_Disc_Cum_Gain.csv':'JAR Financial Statement Fraud',
           'customer_trans_fraud_detection_label_Disc_Cum_Gain.csv':'IEEE-CIS Cus. Trans. Fraud',
           'synth_mobile_money_fraud_label_Disc_Cum_Gain.csv':'Synt. Mobile Trans. Fraud',
           'synth_credit_card_fraud_label_Disc_Cum_Gain.csv':'Synthetic Credit Card Fraud'})

    dcg_c_1 = dcg_c[['Logit(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)', 'XGBoost(1, 1, 0, 0)_smote',
        'XGBoost(1, 1, 0, 0)_adasyn', 'XGBoost(1, 1, 0, 0)_rose']]
    
    dcg_c_1 = dcg_c_1.applymap('{:,.3f}'.format)

    latex = dcg_c_1.to_latex()
    for i in range(14):
        latex = add_hline(latex=latex, index=2 * i + 1)
    print(latex)

    dcg_c_2 = dcg_c[['XGBoost(1, 1, 1, 1)_BROOD_ROT',
        'XGBoost(1, 1, 1, 1)_BROOD_KNN','XGBoost(1, 1, 1, 1)_BROOD_ROT_rose', 'XGBoost(1, 1, 1, 1)_BROOD_KNN_rose']]
    
    dcg_c_2 = dcg_c_2.applymap('{:,.3f}'.format)

    latex = dcg_c_2.to_latex()
    for i in range(14):
        latex = add_hline(latex=latex, index=2 * i + 1)
    print(latex)


name_roc_auc = 'all_ROC_AUC.xlsx'
name_ap = 'all_AP.xlsx'
name_dcg= 'all_Disc_Cum_Gain.xlsx'

roc_auc.to_excel((path / name_roc_auc).resolve())
ap.to_excel((path / name_ap).resolve())
dcg.to_excel((path / name_dcg).resolve())
