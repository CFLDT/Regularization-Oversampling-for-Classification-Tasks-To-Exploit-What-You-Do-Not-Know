import pandas as pd
from pathlib import Path


def performance_tables(name, roc_auc_df,
                       ap_df, positive_found_df, disc_cum_gain_df, dict):
    base_path = Path(__file__).parent

    dict['roc_auc'].append(roc_auc_df.mean(axis=0))
    dict['ap'].append(ap_df.mean(axis=0))
    #dict['positive_found'].append(positive_found_df.mean(axis=0))
    dict['disc_cum_gain'].append(disc_cum_gain_df.mean(axis=0))

    df_concat = pd.concat(dict['roc_auc'], axis=1)
    if isinstance(df_concat, pd.Series):
        df_concat = df_concat.to_frame()
    df_means_auc = df_concat.mean(axis=1)

    df_concat = pd.concat(dict['ap'], axis=1)
    if isinstance(df_concat, pd.Series):
        df_concat = df_concat.to_frame()
    df_means_ap = df_concat.mean(axis=1)

    # df_concat = pd.concat(dict['positive_found'], axis=1)
    # if isinstance(df_concat, pd.Series):
    #     df_concat = df_concat.to_frame()
    # df_means_positive_found = df_concat.mean(axis=1)

    df_concat = pd.concat(dict['disc_cum_gain'], axis=1)
    if isinstance(df_concat, pd.Series):
        df_concat = df_concat.to_frame()
    df_means_disc_cum_gain = df_concat.mean(axis=1)

    names = name + '_ROC_AUC' + '.csv'
    df_means_auc.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    names = name + '_ROC_AUC_cv' + '.csv'
    roc_auc_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_AP' + '.csv'
    df_means_ap.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    names = name + '_AP_cv' + '.csv'
    ap_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_Disc_Cum_Gain' + '.csv'
    df_means_disc_cum_gain.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    names = name + '_Disc_Cum_Gain_cv' + '.csv'
    disc_cum_gain_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())
