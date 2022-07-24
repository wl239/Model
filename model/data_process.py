import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from copy import deepcopy


def clean_data(df):
    keep_columns = ['BUY_SELL', 'INSTRUMENT_TYPE', 'VOLUME', 'VWAP', 'IMS', 'TRADES', 'CONC', 'TAM_VOLUME', 'TAM_VWAP',
                    'TAM_IMS', 'TAM_TRADES', 'TAM_CONC', 'HFA_VOLUME', 'HFA_VWAP', 'HFA_IMS', 'HFA_TRADES', 'HFA_CONC',
                    'RTW_VOLUME', 'RTW_VWAP', 'RTW_IMS', 'RTW_TRADES', 'RTW_CONC']
    df = df[keep_columns]

    # Remove NAN
    df_after_nan = df.dropna(axis=0, how='any').reset_index(drop=True)
    # print(df_after_nan)

    temp_data = deepcopy(df_after_nan)

    # Remove Volume = 0
    for i in range(len(df_after_nan)):
        if df_after_nan.iloc[i]['VOLUME'] == 0:
            temp_data.drop(index=i)

    # log Volume
    temp_data['VOLUME'] = temp_data['VOLUME'].apply(np.log)
    # print(temp_data)
    print(temp_data)

    # Dummy Variables
    BSDummy = pd.get_dummies(df_after_nan['BUY_SELL'], prefix="BuySellType", prefix_sep="_", drop_first=True)
    ITDummy = pd.get_dummies(df_after_nan['INSTRUMENT_TYPE'], prefix="InstrumentType", prefix_sep="_",
                             drop_first=True)
    data_after_dummy = pd.concat((df_after_nan, BSDummy, ITDummy), axis=1)
    data_after_dummy.drop(['BUY_SELL', 'INSTRUMENT_TYPE'], axis=1, inplace=True)
    columns = ['VOLUME', 'VWAP', 'IMS', 'TRADES', 'CONC', 'TAM_VOLUME', 'TAM_VWAP', 'TAM_IMS', 'TAM_TRADES',
               'TAM_CONC', 'HFA_VOLUME', 'HFA_VWAP', 'HFA_IMS', 'HFA_TRADES', 'HFA_CONC', 'RTW_VOLUME', 'RTW_VWAP',
               'RTW_IMS', 'RTW_TRADES', 'RTW_CONC', 'BuySellType_S', 'InstrumentType_F']
    data_after_dummy.columns = columns
    return data_after_dummy


def get_correlation_matrix(df, filepath):
    df_corr = df.corr()
    sb.set(rc={'figure.figsize': (20, 20)})
    sb.heatmap(df_corr, center=0, annot=True)
    plt.savefig(filepath)
    plt.close()


