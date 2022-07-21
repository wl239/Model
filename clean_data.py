import pandas as pd
import numpy as np


def clean_data(df):
    # Synthetic Data (Gaussian) IK Data
    df = pd.read_excel('ik_gaussian.xlsx', index_col=0)
    df.sort_values(by='ISIN', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    data = df[['BUY_SELL', 'INSTRUMENT_TYPE', 'VOLUME', 'VWAP', 'IMS', 'TRADES',
               'CONC', 'TAM_VOLUME', 'TAM_VWAP', 'TAM_IMS', 'TAM_TRADES', 'TAM_CONC',
               'HFA_VOLUME', 'HFA_VWAP', 'HFA_IMS', 'HFA_TRADES', 'HFA_CONC', 'RTW_VOLUME',
               'RTW_VWAP', 'RTW_IMS', 'RTW_TRADES', 'RTW_CONC']]

    synthetic_data = data.dropna(axis=0, how='any').reset_index(drop=True)
    synthetic_data['lnVOLUME'] = np.log(synthetic_data['VOLUME'])
    synthetic_data['lnTAM_VOLUME'] = np.log(synthetic_data['TAM_VOLUME'])
    synthetic_data['lnHFA_VOLUME'] = np.log(synthetic_data['HFA_VOLUME'])
    synthetic_data['lnRTW_VOLUME'] = np.log(synthetic_data['RTW_VOLUME'])
    synthetic_data.drop(['VOLUME', 'TAM_VOLUME', 'HFA_VOLUME', 'RTW_VOLUME'], axis=1, inplace=True)

    # Dummy Variables
    BSColumnDummy = pd.get_dummies(synthetic_data['BUY_SELL'], prefix="BuySellType", prefix_sep="_", drop_first=True)
    ITColumnDummy = pd.get_dummies(synthetic_data['INSTRUMENT_TYPE'], prefix="InstrumentType", prefix_sep="_",
                                   drop_first=True)
    data_after_dummy = pd.concat((synthetic_data, BSColumnDummy, ITColumnDummy), axis=1)
    data_after_dummy.drop(['BUY_SELL', 'INSTRUMENT_TYPE'], axis=1, inplace=True)
    return data_after_dummy
