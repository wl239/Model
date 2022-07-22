import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def standard(df):
    # Standardization
    mu = np.mean(df, axis=0)
    sigma = np.std(df, axis=0)
    return (df - mu) / sigma


def create_pca_variables(df):
    y = df['CONC']
    x = df.drop(['CONC'], axis=1, inplace=False)
    return x, y


def build_principal_component_analysis(df, n_component):
    pca_model = PCA(n_components=n_component)
    data_after_pca = pca_model.fit_transform(df)
    pca_df = pd.DataFrame(data=data_after_pca)
    t_value_array = pca_model.explained_variance_ratio_
    t_value = np.sum(t_value_array)
    return pca_model, pca_df, t_value
