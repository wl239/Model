import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def standard(df):
    # Standardization
    mu = np.mean(df, axis=0)
    sigma = np.std(df, axis=0)
    return (df - mu) / sigma


def build_principal_component_analysis(df, n_component):
    pca = PCA(n_components=n_component)
    principalComponents = pca.fit_transform(df)
    principal_df = pd.DataFrame(data=principalComponents)
    t_value_array = pca.explained_variance_ratio_
    t_value = np.sum(t_value_array)
    return principalComponents, principal_df, t_value
