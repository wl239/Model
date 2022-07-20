import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, SVR

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def standard(df):
    # Standardization
    mu = np.mean(df, axis=0)
    sigma = np.std(df, axis=0)
    return (df - mu) / sigma


def create_train_test_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # 70% training 30% test
    return x_train, x_test, y_train, y_test




