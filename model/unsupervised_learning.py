import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, SVR

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def norm_standard(df):
    # Normalization
    #   _range = np.max(df) - np.min(df)
    #   return (df - np.min(df))/_range
    norm_scaler = MinMaxScaler()
    data1 = norm_scaler.fit_transform(df)

    # Standardization
    # mu = np.mean(df, axis=0)
    # sigma = np.std(df, axis=0)
    # return (df - mu) / sigma
    scaler = StandardScaler()
    data2 = pd.DataFrame(scaler.fit_transform(data1))

    return data2


def create_train_test_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # 70% training 30% test
    return x_train, x_test, y_train, y_test




