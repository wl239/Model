import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from model.data_visualize import *


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


def standard(df):
    # Standardization
    mu = np.mean(df, axis=0)
    sigma = np.std(df, axis=0)
    return (df - mu) / sigma


def create_regression_variables(df):
    y = df.iloc[:, 1].values
    x = df.drop(1, axis=1, inplace=False).values  # 'CONC' column number: 1
    return x, y


def create_classification_variables(df, label_name):
    y = df[label_name].values
    x = norm_standard(df.drop(['BuySellType_S', 'InstrumentType_F'], axis=1, inplace=False)).values
    return x, y


def create_train_test_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  # 70% training 30% test
    return x_train, x_test, y_train, y_test


def build_linear_regression(x, y, x_train, x_test, y_train, y_test):
    lr_model = linear_model.Ridge(fit_intercept=True, copy_X=True)
    scores = cross_val_score(lr_model, x, y, cv=10)
    lr_model.fit(x_train, y_train)
    y_pred = lr_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return lr_model, y_pred, r2


def build_random_forest_regression(x, y, x_train, x_test, y_train, y_test):
    rf_model = RandomForestRegressor()
    scores = cross_val_score(rf_model, x, y, cv=10)
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return rf_model, y_pred, r2


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
def build_svm_regression(x, y, x_train, x_test, y_train, y_test):
    SVM_regression = SVR(kernel='rbf')
    scores = cross_val_score(SVM_regression, x, y, cv=10)
    SVM_regression.fit(x_train, y_train)
    y_pred = SVM_regression.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return SVM_regression, y_pred, r2


def build_knn_regression(x, y, x_train, x_test, y_train, y_test):
    knn_regression = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    scores = cross_val_score(knn_regression, x, y, cv=10)
    knn_regression.fit(x_train, y_train)
    y_pred = knn_regression.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return knn_regression, y_pred, r2


def comparison_regression(model_list, file_name, x_all, y_all, x_train, x_test, y_train, y_test):
    result = []
    for x in model_list:
        if x == 'RF':
            rf_regression, y_pred, score = build_random_forest_regression(x_all, y_all, x_train, x_test, y_train, y_test)
        elif x == 'LR':
            linear_regression, y_pred, score = build_linear_regression(x_all, y_all, x_train, x_test, y_train, y_test)
        elif x == 'SVM':
            svm_regression, y_pred, score = build_svm_regression(x_all, y_all, x_train, x_test, y_train, y_test)
        elif x == 'KNN':
            knn_regression, y_pred, score = build_knn_regression(x_all, y_all, x_train, x_test, y_train, y_test)
        else:
            continue
        result.append([x, score])
        plot_scatter(y_test, y_pred, x, file_name)
    return result


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
def build_logistic_regression(x, y, x_train, x_test, y_train, y_test):
    lr_model = linear_model.LogisticRegression(penalty='l2', dual=False, fit_intercept=True, n_jobs=-1)
    # penalty: default -- l2
    scores = cross_val_score(lr_model, x, y, cv=10)
    lr_model.fit(x_train, y_train)
    y_pred = lr_model.predict(x_test)
    return lr_model, y_pred, scores.mean()


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
def build_random_forest_classifier(x, y, x_train, x_test, y_train, y_test):
    """
    random_grid = {'bootstrap': [True, False],
                   'max_depth': [10, 20, 40, 80, 100, None],
                   'max_features': ['auto', 'sqrt', None],
                   'min_samples_leaf': [1, 2, 5, 10],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [50, 100, 200, 400, 600, 800, 1000, 1500]}

    rfc = RandomForestClassifier()
    randomforest_classifier = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
                                                 cv=3, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=0)
    randomforest_classifier.fit(x_train, y_train)
    rf_model = randomforest_classifier.best_estimator_
    """

    rf_model = RandomForestClassifier(n_estimators=100, max_features=2)
    scores = cross_val_score(rf_model, x, y, cv=10)
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    return rf_model, y_pred, scores.mean()


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
def build_svm_classifier(x, y, x_train, x_test, y_train, y_test):
    # Choose parameter and kernel
    SVM_regression = SVC(kernel='rbf')
    scores = cross_val_score(SVM_regression, x, y, cv=10)
    SVM_regression.fit(x_train, y_train)
    y_pred = SVM_regression.predict(x_test)
    return SVM_regression, y_pred, scores.mean()


# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
def build_knn_classifier(x, y, x_train, x_test, y_train, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    scores = cross_val_score(knn_model, x, y, cv=10)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test)
    report = classification_report(y_test, y_pred)
    return knn_model, y_pred, scores.mean(), report


def comparison_classification(model_list, file_name, class_type, x_all, y_all, x_train, x_test, y_train, y_test):
    result = []
    for x in model_list:
        if x == 'RF':
            rf_classifier, y_pred, accuracy = build_random_forest_classifier(x_all, y_all, x_train, x_test, y_train, y_test)
        elif x == 'LR':
            logistic_model, y_pred, accuracy = build_logistic_regression(x_all, y_all, x_train, x_test, y_train, y_test)
        elif x == 'SVM':
            svm_classifier, y_pred, accuracy = build_svm_classifier(x_all, y_all, x_train, x_test, y_train, y_test)
        elif x == 'KNN':
            knn_classifier, y_pred, accuracy, report = build_knn_classifier(x_all, y_all, x_train, x_test, y_train, y_test)
        else:
            continue
        plot_confusion_matrix(y_test, y_pred, x, file_name, class_type)
        result.append([x, accuracy])
    return result


