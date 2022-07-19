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


def build_linear_regression(x_train, x_test, y_train, y_test):
    lr_model = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
    lr_model.fit(x_train, y_train)
    y_pred = lr_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return lr_model, mse


def build_random_forest_regression(x_train, x_test, y_train, y_test):
    rf_model = RandomForestRegressor()
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return rf_model, mse


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
def build_svm_regression(x_train, x_test, y_train, y_test):
    SVM_regression = SVR(kernel='rbf')
    SVM_regression.fit(x_train, y_train)
    y_pred = SVM_regression.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return SVM_regression, mse


def comparison_regression(model_list, x_train, x_test, y_train, y_test):
    result = []
    for x in model_list:
        if x == 'RF':
            rf_regression, mse = build_random_forest_regression(x_train, x_test, y_train, y_test)
        elif x == 'LR':
            linear_regression, mse = build_linear_regression(x_train, x_test, y_train, y_test)
        elif x == 'SVM':
            svm_regression, mse = build_svm_regression(x_train, x_test, y_train, y_test)
        else:
            continue
        result.append([x, mse])
    return result


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
def build_logistic_regression(x_train, x_test, y_train, y_test):
    lr_model = linear_model.LogisticRegression(penalty='l2', dual=False, fit_intercept=True, n_jobs=-1)
    # penalty: default -- l2
    lr_model.fit(x_train, y_train)
    y_pred = lr_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return lr_model, accuracy


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
def build_random_forest_classifier(x_train, x_test, y_train, y_test):
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
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return rf_model, accuracy


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
def build_svm_classifier(x_train, x_test, y_train, y_test):
    # Choose parameter and kernel
    SVM_regression = SVC(kernel='rbf')
    SVM_regression.fit(x_train, y_train)
    y_pred = SVM_regression.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return SVM_regression, accuracy


def comparison_classification(model_list, x_train, x_test, y_train, y_test):
    result = []
    for x in model_list:
        if x == 'RF':
            rf_classifier, accuracy = build_random_forest_classifier(x_train, x_test, y_train, y_test)
        elif x == 'LR':
            logistic_model, accuracy = build_logistic_regression(x_train, x_test, y_train, y_test)
        elif x == 'SVM':
            svm_regression, accuracy = build_svm_classifier(x_train, x_test, y_train, y_test)
        else:
            continue
        result.append([x, accuracy])
    return result

