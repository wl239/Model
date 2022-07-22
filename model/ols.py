import statsmodels.api as sm


def create_ols_variables(df):
    y = df['CONC'].values
    x = df.drop(['CONC'], axis=1, inplace=False).values
    return x, y


def build_ordinary_least_square(x, y):
    X = sm.add_constant(x)
    ols_model = sm.OLS(y, X).fit()
    pred = ols_model.predict(X)
    ols_report = ols_model.summary()
    return pred, ols_report

