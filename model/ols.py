import statsmodels.api as sm
import matplotlib.pyplot as plt


def build_ordinary_least_square(x, y):
    X = sm.add_constant(x)
    ols_model = sm.OLS(y, X).fit()
    pred = ols_model.predict(X)
    return pred, ols_model.summary()


def save_as_png(print_model, name):
    plt.rc('figure', figsize=(12, 14))
    # plt.text(0.01, 0.05, str(print_model), {'fontsize': 12})   # old approach
    plt.text(0.01, 0.05, str(print_model), {'fontsize': 12}, fontproperties='monospace')
      # approach improved by OP -> monospace!
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(str(name) + '_output.png')
    plt.close()
