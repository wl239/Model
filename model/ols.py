import statsmodels.api as sm
import matplotlib.pyplot as plt
import dataframe_image as dfi


def build_ordinary_least_square(x, y):
    X = sm.add_constant(x)
    ols_model = sm.OLS(y, X).fit()
    pred = ols_model.predict(X)
    return pred, ols_model.summary()


def save_text_as_png(print_model, name):
    plt.rc('figure', figsize=(10, 4))
    # plt.text(0.01, 0.05, str(print_model), {'fontsize': 12})   # old approach
    plt.text(0.01, 0.05, str(print_model), {'fontsize': 12}, fontproperties='monospace')
      # approach improved by OP -> monospace!
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(str(name) + '_output.png')
    plt.close()


def save_dataframe_as_png(df, name):
    df_styled = df.style.background_gradient()
    dfi.export(df_styled, str(name) + ".png")
