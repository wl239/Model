import matplotlib.pyplot as plt
import dataframe_image as dfi
import numpy as np
from math import floor, ceil
import seaborn as sb

from sklearn.metrics import confusion_matrix


def save_report_as_png(ols_report, filename):
    filepath = 'image/' + str(filename) + '/' + str(filename) + '_ols_report.png'
    print(filepath)
    plt.rc('figure', figsize=(10, 12))
    # plt.text(0.01, 0.05, str(print_model), {'fontsize': 12})   # old approach
    plt.text(0.01, 0.05, str(ols_report), {'fontsize': 12}, fontproperties='monospace')
      # approach improved by OP -> monospace!
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_report_as_text(ols_report, filename):
    filepath = 'image/' + str(filename) + '/' + str(filename) + '_ols_report.txt'
    print(filepath)
    with open(filepath, 'w') as fh:
        fh.write(ols_report.as_text())


def save_knn_output_as_png(print_model, name):
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


def plot_scatter(x, y, model_name, file_name):
    file_path = 'image/' + str(file_name) + '/' + str(file_name) + '_scatter_' + str(model_name) + '.png'
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    min_x = min_y = min(floor(min(x)), floor(min(y)))
    max_x = max_y = max(ceil(max(x)), ceil(max(y)))
    ideal_x = np.linspace(min_x, max_x, len(x))
    ideal_y = np.linspace(min_y, max_y, len(y))
    plt.scatter(x, y)
    plt.plot(ideal_x, ideal_y, 'red')
    plt.title(str(model_name) + ': Actual Y vs Predicted Y')
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.savefig(file_path)
    plt.close()


def plot_confusion_matrix(x, y, model_name, file_name, class_type):
    file_path = 'image/' + str(file_name) + '/' + str(file_name) + '_' \
                + str(class_type) + '_class_confusion_' + str(model_name) + '.png'
    # Plot the Confusion Matrix
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    if class_type == 'buy':
        title = "Classification (Buy/Sell Label)"
    else:  # Equity Class
        title = "Classification (Equity/Fixed Income Label)"
    plt.title(title)
    sb.heatmap(confusion_matrix(x, y), annot=True, fmt=".0f", annot_kws={"size": 12})
    plt.xlabel("True Y")
    plt.ylabel("Predicted Y")
    plt.savefig(file_path)
    plt.close()
