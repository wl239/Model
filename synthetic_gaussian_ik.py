import numpy as np
from sklearn.metrics import confusion_matrix

from model.supervised_learning import *
from model.ols import *
import seaborn as sns
import matplotlib.pyplot as mp
from model.unsupervised_learning import *

# Synthetic Data (Gaussian) IK Data
df = pd.read_excel('ik_gaussian.xlsx', index_col=0)
df.sort_values(by='ISIN', inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(df.columns[[0]], axis=1, inplace=True)
synthetic_data = df[['BUY_SELL', 'INSTRUMENT_TYPE', 'VOLUME', 'VWAP', 'IMS', 'TRADES',
                     'CONC', 'TAM_VOLUME', 'TAM_VWAP', 'TAM_IMS', 'TAM_TRADES', 'TAM_CONC',
                     'HFA_VOLUME', 'HFA_VWAP', 'HFA_IMS', 'HFA_TRADES', 'HFA_CONC', 'RTW_VOLUME',
                     'RTW_VWAP', 'RTW_IMS', 'RTW_TRADES', 'RTW_CONC']]

# synthetic_data.replace([np.inf, -np.inf], np.nan, inplace=True)
synthetic_data = synthetic_data.dropna(axis=0, how='any').reset_index(drop=True)
synthetic_data['lnVOLUME'] = np.log(synthetic_data['VOLUME'])
synthetic_data['lnTAM_VOLUME'] = np.log(synthetic_data['TAM_VOLUME'])
synthetic_data['lnHFA_VOLUME'] = np.log(synthetic_data['HFA_VOLUME'])
synthetic_data['lnRTW_VOLUME'] = np.log(synthetic_data['RTW_VOLUME'])
synthetic_data.drop(['VOLUME', 'TAM_VOLUME', 'HFA_VOLUME', 'RTW_VOLUME'], axis=1, inplace=True)

# Dummy Variables
BSColumnDummy = pd.get_dummies(synthetic_data['BUY_SELL'], prefix="BuySellType", prefix_sep="_", drop_first=True)
ITColumnDummy = pd.get_dummies(synthetic_data['INSTRUMENT_TYPE'], prefix="InstrumentType", prefix_sep="_",
                               drop_first=True)
data_after_dummy = pd.concat((synthetic_data, BSColumnDummy, ITColumnDummy), axis=1)
data_after_dummy.drop(['BUY_SELL', 'INSTRUMENT_TYPE'], axis=1, inplace=True)

# Correlation Matrix
df_corr = data_after_dummy.corr()
sns.set(rc={'figure.figsize': (20, 20)})
sns.heatmap(df_corr, center=0, annot=True)
mp.show()

data_without_collinear = data_after_dummy.drop(['TRADES', 'TAM_VWAP', 'TAM_TRADES', 'RTW_CONC',
                                                'HFA_TRADES', 'HFA_IMS', 'HFA_VWAP', 'lnTAM_VOLUME'], axis=1,
                                               inplace=False)
# data_without_collinear = data_after_dummy.drop(['RTW_IMS', 'TAM_IMS', 'TRADES', 'HFA_IMS', 'TAM_TRADES', 'TAM_VWAP'],
#                                                axis=1, inplace=False)

# ols
y = data_without_collinear['CONC'].values
x = data_without_collinear.drop(['CONC'], axis=1, inplace=False).values
pred, print_ols = build_ordinary_least_square(x, y)
save_text_as_png(print_ols, 'gaussian_ik')

# Machine Learning
# Data need to be normalized and standardized when we do machine learning
normalized_synthetic_data = norm_standard(data_without_collinear)

# Regression: y is concentration rate, x -- other variables
y1 = normalized_synthetic_data.iloc[:, 2].values
X1 = normalized_synthetic_data.drop(2, axis=1, inplace=False).values

# Classification
# Label y1 -- Buy Sell Type: 'S' and 'B'. 'S' is 1, 'B' is 0
# Label y2 -- Instrument Type: 'E' and 'F. 'F' is 1. 'E' is 0
y2 = data_without_collinear['BuySellType_S'].values
X2 = norm_standard(data_without_collinear.drop(['BuySellType_S'], axis=1, inplace=False)).values
y3 = data_without_collinear['InstrumentType_F'].values
X3 = norm_standard(data_without_collinear.drop(['InstrumentType_F'], axis=1, inplace=False)).values

# Split Dataset: Train and Test
x_train_1, x_test_1, y1_train, y1_test = create_train_test_dataset(X1, y1)
x_train_2, x_test_2, y2_train, y2_test = create_train_test_dataset(X2, y2)
x_train_3, x_test_3, y3_train, y3_test = create_train_test_dataset(X3, y3)

knn_classifier, accuracy, report = build_knn_classifier(x_train_2, x_test_2, y2_train, y2_test)
save_knn_output_as_png(report, 'gaussian_ik_y2_knn')

knn_classifier2, accuracy2, report2 = build_knn_classifier(x_train_3, x_test_3, y3_train, y3_test)
save_knn_output_as_png(report2, 'gaussian_ik_y3_knn')

model_list = ['LR', 'RF', 'SVM', 'KNN']
result1 = comparison_regression(model_list, x_train_1, x_test_1, y1_train, y1_test)
result2 = comparison_classification(model_list, x_train_2, x_test_2, y2_train, y2_test)
result3 = comparison_classification(model_list, x_train_3, x_test_3, y3_train, y3_test)

df_result1 = pd.DataFrame(result1, columns=['Method', 'Synthetic Data - MSE', 'Synthetic Data - R^2']).set_index(
    ['Method'])
df_result2 = pd.DataFrame(result2, columns=['Method', 'Synthetic Data - Accuracy 1']).set_index(['Method'])
df_result3 = pd.DataFrame(result3, columns=['Method', 'Synthetic Data - Accuracy 2']).set_index(['Method'])

df_all_result = pd.concat([df_result1, df_result2, df_result3], axis=1)

print(df_result1)
print(df_result2)
print(df_result3)
print(df_all_result)

svm_model, mse, r_2 = build_svm_regression(x_train_1, x_test_1, y1_train, y1_test)
y_pred = svm_model.predict(x_test_1)

plt.rcParams['figure.figsize'] = (6.0, 6.0)
ideal_x = np.linspace(-2.5, 2.2, len(y1_test))
ideal_y = np.linspace(-2.5, 2.2, len(y_pred))
plt.scatter(y1_test, y_pred)
plt.plot(ideal_x, ideal_y, 'red')
plt.title('Synthetic Data: Actual Y vs Predicted Y')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.show()
plt.close()


rf_class, rf_accuracy1 = build_random_forest_classifier(x_train_2, x_test_2, y2_train, y2_test)
y_pred = rf_class.predict(x_test_2)
# Plot the Confusion Matrix
plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.title("Raw Data: Classification (B/S Label)")
sns.heatmap(confusion_matrix(y2_test, y_pred), annot=True, fmt=".0f", annot_kws={"size": 12})
plt.show()
plt.close()

svm_class, rf_accuracy2 = build_random_forest_classifier(x_train_3, x_test_3, y3_train, y3_test)
y_pred = rf_class.predict(x_test_3)
# Plot the Confusion Matrix
plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.title("Raw Data: Classification (E/F Label)")
sns.heatmap(confusion_matrix(y3_test, y_pred), annot=True, fmt=".0f", annot_kws={"size": 12})
plt.show()
plt.close()

save_dataframe_as_png(df_result1, "gaussian_ik_regression")
save_dataframe_as_png(df_result2, "gaussian_ik_buy_sell_class")
save_dataframe_as_png(df_result3, "gaussian_ik_E_F_class")
save_dataframe_as_png(df_all_result, "gaussian_ik_all")

# Unsupervised Learning
# PCA
print("===================================")
print("***********************************")
print(data_after_dummy.info())
print(data_after_dummy)
pca_x = data_after_dummy.drop(['CONC'], axis=1, inplace=False)
pca_y = data_after_dummy['CONC']
principalComponents, principal_df, t_value = build_principal_component_analysis(standard(data_after_dummy), 5)
print("T value: %f" % t_value)  # T value: 0.711452

X_pca = principal_df.values
y_pca = pca_y.values
x_pca_train, x_pca_test, y_pca_train, y_pca_test = create_train_test_dataset(X_pca, y_pca)
result4 = comparison_regression(model_list, x_pca_train, x_pca_test, y_pca_train, y_pca_test)

df_result4 = pd.DataFrame(result4, columns=['Method', 'MSE', 'R Square']).set_index(['Method'])
print(df_result4)

save_dataframe_as_png(df_result4, "pca_gaussian_ik_regression")
