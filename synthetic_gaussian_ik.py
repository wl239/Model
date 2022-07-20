import numpy as np
from model.supervised_learning import *
from model.ols import *

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

synthetic_data['lnVOLUME'] = np.log(synthetic_data['VOLUME'])
synthetic_data['lnTAM_VOLUME'] = np.log(synthetic_data['TAM_VOLUME'])
synthetic_data['lnHFA_VOLUME'] = np.log(synthetic_data['HFA_VOLUME'])
synthetic_data['lnRTW_VOLUME'] = np.log(synthetic_data['RTW_VOLUME'])
synthetic_data.drop(['VOLUME', 'TAM_VOLUME', 'HFA_VOLUME', 'RTW_VOLUME'], axis=1, inplace=True)

# Dummy Variables
BSColumnDummy = pd.get_dummies(synthetic_data['BUY_SELL'])
ITColumnDummy = pd.get_dummies(synthetic_data['INSTRUMENT_TYPE'])
data_after_dummy = pd.concat((synthetic_data, BSColumnDummy, ITColumnDummy), axis=1)
data_after_dummy.drop(['BUY_SELL', 'INSTRUMENT_TYPE'], axis=1, inplace=True)

# ols
y = data_after_dummy['CONC'].values
x = data_after_dummy.drop(['CONC'], axis=1, inplace=False).values
pred, print_ols = build_ordinary_least_square(x, y)
save_text_as_png(print_ols, 'gaussian_ik')

# Machine Learning
# Data need to be normalized and standardized when we do machine learning
normalized_synthetic_data = norm_standard(data_after_dummy)

# Regression: y is concentration rate, x -- other variables
y1 = normalized_synthetic_data.iloc[:, 3].values
X1 = normalized_synthetic_data.drop(3, axis=1, inplace=False).values

# Classification
# Label y1 -- Buy Sell Type: 'S' and 'B'. 'S' is 1, 'B' is 0
# Label y2 -- Instrument Type: 'E' and 'F. 'F' is 1. 'E' is 0
y2 = data_after_dummy['S'].values
X2 = norm_standard(data_after_dummy.drop(['CONC', 'E', 'F', 'B', 'S'], axis=1, inplace=False)).values
y3 = data_after_dummy['F'].values
X3 = norm_standard(data_after_dummy.drop(['CONC', 'B', 'S', 'E', 'F'], axis=1, inplace=False)).values

# Split Dataset: Train and Test
x_train_1, x_test_1, y1_train, y1_test = create_train_test_dataset(X1, y1)
x_train_2, x_test_2, y2_train, y2_test = create_train_test_dataset(X2, y2)
x_train_3, x_test_3, y3_train, y3_test = create_train_test_dataset(X3, y3)

knn_classifier, accuracy, report = build_knn_classifier(x_train_2, x_test_2, y2_train, y2_test)
save_text_as_png(report, 'gaussian_ik_y2_knn')

knn_classifier2, accuracy2, report2 = build_knn_classifier(x_train_3, x_test_3, y3_train, y3_test)
save_text_as_png(report2, 'gaussian_ik_y3_knn')

model_list = ['LR', 'RF', 'SVM', 'KNN']
result1 = comparison_regression(model_list, x_train_1, x_test_1, y1_train, y1_test)
result2 = comparison_classification(model_list, x_train_2, x_test_2, y2_train, y2_test)
result3 = comparison_classification(model_list, x_train_3, x_test_3, y3_train, y3_test)

df_result1 = pd.DataFrame(result1, columns=['Method', 'Synthetic Data - MSE']).set_index(['Method'])
df_result2 = pd.DataFrame(result2, columns=['Method', 'Synthetic Data - Accuracy 1']).set_index(['Method'])
df_result3 = pd.DataFrame(result3, columns=['Method', 'Synthetic Data - Accuracy 2']).set_index(['Method'])

df_all_result = pd.concat([df_result1, df_result2, df_result3], axis=1)

print(df_result1)
print(df_result2)
print(df_result3)
print(df_all_result)

save_dataframe_as_png(df_result1, "synthetic_ik_regression")
save_dataframe_as_png(df_result2, "synthetic_ik_buy_sell_class")
save_dataframe_as_png(df_result3, "synthetic_ik_E_F_class")
save_dataframe_as_png(df_all_result, "synthetic_ik_all")

