import copy

import numpy as np
from model.supervised_learning import *
from model.ols import *
import matplotlib.pyplot as mp
import seaborn as sns

# Raw Data
df = pd.read_excel('cleaned_ik.xlsx', index_col=0)
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df = df.dropna(axis=0, how='any').reset_index(drop=True)
raw_data = df[['BUY_SELL', 'INSTRUMENT_TYPE', 'VOLUME', 'VWAP', 'IMS', 'TRADES',
               'CONC', 'TAM_VOLUME', 'TAM_VWAP', 'TAM_IMS', 'TAM_TRADES', 'TAM_CONC',
               'HFA_VOLUME', 'HFA_VWAP', 'HFA_IMS', 'HFA_TRADES', 'HFA_CONC', 'RTW_VOLUME',
               'RTW_VWAP', 'RTW_IMS', 'RTW_TRADES', 'RTW_CONC']]
raw_data = raw_data.dropna(axis=0, how='any').reset_index(drop=True)
raw_data['lnVOLUME'] = np.log(raw_data['VOLUME'])
raw_data.drop(['VOLUME'], axis=1, inplace=True)

# Dummy Variables
# drop_first must be True to reduce multicollinearity
BSDummy = pd.get_dummies(raw_data['BUY_SELL'], prefix="BuySellType", prefix_sep="_", drop_first=True)
ITDummy = pd.get_dummies(raw_data['INSTRUMENT_TYPE'], prefix="InstrumentType", prefix_sep="_",
                         drop_first=True)
data_after_dummy = pd.concat((raw_data, BSDummy, ITDummy), axis=1)
data_after_dummy.drop(['BUY_SELL', 'INSTRUMENT_TYPE'], axis=1, inplace=True)

# Correlation Matrix
df_corr = data_after_dummy.corr()
sns.set(rc={'figure.figsize': (20, 20)})
sns.heatmap(df_corr, center=0, annot=True)
mp.show()

data_after_dummy.drop(['RTW_IMS', 'TAM_IMS', 'TRADES', 'HFA_IMS', 'TAM_TRADES'], axis=1, inplace=True)

# ols
y = data_after_dummy['CONC'].values
x = data_after_dummy.drop(['CONC'], axis=1, inplace=False).values
pred, print_ols = build_ordinary_least_square(x, y)
save_text_as_png(print_ols, 'raw_ik')

# Machine Learning
# Data need to be standardized
normalized_raw_data = norm_standard(data_after_dummy)

# Regression: y is concentration rate, x -- other variables
y1 = normalized_raw_data.iloc[:, 2].values
X1 = normalized_raw_data.drop(2, axis=1, inplace=False).values  # column number: 3

# Classification
# Label y1 -- Buy Sell Type: 'S' and 'B'. 'S' is 1, 'B' is 0
# Label y2 -- Instrument Type: 'E' and 'F. 'F' is 1. 'E' is 0
y2 = data_after_dummy['BuySellType_S'].values
X2 = norm_standard(data_after_dummy.drop(['BuySellType_S'], axis=1, inplace=False)).values
y3 = data_after_dummy['InstrumentType_F'].values
X3 = norm_standard(data_after_dummy.drop(['InstrumentType_F'], axis=1, inplace=False)).values

# Split Dataset: Train and Test
raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test = create_train_test_dataset(X1, y1)
raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test = create_train_test_dataset(X2, y2)
raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test = create_train_test_dataset(X3, y3)

knn_classifier, accuracy, report = build_knn_classifier(raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test)
save_knn_output_as_png(report, 'raw_ik_y2_knn')

knn_classifier2, accuracy2, report = build_knn_classifier(raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test)
save_knn_output_as_png(report, 'raw_ik_y3_knn')

model_list = ['LR', 'RF', 'SVM', 'KNN']
result1 = comparison_regression(model_list, raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test)
result2 = comparison_classification(model_list, raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test)
result3 = comparison_classification(model_list, raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test)

df_result1 = pd.DataFrame(result1, columns=['Method', 'Raw Data - MSE', 'Raw Data - R^2']).set_index(['Method'])
df_result2 = pd.DataFrame(result2, columns=['Method', 'Raw Data - Accuracy 1']).set_index(['Method'])
df_result3 = pd.DataFrame(result3, columns=['Method', 'Raw Data - Accuracy 2']).set_index(['Method'])

df_all_result = pd.concat([df_result1, df_result2, df_result3], axis=1)

print(df_result1)
print(df_result2)
print(df_result3)
print(df_all_result)

save_dataframe_as_png(df_result1, "raw_ik_regression")
save_dataframe_as_png(df_result2, "raw_ik_buy_sell_class")
save_dataframe_as_png(df_result3, "raw_ik_E_F_class")
save_dataframe_as_png(df_all_result, "raw_ik_all")
