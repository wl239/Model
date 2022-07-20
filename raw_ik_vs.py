import numpy as np
from model.supervised_learning import *
from model.ols import *

# Raw Data
raw_data = pd.read_csv('cleaned_original_ik_data.csv', index_col=0)
raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
raw_data = raw_data.dropna(axis=0, how='any').reset_index(drop=True)
raw_data['lnVOLUME'] = np.log(raw_data['VOLUME'])
raw_data.drop(['VOLUME'], axis=1, inplace=True)

# ols
y = raw_data['CONC'].values
x = raw_data.drop(['CONC'], axis=1, inplace=False).values
pred, print_ols = build_ordinary_least_square(x, y)
save_text_as_png(print_ols, 'raw_ik')

# Machine Learning
# Data need to be normalized and standardized
normalized_raw_data = norm_standard(raw_data)

# Regression: y is concentration rate, x -- other variables
y1 = normalized_raw_data.iloc[:, 3].values
X1 = normalized_raw_data.drop(3, axis=1, inplace=False).values

# Classification
# Label y1 -- Buy Sell Type: 'S' and 'B'. 'S' is 1, 'B' is 0
# Label y2 -- Instrument Type: 'E' and 'F. 'F' is 1. 'E' is 0
y2 = raw_data['S'].values
X2 = norm_standard(raw_data.drop(['CONC', 'E', 'F', 'B', 'S'], axis=1, inplace=False)).values
y3 = raw_data['F'].values
X3 = norm_standard(raw_data.drop(['CONC', 'B', 'S', 'E', 'F'], axis=1, inplace=False)).values

# Split Dataset: Train and Test
raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test = create_train_test_dataset(X1, y1)
raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test = create_train_test_dataset(X2, y2)
raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test = create_train_test_dataset(X3, y3)

knn_classifier, accuracy, report = build_knn_classifier(raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test)
save_text_as_png(report, 'raw_ik_y2_knn')

knn_classifier, accuracy, report = build_knn_classifier(raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test)
save_text_as_png(report, 'raw_ik_y3_knn')

model_list = ['LR', 'RF', 'SVM', 'KNN']
result1 = comparison_regression(model_list, raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test)
result2 = comparison_classification(model_list, raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test)
result3 = comparison_classification(model_list, raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test)

df_result1 = pd.DataFrame(result1, columns=['Method', 'Raw Data - MSE']).set_index(['Method'])
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

