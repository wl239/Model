import numpy as np
import pandas as pd
from model import *

raw_data = pd.read_csv('cleaned_original_ik_data.csv', index_col=0)
raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
raw_data = raw_data.dropna(axis=0, how='any').reset_index(drop=True)
raw_data['lnVOLUME'] = np.log(raw_data['VOLUME'])
raw_data.drop(['VOLUME'], axis=1, inplace=True)
normalized_raw_data = norm_standard(raw_data)

# y1 -- regression
y1 = normalized_raw_data.iloc[:, 3].values
X1 = normalized_raw_data.drop(3, axis=1, inplace=False).values

# y2 and y3 -- classification
y2 = raw_data['S'].values
X2 = norm_standard(raw_data.drop(['CONC', 'E', 'F'], axis=1, inplace=False)).values
y3 = raw_data['F'].values
X3 = norm_standard(raw_data.drop(['CONC', 'B', 'S'], axis=1, inplace=False)).values

# regression
raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test = create_train_test_dataset(X1, y1)

# classification
raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test = create_train_test_dataset(X2, y2)
raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test = create_train_test_dataset(X3, y3)

model_list = ['LR', 'RF', 'SVM']

# regression
result1 = comparison_regression(model_list, raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test)

# classification
result2 = comparison_classification(model_list, raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test)

result3 = comparison_classification(model_list, raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test)

df_result1 = pd.DataFrame(result1, columns=['Method', 'MSE'])
df_result2 = pd.DataFrame(result2, columns=['Method', 'Accuracy'])
df_result3 = pd.DataFrame(result3, columns=['Method', 'Accuracy'])

print(df_result1)
print(df_result2)
print(df_result3)

