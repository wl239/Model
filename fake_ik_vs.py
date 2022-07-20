import numpy as np
from model.supervised_learning import *

# Classification
fake_data = pd.read_excel('ik_fake.xlsx', index_col=0)
fake_data.drop(fake_data[fake_data['VOLUME'] <= 0.0].index, inplace=True)
fake_data['lnVOLUME'] = np.log(fake_data['VOLUME'])
y1 = np.where(fake_data['BUY_SELL'] == 'B', 0, 1)
y2 = np.where(fake_data['INSTRUMENT_TYPE'] == 'E', 0, 1)
X = norm_standard(fake_data[['lnVOLUME', 'VWAP']]).values

raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test = create_train_test_dataset(X, y1)
raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test = create_train_test_dataset(X, y2)
model_list = ['LR', 'RF']
result1 = comparison_classification(model_list, raw_x_train_1, raw_x_test_1, raw_y1_train, raw_y1_test)
result2 = comparison_classification(model_list, raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test)

df_result1 = pd.DataFrame(result1, columns=['Method', 'Accuracy'])
df_result2 = pd.DataFrame(result2, columns=['Method', 'Accuracy'])

print(df_result1)
print(df_result2)
