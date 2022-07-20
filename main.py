# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from model.supervised_learning import *


if __name__ == '__main__':
    raw_data = pd.read_excel('ik_fake.xlsx', index_col=0)
    raw_data.drop(raw_data[raw_data['VOLUME'] <= 0.0].index, inplace=True)
    raw_data['lnVOLUME'] = np.log(raw_data['VOLUME'])
    y1 = np.where(raw_data['BUY_SELL'] == 'B', 0, 1)
    y2 = np.where(raw_data['INSTRUMENT_TYPE'] == 'E', 0, 1)
    X = raw_data[['lnVOLUME', 'VWAP']].values

    raw_x_train, raw_x_test, raw_y_train, raw_y_test = create_train_test_dataset(X, y1)
    model_list = ['LR', 'RF']
    result = comparison_classification(model_list, raw_x_train, raw_x_test, raw_y_train, raw_y_test)

    df_result = pd.DataFrame(result, columns=['Method', 'Accuracy'])
    print(df_result)
