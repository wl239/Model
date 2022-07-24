from model.supervised_learning import *
from model.ols import *
from model.unsupervised_learning import *
from model.data_process import *
from model.data_visualize import *

# start = time.time(）
# end_time = time.time()
# duration = end_time - start_time

df = pd.read_excel('anon_cleaned_ik.xlsx')      # <================================================================================ CHANGE !!!
cleaned_df = clean_data(df)

file_path = './image/anon_ik/anon_ik'   # <================================================================================ CHANGE !!!
get_correlation_matrix(cleaned_df, file_path + '_corr.png')

# Observe Correlation Matrix
df_without_collinear = cleaned_df

# OLS Estimators
x_ols, y_ols = create_ols_variables(df_without_collinear)  # Create OLS variables
pred, ols_report = build_ordinary_least_square(x_ols, y_ols)  # OLS regression
save_report_as_png(ols_report, 'anon_ik')  # Save ols summary as a picture   # <================================================================================ CHANGE !!!
save_report_as_text(ols_report, 'anon_ik')  # Save ols summary as a text     # <================================================================================ CHANGE !!!


drop_columns = ['RTW_IMS', 'TAM_IMS', 'TRADES', 'IMS', 'RTW_TRADES', 'VWAP', 'TAM_VWAP']
df_without_collinear = cleaned_df.drop(drop_columns, axis=1, inplace=False)

# Supervised Learning

# Create x and y variables
x_reg, y_reg = create_regression_variables(df_without_collinear)  # y is concentration rate, x -- other variables
x_buy_class, y_buy_class = create_classification_variables(df_without_collinear,
                                                           'BuySellType_S')  # Label: 'B' is 0, 'S' is 1
x_equity_class, y_equity_class = create_classification_variables(df_without_collinear,
                                                                 'InstrumentType_F') # Label: 'E' is 0, 'F' is 1

# Split Dataset: Train and Test
x_reg_train, x_reg_test, y_reg_train, y_reg_test = create_train_test_dataset(x_reg, y_reg)
x_buy_train, x_buy_test, y_buy_train, y_buy_test = create_train_test_dataset(x_buy_class, y_buy_class)
x_equity_train, x_equity_test, y_equity_train, y_equity_test = create_train_test_dataset(x_equity_class,
                                                                                         y_equity_class)

# knn_classifier, accuracy, report = build_knn_classifier(raw_x_train_2, raw_x_test_2, raw_y2_train, raw_y2_test)
# save_knn_output_as_png(report, 'raw_ik_y2_knn')
# knn_classifier2, accuracy2, report = build_knn_classifier(raw_x_train_3, raw_x_test_3, raw_y3_train, raw_y3_test)
# save_knn_output_as_png(report, 'raw_ik_y3_knn')

# model_list = ['LR']
model_list = ['LR', 'RF', 'SVM', 'KNN']
file_name = 'anon_ik'                                   # <================================================================================ CHANGE !!!
result1 = comparison_regression(model_list, file_name, x_reg, y_reg, x_reg_train, x_reg_test, y_reg_train, y_reg_test)
result2 = comparison_classification(model_list, file_name, 'buy', x_buy_class, y_buy_class, x_buy_train, x_buy_test, y_buy_train, y_buy_test)
result3 = comparison_classification(model_list, file_name, 'equity', x_equity_class, y_equity_class, x_equity_train, x_equity_test, y_equity_train, y_equity_test)

df_result1 = pd.DataFrame(result1, columns=['Method', 'Goodness of fit']).set_index(['Method'])
df_result2 = pd.DataFrame(result2, columns=['Method', 'Accuracy']).set_index(['Method'])
df_result3 = pd.DataFrame(result3, columns=['Method', 'Accuracy']).set_index(['Method'])
df_all_result = pd.concat([df_result1, df_result2, df_result3], axis=1)

# MSE, R square, Accuracy Table
save_dataframe_as_png(df_result1, file_path + '_regression')
save_dataframe_as_png(df_result2, file_path + '_buy_sell_class')
save_dataframe_as_png(df_result3, file_path + '_E_F_class')


# Unsupervised Learning -- PCA
x_pca, y_pca = create_pca_variables(cleaned_df)

n_component = 5
pca_model, pca_df, t_value = build_principal_component_analysis(standard(x_pca), n_component)
print("T value: %f" % t_value)  # when n = 5, 0.664694, explained variance ratio
pca_df.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5']
pca_df.to_excel('pca_anon_ik.xlsx')         # <=================================================================================== CHANGE !!!

# colors = ['red', 'black', 'orange', 'green', 'blue']
# plt.figure()
# for i in [0, 1, 2, 3, 4]:
# plt.scatter(principal_df[y == i, 0], principal_df[y == i, 1], alpha=.7, c=colors[i], label=principal_df.columns[i])
# plt.legend()
# plt.title('PCA of Raw Dataset')
# plt.show()
# plt.close()

x_pca = pca_df.values
y_pca = y_pca.values
x_pca_train, x_pca_test, y_pca_train, y_pca_test = create_train_test_dataset(x_pca, y_pca)
result4 = comparison_regression(model_list, 'pca_anon_ik', x_pca, y_pca, x_pca_train, x_pca_test, y_pca_train, y_pca_test)  # <=================================================================================== CHANGE !!!

df_result4 = pd.DataFrame(result4, columns=['Method', 'Goodness of fit']).set_index(['Method'])
save_dataframe_as_png(df_result4, "image/pca_anon_ik/pca_anon_ik_regression")     # <=================================================================================== CHANGE !!!
