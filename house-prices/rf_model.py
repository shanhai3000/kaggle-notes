# -*- encoding: utf-8 -*-
"""
@Author  : shan hai
@Time    : 2019/12/14 10:17 PM
@Email   : shanhai3000@gmail.com
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def predict_mae(model, X_test, y_test):
    pre_res = model.predict(X_test)
    return mean_absolute_error(pre_res, y_test)


# read the data
def get_data_set():
    data_set = pd.read_csv("./data-set/train.csv")  # type: pd.DataFrame
    c_w_m = [col for col in data_set.columns if data_set[col].isnull().any()]
    X_full = data_set.copy()
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)  # drop not available rows
    # x相当于横轴， y 纵轴，类似直角坐标系，但axis=1时对应整行data向y轴方向移动，axis=1时为列沿着横轴方向移动
    # 或简单理解为axis=0影响符合条件下的整行，axis=1影响符合条件下的整列
    # X_full.drop(['SalePrice'], axis=1, inplace=True) #axis=1: remove名字为'SalePrice'的整个列
    numerical_features = data_set.select_dtypes(exclude=['object']).columns


def train_and_valid():
    data_set = pd.read_csv("./data-set/train.csv")  # type: pd.DataFrame
    numerical_features = data_set.select_dtypes(exclude=['object']).columns
    # use only numerical predictors, filter only numerical columns
    X_numerical = data_set[numerical_features]
    # remove target column :SalePrice
    y = X_numerical.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X_numerical, y, random_state=0, train_size=0.8, test_size=0.2)
    rf = RandomForestRegressor(random_state=0, n_estimators=100)
    # # before fit, the dataset stil has lots NA need to be imputed
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_train.columns = numerical_features
    imputed_X_train.drop(['SalePrice'], axis=1, inplace=True)

    rf.fit(imputed_X_train, y_train)
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_test))
    imputed_X_valid.columns = numerical_features

    label_X_valid = imputed_X_valid.copy()

    label_X_valid.drop(['SalePrice'], axis=1, inplace=True)
    test_pred = rf.predict(label_X_valid)

    return mean_absolute_error(y_test, test_pred)


def train():
    data_set = pd.read_csv("./data-set/train.csv")  # type: pd.DataFrame
    numerical_features = data_set.select_dtypes(
        exclude=['object']).columns  # use only numerical predictors, filter only numerical columns
    X_numerical = data_set[numerical_features]
    # remove target column :SalePrice
    y = X_numerical.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X_numerical, y, random_state=1, train_size=0.8, test_size=0.2)
    rf = RandomForestRegressor(random_state=0, n_estimators=100)
    # # before fit, the dataset stil has lots NA need to be imputed
    my_imputor = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputor.fit_transform(X_train))
    imputed_X_train.columns = numerical_features
    imputed_X_train.drop(['SalePrice'], axis=1, inplace=True)
    rf.fit(imputed_X_train, y_train)

    return rf


# predict is after model fit
def predict(model, data_set):
    columns = data_set.columns
    # # before fit, the dataset stil has lots NA need to be imputed
    my_imputor = SimpleImputer()
    imputed_X_test = pd.DataFrame(my_imputor.transform(data_set))
    imputed_X_test.columns = columns
    return model.predict(imputed_X_test)


def test():
    data_set = pd.read_csv("./data-set/train.csv")  # type: pd.DataFrame
    numerical_features = data_set.select_dtypes(exclude=['object']).columns
    test_data = pd.read_csv("./data-set/test.csv")
    X_test = test_data.select_dtypes(exclude=['object'])
    test_columns = X_test.columns
    rf = train()
    my_imputor = SimpleImputer(strategy='median')
    X_numerical = data_set[numerical_features].copy()
    X_numerical.drop(axis=1, columns=['SalePrice'], inplace=True)
    my_imputor.fit_transform(X_numerical)

    imputed_X_test = pd.DataFrame(my_imputor.transform(X_test))
    imputed_X_test.columns = test_columns
    test_pred = rf.predict(imputed_X_test)
    # mae = mean_absolute_error('y_truth', 'y_predict') #validate by mean absolute error
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_pred})
    output.to_csv('./submission.csv', index=False)


def train_and_valid_with_object_values():
    X = pd.read_csv("./data-set/train.csv", index_col='Id')
    all_columns = X.columns

    # prepare, Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)

    # drop columns with missing values
    cols_with_missing_val = [col for col in all_columns if X[col].isnull().any()]
    X.drop(columns=cols_with_missing_val, axis=1, inplace=True)
    # prepare end  # test数据集也需要drop cols_with_missing_val
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # 1 step: 拆分数据集为 train/test两部分
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)
    rf = RandomForestRegressor(random_state=0, n_estimators=100)

    numerical_features = X.select_dtypes(exclude=['object']).columns

    # 2 step: object列数据分类
    object_cols = X.select_dtypes(include=['object']).columns
    # object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    # print(object_nunique)#[5, 2, 4, 4, 2, 5, 3, 25, 9, 6, 5, 8, 6, 7, 15, 16, 4, 5, 6, 6, 5, 2, 4, 6, 3, 9, 6]

    # Columns that can be safely label encoded
    good_label_cols = [col for col in object_cols if
                       set(X_train[col]) == set(X_test[col])]  # 列中出现的分类相同，即没有变异类
    bad_label_cols = list(set(object_cols) - (set(good_label_cols)))
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_test = X_test.drop(bad_label_cols, axis=1)

    label_encoder = LabelEncoder()
    labeled_X_train = label_X_train.copy()
    labeled_X_test = label_X_test.copy()
    for col in good_label_cols:
        labeled_X_train[col] = label_encoder.fit_transform(label_X_train[col])
        labeled_X_test[col] = label_encoder.transform(label_X_test[col])

    # impute
    my_imputer = SimpleImputer(strategy='median')
    imputed_labeled_X_train = pd.DataFrame(my_imputer.fit_transform(labeled_X_train))
    imputed_labeled_X_test = pd.DataFrame(my_imputer.transform(labeled_X_test))
    imputed_labeled_X_train.columns = labeled_X_train.columns
    imputed_labeled_X_test.columns = labeled_X_test.columns

    # label_encoder = LabelEncoder()
    # one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # OH_col_X_train = pd.DataFrame(one_hot_encoder.fit_transform(X_train[object_cols]))
    # OH_col_X_test = pd.DataFrame(one_hot_encoder.transform(X_test[object_cols]))
    # todo OH_col_X_train.columns
    # OH_col_X_train.index = X_train.index
    # OH_col_X_test.index = X_test.index

    # 3 step: impute missing values
    numerical_X_train = X_train[numerical_features].copy()

    # OH_X_train = pd.concat([numerical_X_train, OH_col_X_train], axis=1)
    rf.fit(imputed_labeled_X_train, y_train)

    # OH_X_test = pd.concat([imputed_X_valid, OH_col_X_test], axis=1)
    # label_X_valid = imputed_X_valid.copy()
    # for col in object_cols:
    #     label_X_valid[col] = label_encoder.fit_transform(imputed_X_train[col])

    # label_X_valid.drop(['SalePrice'], axis=1, inplace=True)
    test_pred = rf.predict(imputed_labeled_X_test)

    return mean_absolute_error(y_test, test_pred)


def encode_and_impute_train():
    X = pd.read_csv("./data-set/train.csv")
    test_data = pd.read_csv("./data-set/test.csv")  # type: pd.DataFrame
    # step 1, drop nan target rows
    X = X.dropna(subset=['SalePrice'], axis=0)
    y = X.SalePrice

    object_X_cols = X.select_dtypes(include=['object']).columns
    object_test_cols = test_data.select_dtypes(include=['object']).columns
    object_cols = list(set(object_X_cols).union(object_test_cols))

    # step 2, drop missing value from object value columns(missing either in X or test)
    X_missing_val_cols = [col for col in object_cols if X[col].isnull().any()]
    test_missing_val_cols = [col for col in object_cols if test_data[col].isnull().any()]
    missing_val_cols = list(set(X_missing_val_cols).union(set(test_missing_val_cols)))
    X = X.drop(columns=missing_val_cols, axis=1).drop(columns=['SalePrice'], axis=1)
    test_data = test_data.drop(columns=missing_val_cols, axis=1)
    object_cols = list(set(object_cols) - set(missing_val_cols))

    # step 3, encode
    # step 3-1: label encoding
    good_label_cols = [col for col in object_cols if set(X[col]) == set(test_data[col])]
    bad_label_cols = list(set(object_cols) - set(good_label_cols))
    label_encoder = LabelEncoder()
    for col in good_label_cols:
        X[col] = label_encoder.fit_transform(X[col])
        test_data[col] = label_encoder.transform(test_data[col])
    X = X.drop(columns=bad_label_cols, axis=1)
    test_data = test_data.drop(columns=bad_label_cols, axis=1)
    object_cols = list(set(object_cols) - set(bad_label_cols))
    # step 3-2, encode: low cardinality--oneHotEncode; high cardinality--labelEncode
    object_nunique = list(map(lambda col: X[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))
    # step 3-2-1: low cardinality--oneHotEncode
    low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_X_col = pd.DataFrame(OH_encoder.fit_transform(X[low_cardinality_cols]))
    OH_test_col = pd.DataFrame(OH_encoder.transform(test_data[low_cardinality_cols]))
    OH_X_col.index = X.index
    OH_test_col.index = test_data.index

    # step 4, impute the missing value
    my_imputer = SimpleImputer(strategy='median')
    numerical_X = X.select_dtypes(exclude=['object'])
    imputed_X = pd.DataFrame(my_imputer.fit_transform(numerical_X))
    imputed_X.columns = numerical_X.columns
    numerical_test = test_data.select_dtypes(exclude=['object'])
    imputed_test = pd.DataFrame(my_imputer.transform(numerical_test))
    imputed_test.columns = numerical_test.columns
    OH_imputed_X = pd.concat([imputed_X, OH_X_col], axis=1)
    OH_imputed_test = pd.concat([imputed_test, OH_test_col], axis=1)

    # step 5 fit the model
    model = RandomForestRegressor(random_state=0, n_estimators=100)
    model.fit(OH_imputed_X, y)

    test_pred = model.predict(OH_imputed_test)
    # mae = mean_absolute_error('y_truth', 'y_predict') #validate by mean absolute error
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_pred})
    output.to_csv('./submission.csv', index=False)
    ## 16037.12085

# print(train_and_valid_with_object_values())  # output = 18237.842397260272
# label_encoder 17575.291883561644
encode_and_impute_train()
