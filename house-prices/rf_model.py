import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


def predict_mae(model, X_test, y_test):
    pre_res = model.predict(X_test)
    return mean_absolute_error(pre_res, y_test)


# read the data
data_set = pd.read_csv("./data-set/train.csv")  # type: pd.DataFrame
c_w_m = [col for col in data_set.columns if data_set[col].isnull().any()]
X_full = data_set.copy()
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)  # drop not availible rows
# (x和y是一对)
# X_full.drop(['SalePrice'], axis=1, inplace=True)
numerical_features = data_set.select_dtypes(exclude=['object']).columns


def train_and_valid(data_set):
    # use only numerical predictors, filter only numerical columns
    X_numerical = data_set[numerical_features]
    # remove target column :SalePrice
    y = X_numerical.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X_numerical, y, random_state=0, train_size=0.8, test_size=0.2)
    rf = RandomForestRegressor(random_state=0, n_estimators=100)
    # # before fit, the dataset stil has lots NA need to be imputed
    my_imputor = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputor.fit_transform(X_train))
    imputed_X_train.columns = numerical_features
    imputed_X_train.drop(['SalePrice'], axis=1, inplace=True)
    rf.fit(imputed_X_train, y_train)
    imputed_X_valid = pd.DataFrame(my_imputor.fit_transform(X_test))
    imputed_X_valid.columns = numerical_features
    imputed_X_valid.drop(['SalePrice'], axis=1, inplace=True)
    test_pred = rf.predict(imputed_X_valid)

    return mean_absolute_error(X_test.SalePrice, test_pred)


def train(data_set):
    # use only numerical predictors, filter only numerical columns
    X_numerical = data_set[numerical_features]
    # remove target column :SalePrice
    y = X_numerical.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X_numerical, y, random_state=1, train_size=0.8, test_size=0.2)
    print(X_train.head())
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
    test_data = pd.read_csv("./data-set/test.csv")
    X_test = test_data.select_dtypes(exclude=['object'])
    test_columns = X_test.columns
    rf = train(data_set)
    my_imputor = SimpleImputer()
    imputed_X_test = pd.DataFrame(my_imputor.fit_transform(X_test))
    imputed_X_test.columns = test_columns
    test_pred = rf.predict(imputed_X_test)
    # mae = mean_absolute_error('y_truth', 'y_predict') #validate by mean absolute error
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_pred})
    output.to_csv('./submission.csv', index=False)

test()