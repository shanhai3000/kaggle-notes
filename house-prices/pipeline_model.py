# -*- encoding: utf-8 -*-
"""
@Author  : shan hai
@Time    : 2019/12/26 3:16 PM
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


def pipeline_with_data_separate():
    X = pd.read_csv("./data-set/train.csv")
    test_data = pd.read_csv("./data-set/test.csv")
    X = X.dropna(subset=['SalePrice'], axis=0)
    y = X.SalePrice
    X.drop(columns=['SalePrice'], axis=1, inplace=True)
    object_cols = X.select_dtypes(include=['object']).columns
    missing_val_cols = [col for col in object_cols if X[col].isnull().any()]
    X = X.drop(columns=missing_val_cols, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
    # step 1. define preprocessing
    # preprocessing for numerical data
    numerical_cols = list(X_train.select_dtypes(exclude=['object']).columns)
    numerical_transformer = SimpleImputer(strategy='median')
    # preprocessing for categorical data
    # todo categorical cols should be low_cardinality and object_val cols
    # categorical_cols = X_train.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in X_train.select_dtypes(include=['object']).columns
                        if X_train[col].nunique() < 10]
    categorical_transformer = Pipeline(steps=[
        # ('label', LabelEncoder()),
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # step 2. define the model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    # step 3. create and evaluate the pipeline
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    all_cols = numerical_cols + categorical_cols
    X_train = X_train[all_cols].copy()
    X_test = X_test[all_cols].copy()
    test_data = test_data[all_cols].copy()
    # preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)
    # preprocessing of validation data, get prediction
    preds = my_pipeline.predict(X_test)

    # evaluate the model
    score = mean_absolute_error(y_test, preds)
    print(score)


def pipeline_with_cross_validation():
    X = pd.read_csv("./data-set/train.csv")
    X.dropna(subset=['SalePrice'], axis=0, inplace=True)
    y = X.SalePrice
    X.drop(columns=['SalePrice'], axis=1, inplace=True)
    object_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = list(X.select_dtypes(exclude=['object']).columns)
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_cols = [col for col in X.select_dtypes(include=['object'])
                        if X[col].nunique() < 10]
    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    all_columns = numerical_cols + categorical_cols
    X = X[all_columns]

    model = RandomForestRegressor(n_estimators=140, random_state=0)
    my_pipeline = Pipeline(steps=[
        ('preprocesser', preprocessor),
        ('model', model)
    ])
    test_data = pd.read_csv("./data-set/test.csv")
    clf = my_pipeline.fit(X, y)
    pred = clf.predict(test_data)
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': pred})
    output.to_csv('./submission.csv', index=False)

    # score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

    # results = {}
    # for i in range(1, 9):
    #     results[i * 10 + 100] = int(score(i * 10 + 100, preprocessor, X, y))

    # plt.plot(list(results.keys()), list(results.values()))
    # plt.show()


def score(n_estimators, preprocessor, X, y):
    model = RandomForestRegressor(n_estimators=n_estimators)
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return score.mean()


pipeline_with_cross_validation()
