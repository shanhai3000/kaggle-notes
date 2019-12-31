# -*- encoding: utf-8 -*-
"""
@Author  : shan hai
@Time    : 2019/12/27 2:58 AM
"""

# extreme gradient boosting
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

X = pd.read_csv("./data-set/train.csv")
X.dropna(subset=['SalePrice'], axis=0, inplace=True)
y = X.SalePrice
X.drop(columns=['SalePrice'], axis=1, inplace=True)
numerical_cols = list(X.select_dtypes(exclude=['object']).columns)
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


def getPipeline(preprocessor, n_estimators=200):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=n_estimators))
    ])
    return pipeline


def score(n_estimators, preprocessor, X, y):
    my_pipeline = getPipeline(preprocessor, n_estimators)
    score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return score.mean()


# visual
# scrore = cross_val_score(my_pipeline, X, y)
# plt.plot(list(results.keys()), list(results.values()))
# plt.show()
#


results = {}

for i in range(1, 10):
    results[i * 50 + 300] = score(i * 50 + 100, preprocessor, X, y)

best_n_estimators = min(results, key=lambda i: results[i])
my_pipeline = getPipeline(preprocessor, best_n_estimators)
clf = my_pipeline.fit(X, y)
test_data = pd.read_csv("./data-set/test.csv")
pred = clf.predict(test_data)
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': pred})
output.to_csv("./submission.csv", index=False)


