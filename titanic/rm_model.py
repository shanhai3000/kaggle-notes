# -*- encoding: utf-8 -*-
"""
@Author  : shan hai
@Time    : 2019/12/27 4:59 PM
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score

X = pd.read_csv("./data/train.csv")
X.dropna(subset=['Survived'], axis=0, inplace=True)
test_data = pd.read_csv("./data/test.csv")

test_data_name = [col for col in test_data.Name]
X_data_name = [col for col in X.Name]
for name in test_data_name:
    if name in X_data_name:
        print(name)


women = X.loc[X['Sex'] == 'female']['Survived']
rate_women = sum(women) / len(women)
man = X[X['Sex'] == 'male']['Survived']
rate_man = sum(man) / len(man)
#
# y = X.Survived
#
# col_nunique = X.nunique()
# nunique_map = dict(zip(col_nunique.keys(), col_nunique.values))
# nunique_map = sorted(nunique_map, key=lambda i: nunique_map[i])
#
# feature_cols = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(X[feature_cols])
# X_test = pd.get_dummies(test_data[feature_cols])
# model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=1)
# model.fit(X, y)
#
# predictions = model.predict(X_test)
# predictions = map(lambda i: 0 if i < 0.5 else 1, predictions)
# print(predictions)
# output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
# output.to_csv("./my_submission.csv", index=False)
#
#
