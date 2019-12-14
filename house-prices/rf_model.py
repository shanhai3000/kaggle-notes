import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data_set = pd.read_csv("./data-set/train.csv")
data_columns = [i for i in data_set.columns]
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data_set[features]
y = data_set.SalePrice
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
rf = RandomForestRegressor(random_state=0, n_estimators=10)
rf.fit(X, y)

test_data = pd.read_csv("./data-set/test.csv")
test_pred = rf.predict(test_data[features])

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_pred})
output.to_csv('./submission.csv', index=False)

