import pandas as pd

"""
the most important part is the DataFrame
"""
reader = pd.read_csv("./Prostate_Cancer.csv")

# print(reader.values[:, 3:4].sum())
# print(reader.columns)
# print(reader.values[:, 3:4].min())

df = pd.DataFrame([[1, 2, 3, 4], [2, 2, 2, 2], [3, 3, 3, 3]], columns=["col1", "col2", "col3", "col4"])

# print(df)
"""
df.mean()默认计算矩阵每列的均值，
"""

"""
axis = 0 为横轴（缺省），沿着行垂直向下，axis = 1 为纵轴，沿着横轴向右


"""
# df = df.drop("col2", axis=1)
# print(df)

# features = [i for i in reader.columns]
# print(type(features))
# l = [i for i in features]
# print(l)

"""
pandas: df.loc[1:2]
numpy: arr[1:2]
"""
print(df.iloc[:1, 1:])

a = [11.1,11.2,11.3]
print(a.index(min(a)))