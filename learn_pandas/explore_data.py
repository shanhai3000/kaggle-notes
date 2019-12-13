import pandas as pd

"""
the most important part is the DataFrame
"""
reader = pd.read_csv("./Prostate_Cancer.csv")

print(reader.values[:, 3:4].sum())
print(reader.columns)
print(reader.values[:, 3:4].min())

