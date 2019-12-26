# -*- encoding: utf-8 -*-
"""
@Author  : shan hai
@Time    : 2019/12/24 4:30 PM
@Email   : shanhai3000@gmail.com
"""
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 3), index=list('abcde'), columns=['one', 'two', 'three'])  # 随机产生5行3列的数据
# df2 = df.copy()  # type:  pd.DataFrame
df.dropna(axis='columns', how='any')
df.loc['a', 'one'] = None
missing_val_cols = [col for col in df.columns if df[col].isnull().any()]
# missing_val_cols = missing_val_cols[missing_val_cols > 0]
print(missing_val_cols)
df.drop(axis=1, columns=missing_val_cols, inplace=True)
print(df)

df2 = pd.DataFrame(np.random.randn(5, 3), columns=['one', 'two', 'three'])  # 随机产生5行3列的数据
df2.loc[0, 'one'] = None

col2 = df2.isnull().sum()
print(type(col2))

# 如果使用dropna， subset对应的参数和axis相关，
df2.dropna(axis=1, subset=[1, 2, 3], inplace=True)#有None记录在row = 0， 定义subset = [1,2,3]的row，就没有drop的效果
print(df2)
