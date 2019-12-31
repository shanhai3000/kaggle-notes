# -*- encoding: utf-8 -*-
"""
@Author  : shan hai
@Time    : 2019/12/25 7:30 PM
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
'''
4行，代表4个样本；4列 代表4个特征
第一列包含0、1、2三个unique值，经oneHotEncoder编码后转化为[[1,0,0],[0,1,0],[0,0,1]]
第二列包含2、3两个unique值，经oneHotEncoder编码后转化为[[1,0], [0,1]]
不难看出，包含n个unique值的特征，经oneHotEncoder编码后，转化为∑a, a=[0,...,1,0,0],其中a[i]=1,(i=0,1,2,3,...,n-1)
'''
data_set = pd.DataFrame([
    [0, 8],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 2],
    [1, 9],
    [2, 9],
])
s = encoder.fit_transform(data_set)
"""
用矩阵思维理解,如果矩阵的行为r，列为c,每一列unique的数量为ci,则:
1.oneHotEncoder后形成的新矩阵规模为r*(∑(1,c)ci)个entries
2.增加了 r*[(∑(1,c)ci) - c ]个entries
output
[[1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1.]]
 """
# 2+9 = 11 列
# 如果是全唯一的set， r行 c 列 r*c -
# OH_col_X_train = pd.DataFrame(one_hot_encoder.fit_transform(X_train[object_cols]))

print(s.toarray())
encoded_vector = encoder.transform([[0, 8], [0, 1]]).toarray()
# Encoded vector = [[1. 0. 0. 0. 1. 0. 0. 0. 1. 1. 0.]]
