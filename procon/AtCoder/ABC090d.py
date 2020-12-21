# -*- coding: utf-8 -*-

"""
調査用
"""

import matplotlib.pyplot as plt 
import pandas as pd

table = [[-1] * 100 for i in range(100)]
for i in range(1, 100):
    for j in range(100):
        table[i][j] = j % i

df = pd.DataFrame(table)
print(df)
