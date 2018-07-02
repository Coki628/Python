# -*- coding: utf-8 -*-

from collections import Counter 

w = list(input())

# 各文字ごとに数を集計
str_dict = Counter(w)

for key, value in str_dict.items():
    if value % 2 != 0:
        print("No")
        exit()
print("Yes")