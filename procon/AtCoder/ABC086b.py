# -*- coding: utf-8 -*-

import math

str_list = input().split()
num1 = int(str_list[0] + str_list[1])

# 連結して作った数値の平方根を求める
num2 = math.sqrt(num1)
# 整数型で返っていれば、平方数だとみなせる
if num2.is_integer():
    print("Yes")
else:
    print("No")