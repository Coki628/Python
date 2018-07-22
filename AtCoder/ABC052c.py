# -*- coding: utf-8 -*-

import math

x = int(input())

# 階乗
a = math.factorial(x)

aN = []
i = 1
# かけ合わせる数の左右逆はやる意味ない
while i <= a//i:
    # 約数として使えるものを追加していく
    if a % i == 0:
        aN.append(i)
        aN.append(a//i)
    i += 1
print(len(aN) % (10 ** 9 + 7))