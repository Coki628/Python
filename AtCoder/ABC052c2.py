# -*- coding: utf-8 -*-

import math

x = int(input())

# 階乗
a = math.factorial(x)

i = 1
cnt = 0
# かけ合わせる数の左右逆はやる意味ない
while i <= a//i:
    # 約数として使えるものを追加していく
    if a % i == 0:
        cnt += 2
    i += 1
print(cnt % (10 ** 9 + 7))