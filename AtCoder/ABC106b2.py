# -*- coding: utf-8 -*-

N = int(input())

# 約数の個数
def divisor_num(n):
    res = 0
    for i in range(1, n+1):
        if n % i == 0:
            res += 1
    return res

ans = 0
for i in range(N, -1, -1):
    if i % 2 != 0:
        if divisor_num(i) == 8:
            ans += 1
print(ans)