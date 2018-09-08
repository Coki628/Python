# -*- coding: utf-8 -*-

from functools import reduce

N,X = map(int, input().split())
xN = list(map(int, input().split()))

# 最大公約数
def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def gcd_list(numbers):
    return reduce(gcd, numbers)

xN2 = [0] * N
for i in range(N):
    xN2[i] = abs(xN[i] - X)

print(gcd_list(xN2))