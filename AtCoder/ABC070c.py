# -*- coding: utf-8 -*-

from functools import reduce

N = int(input())
TN = [int(input()) for i in range(N)]

# ようは最小公倍数を出せればいいと思う
def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def lcm_base(x, y):
    return (x * y) // gcd(x, y)
def lcm_list(numbers):
    return reduce(lcm_base, numbers, 1)

print(lcm_list(TN))