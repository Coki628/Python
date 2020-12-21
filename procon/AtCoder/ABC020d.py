# -*- coding: utf-8 -*-

"""
・だいぶ前にやった部分点解法
"""

from functools import reduce
 
# 最大公約数と最小公倍数
def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def lcm_base(x, y):
    return (x * y) // gcd(x, y)
def lcm_list(numbers):
    # reduce(使う関数, 足し合わせるリスト, 初期値)
    return reduce(lcm_base, numbers, initial=1)

N,K = map(int, input().split())

ans = 0
for i in range(1,N+1):
    ans += lcm_base(i, K) % (10**9+7)
print(ans % (10**9+7))
