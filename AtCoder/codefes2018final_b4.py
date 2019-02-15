# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-festival-2018-final/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2018/11/18/184215
・そもそも問題理解できてなかった。
・M配列に書いてある条件を全て満たすような確率ってことか…。
・多分ダメなんだけど、階乗テーブル作ってやってみる。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import copy, deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def prod(nums): return reduce(mul, nums, 1)
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def init_factorial(MAX: int) -> list:
    # 階乗テーブル
    factorial = [1] * (MAX)
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i
    return factorial

def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = factorial[r] * factorial[n-r]
    return numerator // denominator

N, M = MAP()
rM = LIST()

factorial = init_factorial(100001)

N2 = N
p = 1
for r in rM:
    p *= nCr(N2, N2-r)
    N2 -= r
p /= pow(M, N)
x = 1
while p < pow(10, -x):
    x += 1
print(x)