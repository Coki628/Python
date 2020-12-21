# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/65295
　　　submission:50019541
・左右でxorが等しい → 左右をxorしたら0になる。
・予め累積xorを取っておいて、区間xorが0になる場所を数える。
・要素数が偶数個の区間しか見ないので、奇数始まりと偶数始まりで分けて集計する。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, xor
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
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

# 累積xor
acc = [0] + list(accumulate(A, xor))

# 偶数ポジと奇数ポジで分けて集計
C1 = Counter()
for i in range(0, N+1, 2):
    C1[acc[i]] += 1
C2 = Counter()
for i in range(1, N+1, 2):
    C2[acc[i]] += 1

ans = 0
for k, v in C1.items():
    if v >= 2:
        # nC2の組み合わせ
        ans += v * (v-1) // 2
for k, v in C2.items():
    if v >= 2:
        ans += v * (v-1) // 2       
print(ans)
