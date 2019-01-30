# -*- coding: utf-8 -*-

"""
・結構考えたけど、なんとか自力AC
・これ200点かなぁ…。
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
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def ceil(a, b=1): return int(-(-a // b))
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

def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

N, P = MAP()
aN = LIST()

# この問題では偶奇しか関係しないのでmod 2する
for i in range(N):
    aN[i] %= 2
c = Counter(aN)

ans = 0
# 最終的に偶奇どっちにするかによって、奇数を取る数を変える
if P == 0:
    # 奇数を偶数個取る組み合わせ
    for i in range(0, c[1]+1, 2):
        ans += nCr(c[1], i)
else:
    # 同じように奇数個取る組み合わせ
    for i in range(1, c[1]+1, 2):
        ans += nCr(c[1], i)
# 偶数の数は偶奇の変化に影響を与えないので、取るor取らないの全組み合わせ(2**偶数の数)をかける
print(ans * 2**c[0])
