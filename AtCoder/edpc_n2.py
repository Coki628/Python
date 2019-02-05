# -*- coding: utf-8 -*-

"""
参考：https://kyopro-friends.hatenablog.com/entry/2019/01/12/231035
　　　https://www.hamayanhamayan.com/entry/2019/01/10/235656
　　　https://atcoder.jp/contests/dp/submissions/3970847
・区間DP
・メモ化再帰
・累積和
・計算量O(N^3)=6400万で、pythonはTLE、pypyはAC0.8秒。
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
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

N = INT()
aN = LIST()

# 累積和
sm = [0] * (N+1)
for i in range(N):
    sm[i+1] = sm[i] + aN[i]

memo = list2d(N+1, N+1, INF)
def rec(l, r):

    if memo[l][r] != INF: 
        return memo[l][r]
    if l == r:
        return 0
    mn = INF
    # 開区間にした
    for m in range(l, r):
        mn = min(mn, rec(l, m) + rec(m+1, r))
    # この先切った分の最小値に、今回のlとrを合体させる分を足す
    memo[l][r] = mn + (sm[r] - sm[l-1])
    return memo[l][r]

print(rec(1, N))
