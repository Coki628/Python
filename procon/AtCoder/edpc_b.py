# -*- coding: utf-8 -*-

"""
・自力AC
・1次元、配るDP
・1～K個先まで送る。ループは増えたけど添字は増えない。
・計算量O(N*K)=10万*100=1000万、pythonでTLEしてpypyで0.6秒くらいでAC。
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

N, K = MAP()
hN = LIST()

# dp[足場番号] = そこまで行くための最小コスト
dp = [INF] * (N)
dp[0] = 0
for i in range(N):
    for j in range(1, K+1):
        if i+j < N:
            # j個先に跳ぶ
            dp[i+j] = min(dp[i] + abs(hN[i+j]-hN[i]), dp[i+j])
print(dp[N-1])
