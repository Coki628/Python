# -*- coding: utf-8 -*-

"""
・自力AC
・1次元、配るDP
・1つ先と2つ先に送る。
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

N = INT()
hN = LIST()

# dp[足場番号] = そこまで行くための最小コスト
dp = [INF] * (N)
dp[0] = 0
for i in range(N):
    if i+1 < N:
        # 1つ先に跳ぶ
        dp[i+1] = min(dp[i] + abs(hN[i+1]-hN[i]), dp[i+1])
    if i+2 < N:
        # 2つ先に跳ぶ
        dp[i+2] = min(dp[i] + abs(hN[i+2]-hN[i]), dp[i+2])
print(dp[N-1])
