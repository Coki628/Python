# -*- coding: utf-8 -*-

"""
・自力AC
・いわゆる普通のナップザック
・よく貰うDPで紹介されてるけど、配るDPにしてみた。
・計算量O(N*W)=100*10万=1000万、pythonでTLEしてpypyで0.6秒くらいでAC。
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

N, W = MAP()
wN, vN = [0]*N, [0]*N
for i in range(N):
    wN[i], vN[i] = MAP()

# dp[何個目までから選ぶか][これまでに選んだ荷物の重さはこれ以下] = 選んだ荷物の価値
dp = [[0] * (W+1) for i in range(N+1)]

for i in range(N):
    for j in range(W+1):
        # 真下には常に送る
        dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        if j + wN[i] <= W:
            # 重さが制限内なら右下にも送る
            dp[i+1][j+wN[i]] = max(dp[i+1][j+wN[i]], dp[i][j] + vN[i])
print(dp[N][W])
