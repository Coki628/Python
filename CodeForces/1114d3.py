# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/65136
・両側同時に消せる回数は、最長共通部分列から出せる。
・LCSのDP
・回文の最長部分列
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
cN = LIST()
# 隣接して重複している値は不要なので削る
cN2 = [cN[0]]
for i in range(1, N):
    if cN[i] != cN[i-1]:
        cN2.append(cN[i])
# 回文の最長部分列を出すために、逆順との比較をする
cN3 = list(reversed(cN2))
N2 = len(cN2)

dp = [[0] * (N2+1) for i in range(N2+1)]

for i in range(N2+1):
    for j in range(N2+1):
        # 下に送る
        if i < N2:
            dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        # 右に送る
        if j < N2:
            dp[i][j+1] = max(dp[i][j+1], dp[i][j])
        # 今回増やす文字が一致するなら、LCSが1文字増える
        if i < N2 and j < N2 and cN2[i] == cN3[j]:
            # 右下に送る
            dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j]+1)
# 元の間隔の数 - LCS/2切り捨て(奇数の真ん中は減らせる対象じゃない)
print(N2-1 - dp[N2][N2] // 2)
