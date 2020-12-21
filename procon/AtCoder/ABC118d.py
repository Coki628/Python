# -*- coding: utf-8 -*-

"""
・余計なソートしてたり試行錯誤あったけど自力AC！
・400点のDPをちゃんとほぼ想定解できたのは嬉しい。
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

N, M = MAP()
A = LIST()
need = [6,2,5,5,4,5,6,3,7,6]

# dp[i] := ちょうどi本のマッチを使って作れる最大値
dp = [''] * (N+10)
for a in A:
    dp[need[a]] = max(dp[need[a]], str(a))

for i in range(N):
    if dp[i] != '':
        for a in A:
            # 基本は文字列の長さ(桁数)で比較
            if len(dp[i+need[a]]) < len(dp[i])+1:
                dp[i+need[a]] = dp[i]+str(a)
            # それが同じ時だけ辞書順の大小で比較
            elif len(dp[i+need[a]]) == len(dp[i])+1:
                dp[i+need[a]] = max(dp[i+need[a]], dp[i]+str(a))
print(dp[N])
