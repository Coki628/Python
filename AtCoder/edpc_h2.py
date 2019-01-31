# -*- coding: utf-8 -*-

"""
・自力AC
・数え上げDP
・計算量O(H*W)=1000*1000=100万、pythonで1.4秒、pypyで0.8秒くらい。
・貰うDPを配るDPにしてみた。
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

H, W = MAP()

# 四方に一回り大きいグリッドを作る
grid = [['#'] * (W+2) for i in range(H+2)]
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

dp = [[0] * (W+2) for i in range(H+2)]
# スタート地点に1通りを初期値設定
dp[1][1] = 1
for i in range(1,H+1):
    for j in range(1, W+1):
        # 現在位置が壁なら何もできない
        if grid[i][j] == '#':
            continue
        # 下が壁でなければ、値を送る
        if grid[i+1][j] != '#':
             dp[i+1][j] += dp[i][j]
        # 右が壁でなければ、値を送る
        if grid[i][j+1] != '#':
            dp[i][j+1] += dp[i][j]
# ゴール地点の値を確認する
print(dp[H][W] % MOD)
