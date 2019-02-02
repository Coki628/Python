# -*- coding: utf-8 -*-

"""
参考：https://kyopro-friends.hatenablog.com/entry/2019/01/12/231035
・数え上げDP
・累積和で高速化
・計算量O(N*K*2)=2000万がpypyでもTLE…
・C++でAC、0.09秒くらいで。速っ…。
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
aN = LIST()

# dp[この子までの範囲で][飴を配る総数] = 場合の数
dp = [[0] * (K+1) for i in range(N+1)]
# 0番目まで(誰もいない)の範囲で0個配る場合の数は1通り
dp[0][0] = 1

for i in range(N):
    sm = [0] * (K+2)
    # 今回送るi行について累積和を取る
    for j in range(1, K+2):
        sm[j] = sm[j-1] + dp[i][j-1]
    for j in range(K+1):
        # iが送る予定の長さに応じた区間和を取る
        dp[i+1][j] = sm[j+1] - sm[max(j-aN[i], 0)]
        # for k in range(aN[i]+1):
        #     dp[i+1][j+k] += dp[i][j]
print(dp[N][K] % MOD)
