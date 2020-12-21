# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/drken/items/03c7db44ccd27820ea0d
・確率DP
・添字に何を使うべきかちゃんと考えられるようになりたいね。
・計算量O(N*N)=3000*3000=900万、pypyで0.9秒くらい。
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
pN = list(map(float, input().split()))

# dp[コインを投げた枚数][表が出た枚数] = そうなる確率
dp = [[0] * (N+1) for i in range(N+1)]
# 0枚投げて0になるのは100%
dp[0][0] = 1

for i in range(N):
    # 投げた枚数より表が多く出る訳ないのでi=jまで回せばOK
    for j in range(i+1):
        # 裏になる確率を下に送る
        dp[i+1][j] += dp[i][j] * (1-pN[i])
        # 表になる確率を右下に送る
        # 内ループは短く回るので添字オーバーは気にしなくていい
        dp[i+1][j+1] += dp[i][j] * (pN[i])

print('{:.10f}'.format(sum(dp[N][ceil(N, 2):])))
