# -*- coding: utf-8 -*-

"""
・ちょっと時間かかったけど自力AC
・2次元、配るDP
・2つ目の添字が3通りしかないから、制約10万でもいける。
・計算量O(N*3*3)=10万*9=90万、pythonで0.9秒、pypyで0.5秒くらい。
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

# dp[日数][今回ABCのどれにしたか] = その時点での最大値
dp = [[0] * 3 for i in range(N)]
dp[0][0], dp[0][1], dp[0][2] = MAP()

for i in range(N-1):
    abc = LIST()
    # 送り元ループ
    for j in range(3):
        # 送り先ループ
        for k in range(3):
            # 今回と同じ所には送れない
            if j != k:
                dp[i+1][k] = max(dp[i][j] + abc[k], dp[i+1][k])
# 最終日にABCどれにしたかの最大を取る
print(max(dp[N-1]))
