# -*- coding: utf-8 -*-

"""
・昔やったやつを振り返る会
・内ループをdictじゃなくて普通にlistで持つ版
・もちろんこっちのが盛大にTLE
・似たような実装をC++でAC(色々大変だったけど)
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

N, Y = MAP()
Y //= 1000

if N*10 < Y:
    print(-1, -1, -1)
    exit()

dp = [[[-1, -1, -1] for j in range(20020)] for i in range(N+1)]
dp[1][1] = [0, 0, 1]
dp[1][5] = [0, 1, 0]
dp[1][10] = [1, 0, 0]

for i in range(1, N):
    for j in range(20000):
        # 極端に目的の金額から遠くなるケースはそれ以上やらない
        if j > Y or j + 10*(N-i) < Y: 
            continue
        if dp[i][j] != [-1, -1, -1]:
            x, y, z = dp[i][j]
            dp[i+1][j+1] = [x, y, z+1]
            dp[i+1][j+5] = [x, y+1, z]
            dp[i+1][j+10] = [x+1, y, z]
print(*dp[N][Y])
