# -*- coding: utf-8 -*-

"""
・ちょっと場合分けとか時間かかったけど、問題なく自力AC
・でもうまい人とかもっとキレイに条件作るんだろーなー。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul
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

N = INT()

grid = list2d(N, N, '.')
for i in range(N):
    for j in range(N):
        if (i % 5 == 0 and j % 5 == 0
                or i % 5 == 1 and j % 5 == 2
                or i % 5 == 2 and j % 5 == 4
                or i % 5 == 3 and j % 5 == 1
                or i % 5 == 4 and j % 5 == 3):
            grid[i][j] = 'X'
# 最初の行の例外
for i in range(N):
    if i % 5 == 3:
        grid[0][i] = 'X'
# 最後の行の例外
if (N-1) % 5 == 0: key = 2
elif (N-1) % 5 == 1: key = 4
elif (N-1) % 5 == 2: key = 1
elif (N-1) % 5 == 3: key = 3
elif (N-1) % 5 == 4: key = 0
for i in range(N):
    if i % 5 == key:
        grid[-1][i] = 'X'
# 最後の行の例外
for i in range(N):
    if i % 5 == 2:
        grid[i][0] = 'X'
# 最後の列の例外
if (N-1) % 5 == 0: key = 3
elif (N-1) % 5 == 1: key = 1
elif (N-1) % 5 == 2: key = 4
elif (N-1) % 5 == 3: key = 2
elif (N-1) % 5 == 4: key = 0
for i in range(N):
    if i % 5 == key:
        grid[i][-1] = 'X'
for i in range(N):
    print(''.join(grid[i]))
