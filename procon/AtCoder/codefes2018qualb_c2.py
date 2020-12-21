# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-festival-2018-qualb/editorial.pdf
・規則的な部分は、y = x/2 の1次関数がx=0～4までで循環してると思えばいい。
・例外部分は、訪問済フラグを持っておけば、後で探索して見つけられる。
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

grid = list2d(N+2, N+2, '*')
for i in range(1,N+1):
    for j in range(1, N+1):
        grid[i][j] = '.'

# 例外箇所を埋める時に使う訪問済フラグ
done = list2d(N+2, N+2, False)
for y in range(1,N+1):
    for x in range(1, N+1):
        # y = x/2 の1次関数がx=0～4までで循環してる
        if (2*y - x) % 5 == 1:
            grid[y][x] = 'X'
            # そのマスと上下左右を訪問済にしておく
            done[y][x] = True
            done[y-1][x] = True
            done[y][x-1] = True
            done[y+1][x] = True
            done[y][x+1] = True

# 最初の行の例外
for i in range(1,N+1):
    if not done[1][i]:
        grid[1][i] = 'X'
# 最後の行の例外
for i in range(1,N+1):
    if not done[-2][i]:
        grid[-2][i] = 'X'
# 最初の列の例外
for i in range(1,N+1):
    if not done[i][1]:
        grid[i][1] = 'X'
# 最後の列の例外
for i in range(1,N+1):
    if not done[i][-2]:
        grid[i][-2] = 'X'

for i in range(1, N+1):
    print(''.join(grid[i][1:N+1]))
