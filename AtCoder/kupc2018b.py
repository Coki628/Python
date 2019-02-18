# -*- coding: utf-8 -*-

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

H, W = MAP()

grid = [['*'] * (W+2) for i in range(H+2)]
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]
        if grid[i][j] == 's':
            start = (i, j)

# 制約的に多分メモ化しなくても大丈夫だろう
def rec(h, w, ans):
    # 敵のいるマスか範囲外だったのでこのルートはなし
    if grid[h][w] == 'x' or grid[h][w] == '*':
        return
    # 最後まで行った
    if h == 1:
        print(ans)
        exit()
    # 左に行く
    rec(h-1, w-1, ans+'L')
    # 留まる
    rec(h-1, w, ans+'S')
    # 右に行く
    rec(h-1, w+1, ans+'R')
    return

rec(start[0], start[1], '')
# 最後まで行けなかった
print('impossible')
