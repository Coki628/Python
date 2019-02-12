# -*- coding: utf-8 -*-

"""
・式変形
・なんとか自力ACしたけど、式立てればいいって気付くのに時間かかったー…。
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

X, Y = MAP()

# 連立方程式を変形する
# 1. X-a-3b = 0
# 2. Y-3a-b = 0
a = (X - 3*Y) / -8
b = (Y - 3*X) / -8

# 答えが正の整数なら、可能な方法
if a >= 0 and b >= 0 and a % 1 == 0 and b % 1 == 0:
    print('Yes')
else:
    print('No')
