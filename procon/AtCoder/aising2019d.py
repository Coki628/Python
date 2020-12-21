# -*- coding: utf-8 -*-

"""
・調査用
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

# 参考テストケース
# 5
# 3 5 7 11 13
# 10
# 2 4 7 11 12 19 24 26 29 35

N = INT()
A = LIST()

def calc(x):
    taken = [False] * N
    ans = 0
    i = N-1
    for turn in range(N):
        if turn % 2 == 0:
            while i >= 0:
                if not taken[i]:
                    ans += A[i]
                    taken[i] = True
                    i -= 1
                    break
                else:
                    i -= 1
        else:
            mn = INF
            idx = -1
            for i in range(N):
                if not taken[i]:
                    if mn > abs(A[i] - x):
                        mn = abs(A[i] - x)
                        idx = i
                    elif mn == abs(A[i] - x):
                        if A[i] < A[idx]:
                            mn = abs(A[i] - x)
                            idx = i
            taken[idx] = True
    return ans

for i in range(1, 40):
    print(i, calc(i))
