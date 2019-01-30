# -*- coding: utf-8 -*-

"""
・しゃくとり法の変形
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
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
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
aN = LIST()

ans = 0
for i in range(1, max(aN)+1):
    l = r = 0
    # 外ループで左を動かす
    while l < N:
        # 内ループは条件を満たす限り右を動かす
        while r < N and aN[r] >= i:
            r += 1
        if l != r:
            ans += 1
            l = r
        else:
            l += 1
            r += 1
print(ans)
