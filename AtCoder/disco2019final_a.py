# -*- coding: utf-8 -*-

"""
・しゃくとり変形
・自力AC。まあ300点だし。
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

N = INT()
S = input()

l, r, mx = 0, 0, 0
lmx, rmx = 0, 0
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N and S[r] == '>':
        r += 1
    if mx < r - l:
        mx = r - l
        lmx = l
        rmx = r
    r += 1
    l = r

# 全部氷
if lmx == 0 and rmx == N:
    pass
# 全部雪
elif lmx == 0 and rmx == 0:
    S = list(S)
    S[0] = '>'
    S = ''.join(S)
# 最大区間が先頭を含む
elif lmx == 0:
    S = list(S)
    # 最大区間の後ろに氷を追加
    S[rmx] = '>'
    S = ''.join(S)
else:
    S = list(S)
    # 最大区間の先頭に氷を追加
    S[lmx-1] = '>'
    S = ''.join(S)

cnt = 0
k = 0
for i in range(N):
    if S[i] == '-':
        cnt += 1
        k = 0
    else:
        cnt += 1 / (k+2)
        k += 1
print(cnt)
