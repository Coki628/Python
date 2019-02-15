# -*- coding: utf-8 -*-

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
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

def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

N = INT()
A = LIST()

# まず累積和取る
sm = [0] * (N+1)
for i in range(N):
    sm[i+1] = sm[i] + A[i]

# 重複する数の集計
C = Counter(sm)
# 半分で区切る位置を探す用
C2 = Counter()
# 半分で区切れる区間
segs = defaultdict(lambda: [0] * 2)
for i in range(N+1):
    if C[sm[i]] >= 2:
        C2[sm[i]] += 1
        # その区間の開始位置
        if C2[sm[i]] == C[sm[i]] // 2:
            segs[sm[i]] = [i+1, 0]
        # 終了位置
        if C2[sm[i]] == ceil(C[sm[i]], 2) + 1:
            segs[sm[i]][1] = i

imos = [0] * (N+2)
for k, v in C2.items():
    # 考慮するのは組み合わせに影響のある、集計で2つ以上あったところ
    if v >= 2:
        # 該当区間に、重複の多い値ほど多く加点していく
        imos[segs[k][0]] += v
        imos[segs[k][1]+1] -= v
mx_i = 0
mx = imos[0]
for i in range(N+1):
    imos[i+1] += imos[i]
    # 最大値の位置は取っておく
    if mx < imos[i+1]:
        mx = imos[i+1]
        mx_i = i+1
# 一番いい位置で分割
sm1 = sm[:mx_i]
sm2 = sm[mx_i:]
C3 = Counter(sm1)
C4 = Counter(sm2)

# 準備ができたので区間和が0になる組み合わせを数える
ans = 0
for k, v in C3.items():
    if v >= 2:
        ans += nCr(v, 2)
for k, v in C4.items():
    if v >= 2:
        ans += nCr(v, 2)
print(ans)
