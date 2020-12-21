# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/65136
　　　submission:49738197
・区間DP
・メモ化再帰
・DPコンのスライムと一緒で、完成から戻してく方針でやれば、
　開始位置を意識せずに最適が出せるみたいだ。
・TLEだったので、C++で同じ実装作ったらACした。
・計算量O(N^2)=5000^2=2500万くらい。
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
cN = LIST()
# 隣接して重複している値は不要なので削る
cN2 = [cN[0]]
for i in range(1, N):
    if cN[i] != cN[i-1]:
        cN2.append(cN[i])

# メモ[左端位置][右端位置] = この局面からの最短ターン数
memo = list2d(len(cN2)+5, len(cN2)+5, INF)
def rec(l, r):
    if l == r:
        return 0
    if memo[l][r] != INF:
        return memo[l][r]
    res = INF
    if cN2[l] == cN2[r]:
        res = min(res, rec(l+1, r-1)+1)
    else:
        res = min(res, rec(l+1, r)+1)
        res = min(res, rec(l, r-1)+1)
    memo[l][r] = res
    return res

print(rec(0, len(cN2)-1))
