# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc119/editorial.pdf
・公式解の再帰
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, xor
from copy import copy, deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits
from datetime import date

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N,A,B,C=MAP()
L=[INT() for i in range(N)]

def rec(i, a, b, c):
    if i==N:
        # a,b,cどれか1つでも0なら1本も竹を使ってないので無効
        if min(a, b, c)==0:
            return INF
        # 合体させてきたabcを伸縮させてそれぞれABCにする
        a=abs(A-a)
        b=abs(B-b)
        c=abs(C-c)
        # それぞれ1本目には合体コストが要らないので各-10で-30
        return a+b+c-30
    res=INF
    # i本目の竹は使わない
    res=min(res, rec(i+1, a, b, c))
    # i本目の竹をa(Aにする予定の集まり)に合体させる(b,cも同じ)
    res=min(res, rec(i+1, a+L[i], b, c)+10)
    res=min(res, rec(i+1, a, b+L[i], c)+10)
    res=min(res, rec(i+1, a, b, c+L[i])+10)
    return res

print(rec(0, 0, 0, 0))
