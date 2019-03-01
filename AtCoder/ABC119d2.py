# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc119/editorial.pdf
・先頭の番兵、0じゃなくて-INFにすれば後で場合分けいらなかった。
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

A,B,Q=MAP()

S=[-INF]*(A+2)
for i in range(A):
    S[i+1]=INT()
T=[-INF]*(B+2)
for i in range(B):
    T[i+1]=INT()
S[-1]=T[-1]=INF

# Si番目の神社から左に一番近い寺
L1=[INF]*(A+2)
# Si番目の神社から右に一番近い寺
R1=[INF]*(A+2)
for i in range(1, A+1):
    idx=bisect_right(T, S[i])-1
    if T[idx]!=0:
        L1[i]=S[i]-T[idx]
    idx=bisect_left(T, S[i])
    if T[idx]!=INF:
        R1[i]=T[idx]-S[i]

# Ti番目の寺から左に一番近い神社
L2=[INF]*(B+2)
# Ti番目の寺から右に一番近い神社
R2=[INF]*(B+2)
for i in range(1, B+1):
    idx=bisect_right(S, T[i])-1
    if S[idx]!=0:
        L2[i]=T[i]-S[idx]
    idx=bisect_left(S, T[i])
    if S[idx]!=INF:
        R2[i]=S[idx]-T[i]

for i in range(Q):
    dist1=dist2=dist3=dist4=INF
    x=INT()
    # 地点xから左に一番近い神社
    idx=bisect_right(S, x)-1
    # idx番目の神社に行く
    dist1=x-S[idx]
    # idx番目の神社から一番近い寺に行く
    dist1+=min(L1[idx], R1[idx])
    idx=bisect_left(S, x)
    dist2=S[idx]-x
    dist2+=min(L1[idx], R1[idx])
    idx=bisect_right(T, x)-1
    dist3=x-T[idx]
    dist3+=min(L2[idx], R2[idx])
    idx=bisect_left(T, x)
    dist4=T[idx]-x
    dist4+=min(L2[idx], R2[idx])
    print(min(dist1, dist2, dist3, dist4))
