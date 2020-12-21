# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc119/editorial.pdf
・前計算しないで直接やる版
・INFがfloat('inf')だとダメで10**18にしたらいけた。INF-INFがnanになるからだった。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

A,B,Q=MAP()

S=[-INF]*(A+2)
for i in range(A):
    S[i+1]=INT()
T=[-INF]*(B+2)
for i in range(B):
    T[i+1]=INT()
S[-1]=T[-1]=INF

for i in range(Q):
    dist1=dist2=dist3=dist4=INF
    x=INT()
    # 地点xから左に一番近い神社
    idx1=bisect_right(S, x)-1
    # 右に一番近い神社
    idx2=bisect_left(S, x)
    idx3=bisect_right(T, x)-1
    idx4=bisect_left(T, x)
    # idx1番目の神社に行く
    dist1=x-S[idx1]
    # idx1番目の神社から、xから右か左に一番近かった寺に行く
    dist1+=min(abs(S[idx1]-T[idx3]), abs(S[idx1]-T[idx4]))
    dist2=S[idx2]-x
    dist2+=min(abs(S[idx2]-T[idx3]), abs(S[idx2]-T[idx4]))
    dist3=x-T[idx3]
    dist3+=min(abs(T[idx3]-S[idx1]), abs(T[idx3]-S[idx2]))
    dist4=T[idx4]-x
    dist4+=min(abs(T[idx4]-S[idx1]), abs(T[idx4]-S[idx2]))
    print(min(dist1, dist2, dist3, dist4))
