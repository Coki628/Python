# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/tkppc3/editorial.pdf
・公式解に寄せる。
・累積和を作る時に偶奇で格納場所を分けるのでなく、マイナスする。
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

H,W,Q=MAP()
A=LIST()
B=LIST()

# 白になる場所をマイナスしていけば配列は2ついらない
H1=[0]*(H+1)
for i in range(H):
    if i%2==0:
        H1[i+1]=H1[i]+A[i]
    else:
        H1[i+1]=H1[i]-A[i]
W1=[0]*(W+1)
for i in range(W):
    if i%2==0:
        W1[i+1]=W1[i]+B[i]
    else:
        W1[i+1]=W1[i]-B[i]

for i in range(Q):
    px,py,qx,qy=MAP()
    # ここもシンプルに区間和取るだけでよくなる
    print((H1[qx]-H1[px-1])*(W1[qy]-W1[py-1]))
