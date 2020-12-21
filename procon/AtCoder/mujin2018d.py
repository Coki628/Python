# -*- coding: utf-8 -*-

"""
・動きをシミュレートしてみたものの、規則も見つからず解法は出ず。
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

N,M=MAP()

def rev(x): return int(str(x)[::-1])

def f(x, y, mnx, mny, lim):
    # print(x,y)
    if x==0 or y==0:
        return 0
    if x<y:
        x=rev(x)
    else:
        y=rev(y)
    if x<y:
        y=y-x
    else:
        x=x-y
    if mnx>x:
        mnx=x
    elif mny>y:
        mny=y
    else:
        lim+=1
        # これでも数が大きくなると無限じゃないのを拾ってしまう
        if lim==50:
            return 1
    if f(x, y, mnx, mny, lim):
        return 1
    else:
        return 0

cnt=0
for i in range(1, N+1):
    for j in range(1, M+1):
        cnt+=f(i,j,i,j,0)
print(cnt)
# f(N,M,N,M,0)
