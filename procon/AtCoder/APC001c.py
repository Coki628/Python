# -*- coding: utf-8 -*-

"""
・久々の500点自力AC
・でもまあ、かなり似たようなインタラクティブの二分探索やったことあったからなんだけど。
　(kupc2018d)
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

N=INT()

# まず男女の偶奇を確認する
print(0, flush=True)
s=input()
if s=='Male':
    even='Male'
    odd='Female'
elif s=='Female':
    even='Female'
    odd='Male'
else:
    exit()

def calc(q):
    print(q, flush=True)
    s=input()
    if s=='Vacant':
        exit()
    # 順番が合っているかどうかを判定
    if q%2==0: 
        return s==even
    else: 
        return s==odd

# 二分探索
hi=N-1
low=0
while low+1<hi:
    mid=(hi+low)//2
    if calc(mid):
        # 順番が合っていれば、答えはそこより先にある
        low=mid
    else:
        # 違っていれば、手前にある
        hi=mid
print(hi, flush=True)
