# -*- coding: utf-8 -*-

"""
・奇数でmod取れば大小分かるっていう方針は合ってると思うんだけど、
　どうも31回かかってるぽいのとWAもある。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
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

print('? 2', flush=True)
# まずXの偶奇を確認する
ev_od = input()

def search(num):
    print('? ' + str(num), flush=True)
    tmp = input()
    # 奇数でmod取っても偶奇が一致するなら、まだqが大きい
    return tmp == ev_od

# 二分探索で最初にTを上回る場所を見つける
hi = 10 ** 9
low = 0
# low+1 == hi -> lowとhiが隣り合った状態なので、境界が確定する
while low+1 < hi:
    mid = (hi+low) // 2
    # 必ず奇数を与える
    if search(mid - (mid+1)%2):
        hi = mid
    else:
        low = mid
# hiの位置が最初にTを上回る場所、lowがTを超えない最大値
print('! ' + str(low))
