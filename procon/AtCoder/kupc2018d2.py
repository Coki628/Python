# -*- coding: utf-8 -*-

"""
参考：https://www.kupc.jp/#/2018/
　　　https://atcoder.jp/contests/kupc2018/submissions/3329535
・インタラクティブ問題
・方針はすぐ立ったけど、検索に奇数しか使えない状況での細かい位置調整に困った。
・奇数でしか探索できないので、最後は2値まで絞ったら偶奇でどちらか判断する。
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

print('?', 2, flush=True)
# まずXの偶奇を確認する
is_odd = 0
if input() == 'odd':
    is_odd = 1

def search(q):
    print('?', q, flush=True)
    # 奇数でmod取っても偶奇が一致するなら、まだqが大きい(ans<q)
    return (input() == 'odd') == is_odd

# 二分探索
hi = 10 ** 9 + 1
low = 1
while low+2 < hi:
    # 1で論理和取れば、1がなければ足されるしあればそのままで都合がいい
    mid = ((hi+low) // 2) | 1
    # 必ず奇数を与える
    if search(mid):
        hi = mid
    else:
        low = mid
# lowかlow+1までは絞り込めているので、偶奇でどちらか判断する
if is_odd == low % 2:
    print('!', low, flush=True)
else:
    print('!', low+1, flush=True)
