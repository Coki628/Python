# -*- coding: utf-8 -*-

"""
・桁和系
・自力AC！
・まあ、調べる値の取り方はほぼABC101Dの焼き増しなんだけどね。
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
def ceil(a, b=1): return int(-(-a // b))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, initial=1)
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()

def digit_sum(i):
    sm = 0
    for s in str(i):
        sm += int(s)
    return sm

suff = '9999999999999'
ans = 0 
# 後ろに付ける9の数を減らして、10^16-1から10099までを作る
for i in range(12):
    # 999～100までループ
    for j in range(999, 99, -1):
        j = int(str(j) + suff)
        if j > N:
            continue
        ans = max(digit_sum(j), ans)
    suff = suff[:-1]
# 9999以下は普通に全部やる(大した大きさじゃないので)
for i in range(9999, 0, -1):
        if i > N:
            continue
        ans = max(digit_sum(i), ans)
print(ans)
