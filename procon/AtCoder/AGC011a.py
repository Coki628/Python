# -*- coding: utf-8 -*-

"""
・シミュレーション
・さくっとAC
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
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, C, K = MAP()
tN = [INT() for i in range(N)]

tN.sort()
ans = 0
bus = 0
timer = K
for i in range(N):
    # バスに人が乗っていたら時間制限を確認
    if 1 <= bus <= C-1:
        timer -= tN[i] - tN[i-1]
        # 時間が過ぎていたら今のバスは出発済にする
        if timer < 0:
            ans += 1
            bus = 0
            timer = K
    # 最後の1人以外は普通に乗せる
    if 0 <= bus <= C-2:
        bus += 1
    # 最後の1人が乗ったら出発
    elif bus == C-1:
        ans += 1
        bus = 0
        timer = K
# 未出発のバスがいれば出発させる
if bus != 0:
    ans += 1
print(ans)
