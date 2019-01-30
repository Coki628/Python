# -*- coding: utf-8 -*-

"""
・ちょっとバグ探ししたけどしっかり自力AC
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

N = INT()
aN = LIST()

if N == 1:
    print(1)
    exit()

# フラグで向かう方向を管理
# 0:flat 1:up 2:down
flag = 0
if aN[0] < aN[1]:
    flag = 1
elif aN[0] == aN[1]:
    flag = 0
else:
    flag = 2

cnt = 1
# 向きが変わる所を区切り位置とする
for i in range(2, N):
    if aN[i-1] == aN[i]:
        continue
    elif aN[i-1] > aN[i]:
        if flag == 1:
            cnt += 1
            flag = 0
        else:
            flag = 2
    elif aN[i-1] < aN[i]:
        if flag == 2:
            cnt += 1
            flag = 0
        else:
            flag = 1
print(cnt)
