# -*- coding: utf-8 -*-

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

N, K = MAP()
aN =LIST()

# Kがでかいのは論外
if max(aN) < K:
    print('IMPOSSIBLE')
    exit()

aN.sort()
base = aN[0]
has_odd = False
is_diverse = False
for i in range(N):
    # 奇数が入っているか
    if aN[i] % 2 == 1:
        has_odd = True
    # 倍数以外のものが入っているか
    if aN[i] % base != 0:
        is_diverse = True
# 奇数もあって倍数以外もあれば何でもできる
if has_odd and is_diverse:
    print('POSSIBLE')
# 倍数以外があっても奇数がなかったら行けるのは偶数だけ
elif not has_odd and is_diverse:
    if K % 2 == 0:
        print('POSSIBLE')
    else:
        print('IMPOSSIBLE')
else:
    # 全部baseの倍数だったら、baseの倍数にしか動けない
    if K % base == 0:
        print('POSSIBLE')
    else:
        print('IMPOSSIBLE')
