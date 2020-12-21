# -*- coding: utf-8 -*-

"""
・貪欲なんだけど、条件色々考えなきゃいけない系
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
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

N, K, M = MAP()
aN = LIST()

aN.sort()
mx = max(aN)
idx = bisect_left(aN, mx)

# 最大値以外の全員を消せるか
if idx <= M:
    # 最大値以外消して最大値全員に残りを振る
    M2 = M - idx
    aN2 = aN[idx:]
    for i in range(len(aN2)):
        if M2 >= K:
            aN2[i] += K
            M2 -= K
        else:
            aN2[i] += M2
            M2 = 0
            break
    # 1人だけ残して残りはそいつに全振り
    M -= len(aN)-1
    aN = aN[-1:]
    aN[0] += min(M, K)
    ans = max(sum(aN) / len(aN), sum(aN2) / len(aN2))
else:
    # 消せる所まで消す
    aN2 = aN[M:]
    # 消さないで全員に振る
    for i in range(len(aN)):
        if M >= K:
            aN[i] += K
            M -= K
        else:
            aN[i] += M
            M = 0
            break
    ans = max(sum(aN) / len(aN), sum(aN2) / len(aN2))
print(ans)
