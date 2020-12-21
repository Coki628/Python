# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc017/editorial.pdf
・公式解
・偶数だけなら全部か0、奇数が1つでもあれば(奇数がいくつあっても)偶奇どっちになるかは綺麗に半々。
・まじかよ、奇数の数めっちゃ影響ありそうなのに。
　→ある奇数の袋を取るor取らないで半々、偶数の袋はいくら取ろうと影響なし、で最終的に半々って感じか。
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

N, P = MAP()
aN = LIST()

has_odd = False
for i in range(N):
    if aN[i] % 2 == 1:
        has_odd = True
        break
if not has_odd:
    if P == 0:
        print(2 ** N)
    else:
        print(0)
else:
    print(2 ** (N-1))
