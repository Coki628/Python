# -*- coding: utf-8 -*-

"""
・TLEは分かるんだけど、何でWAなんだろう。
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

N, M, K = MAP()
pN = LIST()

nodes = [[] for i in range(N)]
root = -1
for i in range(N):
    if pN[i] != 0:
        nodes[pN[i]-1].append(i)
    else:
        root = i

depths = [0] * N
def upd_dpth(children, depth):
    for i in children:
        depths[i] = depth+1 
        upd_dpth(nodes[i], depth+1)
for i in range(N):
    # 深さの更新
    if i == root:
        # 深さを引数に持てばルートから1回回るだけで終わる
        upd_dpth(nodes[i], 0)

for comb in combinations([i for i in range(N)], M):
    sm = 0
    for v in comb:
        sm += depths[v]
    sm += M
    if sm == K:
        print(*[v+1 for v in comb])
        exit()
print(-1)
