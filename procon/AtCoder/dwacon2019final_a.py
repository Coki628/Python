# -*- coding: utf-8 -*-

"""
・負け状態を押し付ける
・結構試行錯誤あったけど自力AC！
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
s = input()

blues = [False] * N
reds = [False] * N
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    # 各ノードの遷移先に青/赤があるかどうかを記録しておく
    if s[a-1] == 'B':
        blues[b-1] = True
    else:
        reds[b-1] = True
    if s[b-1] == 'B':
        blues[a-1] = True
    else:
        reds[a-1] = True
    # 隣接リスト
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)

# ターン数の偶奇
if K % 2 == 0:
    for i in range(N):
        # 開始点が赤だと先手は負け状態を抜け出せない
        if s[i] == 'R':
            print('Second')
        else:
            for v in nodes[i]:
                # 遷移先に、赤に行けない頂点が1つでもあれば後手に負けを押し付けられる
                if not reds[v]:
                    print('First')
                    break
            else:
                print('Second')
else:
    for i in range(N):
        # 開始点が青に行ける頂点なら、後手を負け状態にできる
        if blues[i]:
            print('First')
        else:
            print('Second')
