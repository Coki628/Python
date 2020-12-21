# -*- coding: utf-8 -*-

"""
・トポロジカルソート
・時間かかったけど自力AC！
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

# トポロジカルソート(頂点数、辺集合(DAG, 0-indexed))
def topological_sort(N: int, edges: list) -> list:
    # ここからトポロジカルソート準備
    incnts = [0] * N
    outnodes = [[] for i in range(N)]
    for i in range(len(edges)):
        # 流入するノード数
        incnts[edges[i][1]] += 1
        # 流出先ノードのリスト
        outnodes[edges[i][0]].append(edges[i][1])
    # 流入ノード数が0であるノードのセットS
    S = set()
    for i in range(N):
        if incnts[i] == 0:
            S.add(i)

    # ここからトポロジカルソート
    L = []
    # 暫定セットが空になるまでループ
    i = 0
    while S:
        # 暫定セットから結果リストへ1つ入れる
        L.append(S.pop())
        # 確定させたノードから流出するノードでループ
        for node in outnodes[L[-1]]:
            # 流入ノード数を1減らす
            incnts[node] -= 1
            # 流入ノードが0なら暫定セットへ
            if incnts[node] == 0:
                S.add(node)
        # 各頂点が何番目かを保持しておく
        ts2[L[-1]] = i
        i += 1
    # ソートされた頂点のリストを返却
    return L

N, M = MAP()
edges = [None] * (N+M-1)
nodes2 = [[] for i in range(N)]
for i in range(N+M-1):
    u, v = MAP()
    edges[i] = (u-1, v-1)
    # node[到着点] = 出発点 の隣接リスト
    nodes2[v-1].append(u-1)

ts2 = [0] * N
ts = topological_sort(N, edges)

ans = [-1] * N
for v in range(N):
    mx = -1
    for u in nodes2[v]:
        # トポソ順で自分に近い(大きい)方を優先して残す
        if mx < ts2[u]:
            mx = ts2[u]
            ans[v] = u

for i in range(N):
    print(ans[i]+1)
