# -*- coding: utf-8 -*-

"""
・自力AC
・トポロジカルソート
・pythonで1.1秒、pypyでやったら0.7秒くらいまで速くなった。
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

def topological_sort(N, dag):
    # ここからトポロジカルソート準備
    incnts = [0] * N
    outnodes = [[] for i in range(N)]
    for i in range(len(dag)):
        # 流入するノード数
        incnts[dag[i][1]] += 1
        # 流出先ノードのリスト
        outnodes[dag[i][0]].append(dag[i][1])
    # 流入ノード数が0であるノードのセットS
    S = set()
    for i in range(N):
        if incnts[i] == 0:
            S.add(i)

    # ここからトポロジカルソート
    L = []
    # 暫定セットが空になるまでループ
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
    return L

N, M = MAP()
nodes = [[] for i in range(N)]
edges = [None] * M
for i in range(M):
    x, y = MAP()
    nodes[x-1].append(y-1)
    edges[i] = ((x-1, y-1))

# この後の最大値更新で矛盾が生じないようにソートしておく
l = topological_sort(N, edges)

ans = [0] * N
for src in l:
    for dest in nodes[src]:
        ans[dest] = max(ans[dest], ans[src]+1)
print(max(ans))
