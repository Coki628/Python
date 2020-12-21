# -*- coding: utf-8 -*-

"""
・手元では動くけどあとこ環境ではWA
・indices付ダイクストラに普通の2次元配列入れちゃダメみたい。
"""

import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra, floyd_warshall

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
G = np.full((N, N), INF, dtype=np.int64)
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    G[a,b] = c
    G[b,a] = c

fw = floyd_warshall(G, directed=False)

for _ in range(INT()):
    a, b, c = MAP()
    a -= 1; b -= 1
    G[a,b] = min(G[a,b], c)
    G[b,a] = min(G[b,a], c)
    for i in range(N):
        # 今回のコストより遠い頂点があったら最短路確認
        if G[i].max() > c:
            dk = dijkstra(G, indices=i)
            fw[i] = dk
            fw[:,i] = dk
    print(int(fw.sum() // 2))
