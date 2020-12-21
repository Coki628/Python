# -*- coding: utf-8 -*-

"""
・TLE
・scipyダイクストラとワーシャルフロイド
・演算させるにはcsrのが速いみたいなので変換してる。
・元グラフの最小更新の時、INFな場所には0が入ってるから、それを考慮して更新する。
"""

import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra, floyd_warshall
from scipy.sparse import lil_matrix

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
G = lil_matrix((N, N), dtype=np.int64)
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    G[a,b] = c
G = G.tocsr()

fw = floyd_warshall(G, directed=False)

for _ in range(INT()):
    a, b, c = MAP()
    a -= 1; b -= 1
    if G[a,b] == 0 or c < G[a,b]:
        G[a,b] = c
    for i in range(N):
        # 今回のコストより遠い頂点があったら最短路確認
        if fw[i].max() > c:
            dk = dijkstra(G, directed=False, indices=i)
            fw[i] = dk
            fw[:,i] = dk
    print(int(fw.sum() // 2))
