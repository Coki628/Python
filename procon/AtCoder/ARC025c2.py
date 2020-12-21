# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/arc025/submissions/7430527
・scipyダイクストラ版
・scipyダイクストラに対する、疎なグラフの入力はkawap23さんのやつが速いのが良く分かった。
　これの代わりにnumpyで普通の隣接行列とか使うとTLEする。
・ちなみにnp.searchsortedのにぶたん高速化も必要で、これも普通にループとbisectでやるとTLEする。
　ダイクストラがボトルネックかと思って試しににぶたんだけ普通にやってもやっぱりダメだった。
・pythonAC4.6秒。(制約7秒)
"""

import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

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

N, M, R, T = MAP()
edges = np.array([input().split() for _ in range(M)], dtype=np.int64).T
G = csr_matrix((edges[2], (edges[:2] - 1)), (N, N))

RT = R / T
ans = 0
for i in range(N):
    # 頂点iをゴールとして、他の各頂点からの最短距離
    res = dijkstra(G, directed=False ,indices=i)
    res.sort()
    # ゴール以外の各頂点について、頂点jをカメの始点とした時、ウサギの始点候補がいくつあるか
    res = res[1:]
    cnt = N - np.searchsorted(res, res*RT, side='right') - 1
    # カメのが速い場合は、自分のいる頂点を含んでしまうので減らす
    if R < T:
        cnt -= 1
    ans += cnt.sum()
print(ans)
