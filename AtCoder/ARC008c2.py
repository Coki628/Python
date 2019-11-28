# -*- coding: utf-8 -*-

"""
・自力TLE,MLE
・解法としては合ってた。
・初期AtCoderのメモリ制限64Mに苦しまされた。
・ダイクストラを自前のからscipyのものにしたみた。
・これだけだとメモリはほぼ変化なし。
・速度はちょっと良くなって、pythonTLEはMLEになった。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from math import hypot

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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

N = INT()
XYTR = []
for i in range(N):
    x, y, t, r = MAP()
    XYTR.append((x, y, t, r))

# 全ペアについてコスト計算して辺を張る
edges = []
for i in range(N):
    x, y, t, r = XYTR[i]
    for j in range(i+1, N):
        x2, y2, t2, r2 = XYTR[j]
        dist = hypot(x-x2, y-y2) / min(t, r2)
        edges.append((i, j, dist))
        dist = hypot(x-x2, y-y2) / min(t2, r)
        edges.append((j, i, dist))

# 例外処理
if N == 1:
    print(0)
    exit()

# scipyダイクストラに渡すグラフの準備
edges = np.array(edges).T
G = csr_matrix((edges[2], (edges[:2])), (N, N))
del edges

# 自分からの最短距離
res = list(dijkstra(G, indices=0))

res.sort(reverse=True)
ans = 0
for i in range(N-1):
    # 1秒1回しか投げれないので、後回しにする程コストとして秒数が乗る
    ans = max(ans, res[i] + i)
print(ans)
