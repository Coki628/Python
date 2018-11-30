# -*- coding: utf-8 -*-

"""
ワーシャルフロイドで全頂点の最短距離(ライブラリ版)
"""

import numpy as np
from scipy.sparse.csgraph import floyd_warshall

N,M = map(int, input().split())

# ワーシャルフロイド用隣接行列
graph = [[0] * N for i in range(N)]
for i in range(M):
    a,b = map(int, input().split())
    graph[a-1][b-1] = 1
# ワーシャルフロイドで全頂点の最短距離
dist = floyd_warshall(graph, directed=False)
for i in range(N):
    # 距離が2 = 友達の友達
    print(np.sum(dist[i] == 2))
