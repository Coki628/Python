# -*- coding: utf-8 -*-

"""
ワーシャルフロイドで全頂点の最短距離(手作り版)
"""

N,M = map(int, input().split())

# ワーシャルフロイド用隣接行列
graph = [[float('inf')] * N for i in range(N)]
for i in range(M):
    a,b = map(int, input().split())
    # 無向グラフなので両方に
    graph[a-1][b-1] = graph[b-1][a-1] = 1

# ワーシャルフロイドで全頂点の最短距離(手作り版)
for k in range(N):
    for i in range(N):
        for j in range(N):
            # 始点 = 終点、は例外的に距離0にしておく
            if i == j:
                graph[i][j] = 0
            else:
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

for i in range(N):
    # 距離が2 = 友達の友達
    print(graph[i].count(2))
