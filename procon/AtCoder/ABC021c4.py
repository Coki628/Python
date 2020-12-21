# -*- coding: utf-8 -*-

"""
ダイクストラ、最短経路DAG、トポロジカルソート、経路DP
→トポロジカルソートにライブラリ使ってみる(本番使えないんだけどね)
"""

from scipy.sparse.csgraph import dijkstra
import networkx as nx

N = int(input())
a,b = map(int, input().split())
M = int(input())
xM,yM = [0]*M, [0]*M
for i in range(M):
    x, y = map(int, input().split())
    xM[i] = x - 1
    yM[i] = y - 1

# 始点aから全頂点への最短経路を探す(今回はダイクストラした)
dk = [[float('inf')] * N for i in range(N)]
for i in range(M):
    dk[xM[i]][yM[i]] = 1
    dk[yM[i]][xM[i]] = 1
dist = dijkstra(dk, directed=False, indices=a-1)

# 最短経路DAGの作成
# "ある家への最短 + コスト = 次の家への最短" が成り立つ道のみで構成する
dag = []
for i in range(M):
    if dist[xM[i]] + 1 == dist[yM[i]]:
        dag.append((xM[i], yM[i]))
    elif dist[yM[i]] + 1 == dist[xM[i]]:
        dag.append((yM[i], xM[i]))

# トポロジカルソート
G = nx.DiGraph()
G.add_edges_from(dag)
L = list(nx.topological_sort(G))

# 経路DP
dp = [0] * N
# 開始位置は0じゃなくてa-1
dp[a-1] = 1
# ソート済のノードリストLでループ
for i in range(1, N):
    for j in range(len(dag)):
        # ノードL[i]が到着点となる経路があれば
        if dag[j][1] == L[i]:
            # その経路の出発点に記載された経路数を到着点に足す
            dp[dag[j][1]] += dp[dag[j][0]]
# 最終的な目的地の場所(b-1)で出力
print(dp[b-1] % (10 ** 9 + 7))
