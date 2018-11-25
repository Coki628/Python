# -*- coding: utf-8 -*-

"""
ダイクストラ、最短経路DAG、再帰でDFS
→TLE
"""

from scipy.sparse.csgraph import dijkstra

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

cnt = 0
def dfs(cur, _dist):
    global cnt
    if _dist > dist[b-1]:
        return
    if cur == b-1:
        cnt += 1
        return
    for i in range(len(dag)):
        if cur == dag[i][0]:
            dfs(dag[i][1], _dist+1)
    return

# DAGを探索してbまでたどり着く経路の数え上げ
dfs(a-1, 0)
print(cnt)
