# -*- coding: utf-8 -*-

"""
ダイクストラ、最短経路DAG、トポロジカルソート、経路DP
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
# 暫定ソート結果を保持する空リストL
L = []

# ここからトポロジカルソート
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
