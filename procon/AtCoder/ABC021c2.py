# -*- coding: utf-8 -*-

"""
ダイクストラ、最短経路DAG、メモ化再帰
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

# 各位置から目的地(b-1)に辿り着く経路数をメモする
memo = [-1] * N
def dfs(cur, _dist):
    # 距離が遠い場所はもう目的地に着かないので0
    if _dist > dist[b-1]:
        return 0
    # 目的地に着いたら1を返す
    if cur == b-1:
        return 1
    # 既に確認済みの場所はメモの値を返す
    if memo[cur] != -1:
        return memo[cur]
    # 別の枝を探しに行く前にカウントリセット
    cnt = 0
    for i in range(len(dag)):
        # 出発地が現在地の道を見る
        if cur == dag[i][0]:
            # 到着地で再帰して、戻り値をメモに入れる
            memo[dag[i][1]] = dfs(dag[i][1], _dist+1)
            # 現在地より先で、目的地に到達した経路数をカウントしていく
            cnt += memo[dag[i][1]]
    # 現在地より先を探し終えたら、カウントした経路数を返す
    return cnt

# DAGを探索してbまでたどり着く経路の数え上げ
memo[a-1] = dfs(a-1, 0)
print(memo[a-1] % (10 ** 9 + 7))
