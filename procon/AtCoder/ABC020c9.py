# -*- coding: utf-8 -*-

"""
ダイクストラと二分探索
疎行列を使ってみる
ちょっと色々修正、あんま速度変わらなかったけど。。
二分探索ライブラリ確認用
"""

from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix

H,W,T = map(int, input().split())

start,goal = None,None
# 左右上下
directions = ((1,0),(-1,0),(0,1),(0,-1))
# 四方に一回り大きいフィールドを作る
field = [['*'] * (W+2) for i in range(H+2)]
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        field[i][j] = row[j-1]
        if row[j-1] == 'S':
            start = (i, j)
        if row[j-1] == 'G':
            goal = (i, j)
# 1次元にするマッピング
mapping = [[0] * (W+2) for i in range(H+2)]
num = 0
for i in range(H+2):
    for j in range(W+2):
        mapping[i][j] = num
        num += 1

# 引数にするグラフの準備
graph = lil_matrix((num, num))
x_list = []
for h1 in range(1,H+1):
    for w1 in range(1, W+1):
        # 4方向見る
        for direction in directions:
            h2=h1+direction[0]
            w2=w1+direction[1]
            # 壁はスキップ
            if field[h2][w2] == '*':
                continue
            elif (field[h2][w2] == '.' 
                    or field[h2][w2] == 'S' 
                    or field[h2][w2] == 'G'):
                graph[mapping[h1][w1], mapping[h2][w2]] = 1
            elif field[h2][w2] == '#':
                # xの値は都度変更できるようにindexだけ保持
                x_list.append((mapping[h1][w1], mapping[h2][w2]))

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

# 壁抜け時間をxとした時、T秒以内でのゴールが可能か
def calc(x):
    # xの値を設定
    for s, t in x_list:
        graph[s,t] = x
    # ダイクストラ
    res = dijkstra(graph.tocsr(), indices=mapping[start[0]][start[1]])[mapping[goal[0]][goal[1]]]
    # T秒以内かどうか
    return res <= T

# 二分探索
res = bisearch_max(0, 10**9, calc)
print(res)
