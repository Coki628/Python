# -*- coding: utf-8 -*-

"""
ダイクストラと二分探索
疎行列を使ってみる
ちょっと色々修正、あんま速度変わらなかったけど。。
"""

from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix

H,W,T = map(int, input().split())

start,goal = None,None
# 左右上下
directions = [[0,-1],[0,1],[-1,0],[1,0]]
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
for i in range(1,H+1):
    for j in range(1, W+1):
        cur = (i, j)
        # 4方向見る
        for direction in directions:   
            nxt = tuple(map(lambda x,y: x+y, cur, direction))
            # 壁はスキップ
            if field[nxt[0]][nxt[1]] == '*':
                continue
            elif (field[nxt[0]][nxt[1]] == '.' 
                    or field[nxt[0]][nxt[1]] == 'S' 
                    or field[nxt[0]][nxt[1]] == 'G'):
                graph[mapping[cur[0]][cur[1]], mapping[nxt[0]][nxt[1]]] = 1
            elif field[nxt[0]][nxt[1]] == '#':
                # xの値は都度変更できるようにindexだけ保持
                x_list.append((mapping[cur[0]][cur[1]], mapping[nxt[0]][nxt[1]]))

# ダイクストラ
def dk(x):
    # xの値を設定
    for s, t in x_list:
        graph[s,t] = x
    return dijkstra(graph.tocsr(), indices=mapping[start[0]][start[1]])[mapping[goal[0]][goal[1]]]

# 二分探索で最初にTを上回る場所を見つける
hi = 10 ** 9
low = 0
# low+1 == hi -> lowとhiが隣り合った状態なので、境界が確定する
while low+1 < hi:
    mid = (hi+low) // 2
    if dk(mid) > T:
        hi = mid
    else:
        low = mid
# hiの位置が最初にTを上回る場所、lowがTを超えない最大値
print(low)
