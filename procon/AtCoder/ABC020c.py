
# -*- coding: utf-8 -*-

"""
幅優先で経路
"""

from collections import deque

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

# キューが空になるまでやる(BFS)
def bfs(x):
    # 移動距離メモ
    memo = [[float('inf')] * (W+2) for i in range(H+2)]
    # 訪問回数メモ
    visited = [[0] * (W+2) for i in range(H+2)]
    que = deque()
    # スタート位置
    que.append(start)
    memo[start[0]][start[1]] = 0

    while len(que) != 0:
        cur = que.popleft()
        # 4方向見終わった場所はもうやらない
        if visited[cur[0]][cur[1]] == 4:
            continue
        # 4方向見る
        for direction in directions:
            # 訪問回数カウント
            visited[cur[0]][cur[1]] += 1
            # pypy用にnumpy使わないで計算
            nxt = list(map(lambda x,y: x+y, cur, direction))
            # 壁はスキップ
            if field[nxt[0]][nxt[1]] == '*':
                continue
            # 現在位置+1の距離を移動先に記録
            if field[nxt[0]][nxt[1]] == '.' or field[nxt[0]][nxt[1]] == 'G':
                memo[nxt[0]][nxt[1]] = min(memo[cur[0]][cur[1]] + 1, memo[nxt[0]][nxt[1]])
            # 現在位置+xの距離を移動先に記録
            elif field[nxt[0]][nxt[1]] == '#':
                memo[nxt[0]][nxt[1]] = min(memo[cur[0]][cur[1]] + x, memo[nxt[0]][nxt[1]])
            # 始点と終点はそれ以上キューに足さなくていいはず
            if field[nxt[0]][nxt[1]] != 'S' and field[nxt[0]][nxt[1]] != 'G':
                que.append(nxt)
    # 全部見たらゴール位置を返却
    return memo[goal[0]][goal[1]]

i = 2
t = -1
while True:
    t = bfs(i)
    if t > T:
        # bfsの返却値がTを上回ったら結果出力
        print(i-1)
        exit()
    i += 1
