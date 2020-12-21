# -*- coding: utf-8 -*-

"""
幅優先探索
"""

from collections import deque

R, C = map(int, input().split())
sy, sx = map(int, input().split())
gy, gx = map(int, input().split())

start,goal = (sy-1, sx-1), (gy-1, gx-1)
# 左右上下
directions = [[0,-1],[0,1],[-1,0],[1,0]]

field = []
for i in range(R):
    field.append(list(input()))
field[goal[0]][goal[1]] = 'G'

# 移動距離メモ
memo = [[float('inf')] * (C) for i in range(R)]
# 訪問済メモ
visited = [[False] * (C) for i in range(R)]
que = deque()
# スタート位置
que.append(start)
memo[start[0]][start[1]] = 0
# キューが空になるまでやる(BFS)
while len(que) != 0:
    cur = que.popleft()
    # 見終わった場所はもうやらない
    if visited[cur[0]][cur[1]]:
        continue
    # 訪問済にする
    visited[cur[0]][cur[1]] = True
    # 4方向見る
    for direction in directions:
        # pypy用にnumpy使わないで計算
        nxt = list(map(lambda x,y: x+y, cur, direction))
        # 壁はスキップ
        if field[nxt[0]][nxt[1]] == '#':
            continue
        # 訪問済ではない時だけ
        if not visited[nxt[0]][nxt[1]]:
            # 現在位置+1の距離を移動先に記録
            memo[nxt[0]][nxt[1]] = memo[cur[0]][cur[1]] + 1
            # ゴールに着いたら終了
            if field[nxt[0]][nxt[1]] == 'G':
                print(memo[nxt[0]][nxt[1]])
                exit()
            else:
                que.append(nxt)
