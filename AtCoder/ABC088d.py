# -*- coding: utf-8 -*-

"""
きっちり自力AC！
・幅優先探索
"""

import sys
input = sys.stdin.readline

from collections import deque

H, W = map(int, input().split())

start, goal = (1, 1), (H, W)
# 左右上下
directions = [[0,-1],[0,1],[-1,0],[1,0]]
# 四方に一回り大きいフィールドを作る
field = [['*'] * (W+2) for i in range(H+2)]
for i in range(1,H+1):
    row = list(input())[:-1]
    for j in range(1, W+1):
        field[i][j] = row[j-1]

# 移動距離メモ
memo = [[float('inf')] * (W+2) for i in range(H+2)]
memo[start[0]][start[1]] = 0
# 訪問済メモ
visited = [[0] * (W+2) for i in range(H+2)]
# 移動元メモ
prevs = [[None] * (W+2) for i in range(H+2)]

def bfs():
    que = deque()
    # スタート位置
    que.append(start)
    while len(que) != 0:
        cur = que.popleft()
        # 見終わった場所はもうやらない
        if visited[cur[0]][cur[1]] == 1:
            continue
        # 訪問済にする
        visited[cur[0]][cur[1]] = 1
        # 4方向見る
        for direction in directions:
            # 上下左右に1つずつ動かした座標
            nxt = list(map(lambda x,y: x+y, cur, direction))
            # 壁か黒マスはスキップ
            if field[nxt[0]][nxt[1]] == '*' or field[nxt[0]][nxt[1]] == '#':
                continue
            # 現在位置+1の距離を移動先に記録
            if field[nxt[0]][nxt[1]] == '.' and memo[nxt[0]][nxt[1]] > memo[cur[0]][cur[1]] + 1:
                memo[nxt[0]][nxt[1]] = memo[cur[0]][cur[1]] + 1
                # ここで移動元のマスを記録
                prevs[nxt[0]][nxt[1]] = (cur[0], cur[1])
                # キューに次のマスを足す
                que.append(nxt)

# 最短経路を出す
bfs()
# ゴールに辿り着けない場合
if memo[goal[0]][goal[1]] == float('inf'):
    print(-1)
    exit()
# 最短経路で使うマスに目印をつける
field[goal[0]][goal[1]] = '@'
nxt = prevs[goal[0]][goal[1]]
for _ in range(memo[goal[0]][goal[1]]):
    field[nxt[0]][nxt[1]] = '@'
    nxt = prevs[nxt[0]][nxt[1]]
# 残っている白マスが塗り替えられる場所
ans = 0
for i in range(1,H+1):
    for j in range(1, W+1):
        if field[i][j] == '.':
            ans += 1
print(ans)
