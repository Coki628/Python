# -*- coding: utf-8 -*-

"""
幅優先(TLEだけど…)
"""

from collections import deque

R,C,K = map(int, input().split())
field = []
for i in range(R):
    field.append(list(input()))

# 左右上下
directions = [[0,-1],[0,1],[-1,0],[1,0]]
# 移動距離制限
limit = K-1

def bfs(y, x):
    # 移動距離メモ
    memo = [[-1] * C for i in range(R)]
    que = deque()
    # スタート位置設定
    que.append([y, x])
    memo[y][x] = 0
    if field[y][x] == 'x':
        return False
    # キューが空になるまでやる(BFS)
    while len(que) != 0:
        cur = que.popleft()
        # 一定距離以上遠くまでは見ない
        if memo[cur[0]][cur[1]] >= limit:
            continue
        # 4方向見る
        for direction in directions:
            nxt = list(map(lambda x,y: x+y, cur, direction))
            # xがあったらこの位置は無理
            if field[nxt[0]][nxt[1]] == 'x':
                return False
            # 未訪問の位置のみ実施
            if memo[nxt[0]][nxt[1]] == -1:
                que.append(nxt)
                # 現在位置+1の距離を移動先に記録
                memo[nxt[0]][nxt[1]] = memo[cur[0]][cur[1]] + 1
    # 最後までいけたらOK
    return True

ans = 0
for i in range(K-1, R-K+1):
    for j in range(K-1, C-K+1):      
        if bfs(i, j):
            ans += 1
print(ans)
