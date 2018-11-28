
# -*- coding: utf-8 -*-

"""
幅優先と2分探索
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
    # 訪問済メモ
    visited = [[False] * (W+2) for i in range(H+2)]
    que = deque()
    # スタート位置
    que.append(start)
    memo[start[0]][start[1]] = 0

    while len(que) != 0:
        cur = que.popleft()
        # 見終わった場所はもうやらない
        if visited[cur[0]][cur[1]]:
            continue
        # 訪問済
        visited[cur[0]][cur[1]] = True
        # 4方向見る
        for direction in directions:
            # pypy用にnumpy使わないで計算
            nxt = list(map(lambda x,y: x+y, cur, direction))
            # 壁はスキップ
            if field[nxt[0]][nxt[1]] == '*':
                continue
            # 現在位置+1の距離を移動先に記録
            if field[nxt[0]][nxt[1]] == '.' or field[nxt[0]][nxt[1]] == 'G':
                dist = min(memo[cur[0]][cur[1]] + 1, memo[nxt[0]][nxt[1]])
                # 距離がT以下で訪問済ではない時だけ
                if dist <= T and not visited[nxt[0]][nxt[1]]:
                    memo[nxt[0]][nxt[1]] = dist
                    # ゴールはキューに足さなくていいはず
                    if field[nxt[0]][nxt[1]] != 'G':
                        que.append(nxt)
            # 現在位置+xの距離を移動先に記録
            elif field[nxt[0]][nxt[1]] == '#':
                dist = min(memo[cur[0]][cur[1]] + x, memo[nxt[0]][nxt[1]])
                if dist <= T and not visited[nxt[0]][nxt[1]]:
                    memo[nxt[0]][nxt[1]] = dist
                    que.append(nxt)
    # 全部見たらゴール位置を返却
    return memo[goal[0]][goal[1]]

# 2分探索で最初にTを上回る場所を見つける
hi = 10 ** 9
low = 0
# low+1 == hi -> lowとhiが隣り合った状態なので、境界が確定する
while low+1 < hi:
    mid = (hi+low) // 2
    if bfs(mid) > T:
        hi = mid
    else:
        low = mid
# hiの位置が最初にTを上回る場所、lowがTを超えない最大値
print(low)
