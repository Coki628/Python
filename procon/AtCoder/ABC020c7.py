# -*- coding: utf-8 -*-
 
"""
ダイクストラと二分探索(優先度付キュー)
ちょっと見やすく整理した版
"""

# 優先度付キュー
from heapq import heappop, heappush
 
H,W,T = map(int, input().split())
 
start,goal = None,None
# 左右上下
directions = [[0,0,-1],[0,0,1],[0,-1,0],[0,1,0]]
# 四方に一回り大きいフィールドを作る
field = [['*'] * (W+2) for i in range(H+2)]
 
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        field[i][j] = row[j-1]
        if row[j-1] == 'S':
            start = (0, i, j)
        if row[j-1] == 'G':
            goal = (i, j)

# キューが空になるまでやる(BFS)
def bfs(x):
    # 移動距離メモ
    memo = [[float('inf')] * (W+2) for i in range(H+2)]
    # 訪問済メモ
    visited = [[False] * (W+2) for i in range(H+2)]
    # スタート位置
    que = [start]
    memo[start[1]][start[2]] = 0
 
    while len(que) != 0:
        cur_t, cur_x, cur_y = heappop(que)
        # 見終わった場所はもうやらない
        if visited[cur_x][cur_y]:
            continue
        # 訪問済
        visited[cur_x][cur_y] = True
        # 4方向見る
        for direction in directions:
            # pypy用にnumpy使わないで計算
            nxt_t, nxt_x, nxt_y = map(lambda x,y: x+y, (cur_t, cur_x, cur_y), direction)
            # 壁はスキップ
            if field[nxt_x][nxt_y] == '*':
                continue
            # 現在位置+1の距離を移動先に記録
            if field[nxt_x][nxt_y] == '.':
                dist = min(memo[cur_x][cur_y] + 1, memo[nxt_x][nxt_y])
                # 距離がT以下で訪問済ではない時だけ
                if dist <= T and not visited[nxt_x][nxt_y]:
                    memo[nxt_x][nxt_y] = dist
                    # 現在の経過時間をキューの優先度として、早い方から先に処理するようにする
                    nxt_t = dist
                    heappush(que, (nxt_t, nxt_x, nxt_y))
            # 現在位置+xの距離を移動先に記録
            elif field[nxt_x][nxt_y] == '#':
                dist = min(memo[cur_x][cur_y] + x, memo[nxt_x][nxt_y])
                if dist <= T and not visited[nxt_x][nxt_y]:
                    memo[nxt_x][nxt_y] = dist
                    nxt_t = dist
                    heappush(que, (nxt_t, nxt_x, nxt_y))
            # SかG
            else:
                dist = min(memo[cur_x][cur_y] + 1, memo[nxt_x][nxt_y])
                if dist <= T:
                    memo[nxt_x][nxt_y] = dist
 
    # 全部見たらゴール位置を返却
    return memo[goal[0]][goal[1]]

# 二分探索で最初にTを上回る場所を見つける
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
