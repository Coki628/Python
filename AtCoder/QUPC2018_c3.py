# -*- coding: utf-8 -*-

"""
初めての幅優先検索！改2
・これは人のやつ(てか解説)見た。
・BFSの始点は複数でもいいんだね！
・浅い場所から順に探すから、遠い方が先に着く事はないし、
　重なる場所は訪問済み扱いになるだけだから問題ないわけだ。
"""

from collections import deque
import numpy as np

H,W,X = map(int, input().split())

field = [[None] * W for i in range(H)]
boars = []
start,goal = None,None
# 左右上下
directions = np.array([[0,-1],[0,1],[-1,0],[1,0]])

# 入力受け取りと始点終点イノシシの位置記録
for i in range(H):
    row = list(input())
    for j in range(W):
        field[i][j] = row[j]
        if row[j] == '@':
            boars.append(np.array([i,j]))
        elif row[j] == 'S':
            start = np.array([i,j])
        elif row[j] == 'G':
            goal = np.array([i,j])

# 移動距離メモ
memo = [[-1] * W for i in range(H)]
que = deque()
# 各イノシシの守備範囲確定
for i in range(len(boars)):
    # 複数の始点を同時にキューに入れていい
    que.append(boars[i])
    memo[boars[i][0]][boars[i][1]] = 0

# キューが空になるまでやる(BFS)
while len(que) != 0:
    cur = que.popleft()
    # 移動距離がXに達したらそのマスから先はない 
    if memo[cur[0]][cur[1]] >= X:
        continue
    # 4方向見る
    for direction in directions:
        nxt = cur + direction
        # 壁はスキップ
        if field[nxt[0]][nxt[1]] == '#':
            continue
        # ゴールが安全でない
        if field[nxt[0]][nxt[1]] == 'G':
            print(-1)
            exit()
        # 未訪問の位置のみ実施
        if memo[nxt[0]][nxt[1]] == -1:
            que.append(nxt)
            # 現在位置+1の距離を移動先に記録
            memo[nxt[0]][nxt[1]] = memo[cur[0]][cur[1]] + 1
            # イノシシの守備範囲にする
            field[nxt[0]][nxt[1]] = '@'

# 移動距離メモ
memo = [[-1] * W for i in range(H)]
que = deque()
# スタート位置
que.append(start)
memo[start[0]][start[1]] = 0
# キューが空になるまでやる(BFS)
while len(que) != 0:
    cur = que.popleft()
    # 4方向見る
    for direction in directions:
        nxt = cur + direction
        # 壁かイノシシはスキップ
        if field[nxt[0]][nxt[1]] == '#' or field[nxt[0]][nxt[1]] == '@':
            continue
        # ゴールなら結果出力
        if field[nxt[0]][nxt[1]] == 'G':
            print(memo[cur[0]][cur[1]] + 1)
            exit()
        # 未訪問の位置のみ実施
        if memo[nxt[0]][nxt[1]] == -1:
            que.append(nxt)
            # 現在位置+1の距離を移動先に記録
            memo[nxt[0]][nxt[1]] = memo[cur[0]][cur[1]] + 1
# たどり着けなかった場合
print(-1)