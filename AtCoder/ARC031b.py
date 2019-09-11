# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・グリッドBFS
・移動距離ではなく通ったマスの数を数える。
"""

import sys
from collections import deque

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def deepcopy(li): return [x[:] for x in li]

H, W = 10, 10

directions = ((1,0),(-1,0),(0,1),(0,-1))
grid = list2d(H+2, W+2, 'x')
for i in range(1, H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

landcnt = 0
start = None
for i in range(1, 11):
    for j in range(1, 11):
        if grid[i][j] == 'o':
            start = (i, j)
            landcnt += 1

def bfs(grid, que):
    # 通った島マスの数
    cnt = 0
    # 訪問済メモ
    memo = [[-1]*(W+2) for i in range(H+2)]
    while len(que):
        h1, w1 = que.popleft()
        if memo[h1][w1] != -1:
            continue
        memo[h1][w1] = 1
        cnt += 1
        # 4方向見る
        for direction in directions:
            h2 = h1 + direction[0]
            w2 = w1 + direction[1]
            # 海マスは行けない
            if grid[h2][w2] == 'x':
                continue
            que.append((h2, w2))
    return cnt

for i in range(1, 11):
    for j in range(1, 11):
        if grid[i][j] == 'x':
            grid2 = deepcopy(grid)
            # 海マスを1つ島マスにする
            grid2[i][j] = 'o'
            cnt = bfs(grid2, deque([start]))
            # ある島マスから全陸地に到達できればOK
            if cnt == landcnt+1:
                YES()
                exit()
NO()
