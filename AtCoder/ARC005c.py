# -*- coding: utf-8 -*-

"""
・今日の一問から
・きっちり自力AC
・グリッドで01BFS
"""

import sys
from collections import deque

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

H, W = MAP()
grid = list2d(H+2, W+2, '*')
for i in range(1, H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]
        if row[j-1] == 's':
            start = (i, j)
        elif row[j-1] == 'g':
            goal = (i, j)

def bfs01(grid, sh, sw):
    res = list2d(H+2, W+2, -1)
    directions = ((1,0),(-1,0),(0,1),(0,-1))
    que = deque()
    que.append((sh, sw, 0))
    while len(que):
        h1, w1, cost = que.popleft()
        if res[h1][w1] != -1:
            continue
        res[h1][w1] = cost
        for h, w in directions:
            h2 = h1 + h
            w2 = w1 + w
            # 範囲外
            if grid[h2][w2] == '*':
                continue
            # 現在コストが2未満ならまだ塀を突破できる
            elif grid[h2][w2] == '#' and cost < 2:
                # 後ろに詰める
                que.append((h2, w2, cost+1))
            elif grid[h2][w2] == '.' or grid[h2][w2] == 'g':
                # 塀のない場所は優先して進めるよう前に詰める
                que.appendleft((h2, w2, cost))
    return res

res = bfs01(grid, start[0], start[1])
if res[goal[0]][goal[1]] != -1:
    YES()
else:
    NO()
