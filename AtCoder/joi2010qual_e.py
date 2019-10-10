# -*- coding: utf-8 -*-

"""
・蟻本演習2-1-3、自力AC
・グリッドBFS
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
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

directions = ((1,0),(-1,0),(0,1),(0,-1))
ch = cw = 0

H, W, N = MAP()
grid = list2d(H+2, W+2, 'X')
for i in range(1, H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]
        if grid[i][j] == 'S':
            ch, cw = i, j

def bfs(grid, sh, sw, goal):

    visited = list2d(H+2, W+2, 0)
    que = deque()
    que.append((sh, sw, 0))

    while len(que):

        h, w, c = que.popleft()
        if grid[h][w] == goal:
            return h, w, c
        if visited[h][w]:
            continue
        visited[h][w] = 1

        for d in directions:
            h2, w2 = h + d[0], w + d[1]
            if grid[h2][w2] != 'X':
                que.append((h2, w2, c+1))

ans = 0
for i in range(1, N+1):
    res = bfs(grid, ch, cw, str(i))
    ch, cw = res[0], res[1]
    ans += res[2]
print(ans)
