# -*- coding: utf-8 -*-

"""
・蟻本演習2-1-2、自力AC
・グリッドでDFS、再帰の初回だけ別処理
・現在位置は1次元にして持たせた。
"""

import sys

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

# 8方向
directions = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1))
cnt = 0

def rec(cur, prev):

    global cnt
    h, w = cur // (W+2), cur % (W+2)
    if visited[h][w]:
        return
    visited[h][w] = 1
    # 初回のみ、島カウントを足す
    if prev == -1:
        cnt += 1

    for d in directions:
        h2, w2 = h + d[0], w + d[1]
        if grid[h2][w2]:
            rec(h2*(W+2)+w2, cur)
    return 

while True:

    W, H = MAP()
    if H == W == 0:
        break

    grid = list2d(H+2, W+2, 0)
    for i in range(1, H+1):
        row = LIST()
        for j in range(1, W+1):
            grid[i][j] = row[j-1]

    visited = list2d(H+2, W+2, 0)
    cnt = 0
    for i in range(1, H+1):
        for j in range(1, W+1):
            if grid[i][j]:
                rec(i*(W+2)+j, -1)
    print(cnt)
