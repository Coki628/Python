# -*- coding: utf-8 -*-

"""
・参戦中自力WA
・BFSで答えを埋めようとしたけど、その方針が良くなかった。。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def build_grid(H, W, intv, _type, space=True, padding=False):
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

def bfs(grid, sh, sw, i, x):
    directions = ((1,0),(-1,0),(0,1),(0,-1))
    que = deque()
    que.append((sh, sw))
    cost = 0
    while que:
        h1, w1 = que.popleft()
        if visited[h1][w1]:
            continue
        if cost == x and i != K:
            return (h1, w1)
        if cost != x or i == K:
            visited[h1][w1] = i
            if grid[h1][w1] == 'R':
                cost += 1
        for h, w in directions:
            h2 = h1 + h
            w2 = w1 + w
            if grid[h2][w2] == '*':
                continue
            else:
                que.append((h2, w2))
    return (-1, -1)

for _ in range(INT()):
    H, W, K = MAP()
    grid = build_grid(H, W, '*', str, space=False, padding=True)
    rcnt = 0
    for i in range(1, H+1):
        for j in range(1, W+1):
            if grid[i][j] == 'R':
                rcnt += 1
    p = rcnt // K
    q = rcnt % K
    sh = sw = 1
    visited = list2d(H+2, W+2, 0)
    for i in range(q):
        sh, sw = bfs(grid, sh, sw, i+1, p+1)
    for i in range(q, K):
        sh, sw = bfs(grid, sh, sw, i+1, p)

    ans = list2d(H, W, '')
    for i in range(1, H+1):
        for j in range(1, W+1):
            if visited[i][j] <= 10:
                ans[i-1][j-1] = str(visited[i][j]-1)
            elif 11 <= visited[i][j] <= 36:
                ans[i-1][j-1] = chr(visited[i][j]+86)
            else:
                ans[i-1][j-1] = chr(visited[i][j]+28)
    for i in range(H):
        print(''.join(ans[i]))
