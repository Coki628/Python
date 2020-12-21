"""
・結構さくっと自力AC
・グリッドBFS
・実装ちょっと重めだけど、考察考えると全然Cよりこっちのが楽だったけどまあ相性だろうか。
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
INF = 10 ** 19
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

def bfs(grid, src):
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dist = list2d(H, W, INF)
    que = deque()
    for h, w in src:
        if grid[h][w] == '#':
            return dist
        que.append((h, w))
        dist[h][w] = 0
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if grid[h2][w2] == '#':
                continue
            if dist[h2][w2] != INF:
                continue
            dist[h2][w2] = dist[h][w] + 1
            que.append((h2, w2))
    return dist

for _ in range(INT()):
    H, W = MAP()
    grid = build_grid(H, W, '#', str, space=0, padding=1)

    ng = False
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i in range(1, H+1):
        for j in range(1, W+1):
            if grid[i][j] == 'B':
                for di, dj in directions:
                    ii = i + di
                    jj = j + dj
                    if grid[ii][jj] == 'B':
                        continue
                    if grid[ii][jj] == 'G':
                        ng = True
                        break
                    grid[ii][jj] = '#'
                if ng:
                    break
        if ng:
            break
    if ng:
        No()
        continue

    res = bfs(grid, [(H, W)])
    for i in range(1, H+1):
        for j in range(1, W+1):
            if grid[i][j] == 'G' and res[i][j] == INF:
                ng = True
                break
        if ng:
            break
    if ng:
        No()
        continue
    Yes()
