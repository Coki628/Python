"""
参考：https://twitter.com/laycrs/status/1258077665442983936
・自力ならず。。
・グリッドBFS、連結成分
・連結成分毎にするのがいいのは分かったけど、NG条件をうまくまとめられなかった。
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

def bfs(grid, src, grpnum):
    from collections import deque

    h, w = src
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    que = deque([(h, w)])
    dist[h][w] = grpnum
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if grid[h2][w2] == '.':
                continue
            if dist[h2][w2] != -1:
                continue
            dist[h2][w2] = grpnum
            que.append((h2, w2))
    return dist

H, W = MAP()
grid = build_grid(H, W, '.', str, space=0, padding=1)

# 連結成分(黒マスの領域)毎にまとめる
dist = list2d(H+2, W+2, -1)
grpnum = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == '#' and dist[i][j] == -1:
            bfs(grid, (i, j), grpnum)
            grpnum += 1

# ***###...###***があるかのチェック
def check1(grid, H, W):
    for i in range(1, H+1):
        flag = 0
        for j in range(1, W+1):
            if grid[i][j] == '#' and flag % 2 == 0:
                flag += 1
            if grid[i][j] == '.' and flag % 2 == 1:
                flag += 1
        if flag >= 3:
            return False
    return True

# #のない行(列)があるかのチェック
def check2(grid, H, W):
    for i in range(1, H+1):
        if not grid[i].count('#'):
            return True
    return False

# ***###...###***が1箇所でもあったらNG
if not check1(grid, H, W) or not check1(list(zip(*grid)), W, H):
    print(-1)
    exit()

# 行列どちらかだけ#のない行(列)があるとNG
if check2(grid, H, W) ^ check2(list(zip(*grid)), W, H):
    print(-1)
    exit()

print(grpnum)
