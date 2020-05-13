"""
・ほぼ自力AC
・多点スタートのグリッドBFS
・BFSやるよってのだけ、ツイートで見てしまっていたけど、グリッドの色の変化の様子を観察すれば、
　多点スタートで最短距離取ればいいのもそこまで突飛な発想は必要なかった気がするし、
　本番中Dじゃなくてこっちやれば通せてたかもしれんなぁ。。
・ところでメモリ制限がなぜか半分の128GBでギリギリのACだった。
　グリッドは01だからllじゃなくてbool使いなさいよっていう意味だったのかもしれないけど、
　その区別のないpythonにはつらいとこだよなぁ。ただでさえpypyはデフォでメモリ多く食うし。。
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

def bfs(grid, src):
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dist = list2d(H, W, INF)
    que = deque()
    for h, w in src:
        que.append((h, w, 0))
        dist[h][w] = 0
    while que:
        h, w, c = que.popleft()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if grid[h2][w2] == -1:
                continue
            if dist[h2][w2] != INF:
                continue
            dist[h2][w2] = c + 1
            que.append((h2, w2, c+1))
    return dist

H, W, Q = MAP()
grid = build_grid(H, W, -1, int, space=0, padding=1)

src = []
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == grid[i+1][j] or grid[i][j] == grid[i-1][j] \
                or grid[i][j] == grid[i][j+1] or grid[i][j] == grid[i][j-1]:
            # 最初から色変化のある全てのマスが始点になる
            src.append((i, j))
# 最短距離が色が変わるまでの操作回数になる
res = bfs(grid, src)

for i in range(Q):
    h, w, p = MAP()

    if p <= res[h][w]:
        ans = grid[h][w]
    else:
        ans = (grid[h][w] + p - res[h][w]) & 1
    print(ans)
