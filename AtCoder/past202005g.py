"""
・グリッドBFS
・斜め方向の追加。マイナス座標が面倒なので全部正の数で扱う。
　あと無限に広がるグリッドなので1だけ広めに作って、外を通れるようにする。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def bfs(grid, src):
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1))
    dist = list2d(H, W, INF)
    que = deque()
    for h, w in src:
        que.append((h, w))
        dist[h][w] = 0
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if h2 < 0 or w2 < 0 or h2 >= H or w2 >= W:
                continue
            if grid[h2][w2]:
                continue
            if dist[h2][w2] != INF:
                continue
            dist[h2][w2] = dist[h][w] + 1
            que.append((h2, w2))
    return dist

grid = list2d(403, 403, 0)
N, X, Y = MAP()
X += 201; Y += 201
for i in range(N):
    x, y = MAP()
    x += 201; y += 201
    grid[x][y] = 1

res = bfs(grid, [(201, 201)])
ans = res[X][Y]
if ans == INF:
    print(-1)
else:
    print(ans)
