"""
・なんとか自力AC！
・グリッドBFS、枝刈り
・結構苦しかったけど、考えた枝刈りが機能してくれた。。
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

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
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
    """ BFS(グリッド、重みなし) """
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dist = list2d(H, W, INF)
    que = deque()
    for h, w in src:
        que.append((h, w))
        dist[h][w] = 0
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            for k in range(1, K+1):
                h2 = h + dh*k
                w2 = w + dw*k
                if grid[h2][w2] == '@':
                    break
                # 枝刈り
                if dist[h2][w2] < dist[h][w] + 1:
                    break
                elif dist[h2][w2] != INF:
                    continue
                dist[h2][w2] = dist[h][w] + 1
                que.append((h2, w2))
    return dist

H, W, K = MAP()
x1, y1, x2, y2 = MAP()
grid = build_grid(H, W, '@', str, space=0, padding=1)

res = bfs(grid, [(x1, y1)])
ans = res[x2][y2]
if ans == INF:
    print(-1)
else:
    print(ans)
