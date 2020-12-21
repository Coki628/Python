"""
・さくっと自力AC！
・グリッドBFS
・4方向への到達可能な条件を丁寧に整えたら、あとはBFSするだけ。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

def build_grid(H, W, intv, _type, space=False, padding=True):
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
    dist = list2d(H, W, 0)
    que = deque()
    for h, w in src:
        que.append((h, w))
        dist[h][w] = 0
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            nh = h + dh
            nw = w + dw
            if grid[nh][nw] > grid[h][w]+1 or grid[nh][nw] < grid[h][w]-1:
                continue
            if dist[nh][nw]:
                continue
            dist[nh][nw] = 1
            que.append((nh, nw))
        # 2つ先へ行く方
        for dh, dw in directions:
            nh = h + dh
            nw = w + dw
            if grid[nh][nw] > grid[h][w]:
                continue
            nh += dh
            nw += dw
            if grid[nh][nw] != grid[h][w]:
                continue
            if dist[nh][nw]:
                continue
            dist[nh][nw] = 1
            que.append((nh, nw))
    return dist

H, W = MAP()
sh, sw, gh, gw = MAP()
grid = build_grid(H, W, 20, int)

res = bfs(grid, [(sh, sw)])
if res[gh][gw]:
    YES()
else:
    NO()
