"""
・自力AC
・グリッドBFS
・実装おつ。BFS2回やる。1回目で逆側の空洞に着くまでのコストを確認して、
　2回目で逆側の空洞に接する壁面を全部見てさっきのコストから最小を取る。
・想定解の、両側の各位置全部からマンハッタン距離見るのは頭いいな、実装楽そう。
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
EPS = 10 ** -10

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

def bfs1(grid, src):
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
            h2 = h + dh
            w2 = w + dw
            if dist[h2][w2] != INF:
                continue
            if grid[h2][w2] == '.':    
                dist[h2][w2] = dist[h][w]
                que.appendleft((h2, w2))
            elif grid[h2][w2] == '#':
                dist[h2][w2] = dist[h][w] + 1
                que.append((h2, w2))
    return dist

def bfs2(grid, dist, src):
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    visited = list2d(H, W, 0)
    que = deque()
    for h, w in src:
        que.append((h, w))
        visited[h][w] = 1
    res = INF
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if visited[h2][w2]:
                continue
            if grid[h2][w2] == '.':    
                visited[h2][w2] = 1
                que.append((h2, w2))
            elif grid[h2][w2] == '#':
                visited[h2][w2] = 1
                res = min(res, dist[h2][w2])
    return res

W, H = MAP()
grid = build_grid(H, W, '*', str, space=0, padding=1)

done = False
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == '.':
            dist = bfs1(grid, [(i, j)])
            done = True
        if done:
            break
    if done:
        break

done = False
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == '.' and dist[i][j] != 0:
            ans = bfs2(grid, dist, [(i, j)])
            done = True
        if done:
            break
    if done:
        break
print(ans)
