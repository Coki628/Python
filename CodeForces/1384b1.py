"""
・自力AC
・グリッドBFS
・このB1は制約が小さいので愚直に全部見に行ける。
・全部の潮の状態をグリッドにする。上がってる時と下がってる時をフラグ管理しつつ、
　行ける方向を限定したグリッドBFSをやって、向こう岸に到達できるかチェックする。
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

def bfs(grid, src):
    from collections import deque

    H, W = len(grid), len(grid[0])
    visited = list3d(2, H, W, 0)
    que = deque()
    for h, w in src:
        que.append((0, h, w))
        que.append((1, h, w))
        visited[0][h][w] = 1
        visited[1][h][w] = 1
    while que:
        up, h, w = que.popleft()
        # 右端に到達できればOK
        if w == N-1:
            return True
        if h == 0:
            up = 1
        if h == K:
            up = 0
        # 今潮が上昇中か下降中かで行ける方向を変える
        if up:
            for dh, dw in [(1, 0), (1, 1)]:
                nh = h + dh
                nw = w + dw
                if visited[up][nh][nw]:
                    continue
                if grid[nh][nw] <= L:
                    visited[up][nh][nw] = 1
                    que.append((up, nh, nw))
        else:
            for dh, dw in [(-1, 0), (-1, 1)]:
                nh = h + dh
                nw = w + dw
                if visited[up][nh][nw]:
                    continue
                if grid[nh][nw] <= L:
                    visited[up][nh][nw] = 1
                    que.append((up, nh, nw))
    return False

for _ in range(INT()):
    N, K, L = MAP()
    A = LIST()

    # 潮の状態に応じた水位を2次元グリッドにマッピング
    grid = list2d(K+1, N, 0)
    for i in range(K+1):
        for j in range(N):
            grid[i][j] = A[j] + i

    src = []
    for i in range(K+1):
        if grid[i][0] <= L:
            src.append((i, 0))
    ok = False
    for x in range(K+1):
        if bfs(grid, src):
            Yes()
            break
    else:
        No()
