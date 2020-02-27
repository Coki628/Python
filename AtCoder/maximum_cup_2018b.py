# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/maximum-cup-2018/editorial.pdf
・自力はならず。まさか5次元でメモするとは思わなかった。
・グリッドBFS、訪問済メモを工夫
・1つ前の向いている方向を覚えておくことで、次にどの向きに回転するか分かる。
・計算量15^4*4=20万くらい、メモの次元が深いのが心配だったけど、pythonAC0.2秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def list5d(a, b, c, d, e, f): return [[[[[f] * e for j in range(d)] for j in range(c)] for j in range(b)] for i in range(a)]
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
    h, w = src
    # 向いている方向 0:右, 1:左, 2:下, 3:上  
    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    que = deque([(0, 0, h, w, 2)])
    # memo[i][j][k][l][m] := 右からi人、左からj人落として、グリッド[k][l]にいて、今向いている方向がmの状態が到達可能か
    memo = list5d(A+1, B+1, H, W, 4, 0)
    while que:
        a, b, h, w, d = que.popleft()
        if a > A or b > B:
            continue
        if memo[a][b][h][w][d]:
            continue
        memo[a][b][h][w][d] = 1
        for i, (dh, dw) in enumerate(directions):
            h2 = h + dh
            w2 = w + dw
            if grid[h2][w2] == '#':
                continue
            # 直進
            if d == i:
                que.append((a, b, h2, w2, i))
            # 下 → 右, 上 → 左, 左 → 下, 右 → 上 (左回転で右の人が落ちる)
            if d == 2 and i == 0 \
                    or d == 3 and i == 1 \
                    or d == 1 and i == 2 \
                    or d == 0 and i == 3:
                que.append((a+1, b, h2, w2, i))
            # 上 → 右, 下 → 左, 右 → 下, 左 → 上 (右回転で左の人が落ちる)
            if d == 3 and i == 0 \
                    or d == 2 and i == 1 \
                    or d == 0 and i == 2 \
                    or d == 1 and i == 3:
                que.append((a, b+1, h2, w2, i))

    return any(memo[A][B][H-2][W-2][i] for i in range(4))

A, B = MAP()
H, W = MAP()
grid = build_grid(H, W, '', str, space=0)

ans = bfs(grid, (1, 1))
if ans:
    Yes()
else:
    No()
