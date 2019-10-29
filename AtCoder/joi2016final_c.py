# -*- coding: utf-8 -*-

"""
・蟻本演習3-1-3
・二分探索、最小値の最大化(最大値の最小化)
・盛大にWA。色々直したりしたけど、色々考慮できてないことが分かったりで終了。。
"""

import sys
from itertools import accumulate

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

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

H, W = MAP()
grid = [None] * H
for i in range(H):
    grid[i] = LIST()

# 各行列について、累積maxと累積minの差を取ったグリッドを構築
def build_absmx(H, W, grid):
    accmx = [None] * H
    accmn = [None] * H
    accmxrev = [None] * H
    accmnrev = [None] * H
    for i in range(H):
        accmx[i] = list(accumulate(grid[i], max))
        accmn[i] = list(accumulate(grid[i], min))
        accmxrev[i] = list(accumulate(grid[i][::-1], max))[::-1]
        accmnrev[i] = list(accumulate(grid[i][::-1], min))[::-1]
    absmx = list2d(H, W+1, 0)
    absmxrev = list2d(H, W+1, 0)
    for i in range(H):
        for j in range(W):
            absmx[i][j+1] = abs(accmx[i][j]-accmn[i][j])
            absmxrev[i][j] = abs(accmxrev[i][j]-accmnrev[i][j])
    absmx2 = list2d(H+2, W+3, INF)
    for i in range(H):
        for j in range(W+1):
            absmx2[i+1][j+1] = max(absmx[i][j], absmxrev[i][j])
    return absmx2

absmxrows = build_absmx(H, W, grid)
gridrev = list(zip(*grid))
absmxcols = build_absmx(W, H, gridrev)

def check(m):
    # 行か列どちらかでうまくいけばOK
    return check2(m, H+2, W+3, absmxrows) or check2(m, W+2, H+3, absmxcols)

def check2(m, H, W, grid):
    # 左上から右下へ向かう
    stack = []
    # 最上段で右端以外のマスが始点候補
    for i in range(W-3, 0, -1):
        if grid[1][i] <= m:
            stack.append((1, i))
    directions = ((0, 1), (1, 1), (1, 0))
    visited = list2d(H, W, 0)
    while stack:
        h, w = stack.pop()
        # 最下段で左端以外に到達したらOK
        if h == H-2 and w != 1:
            return True
        if visited[h][w]:
            continue
        visited[h][w] = 1
        for d1, d2 in directions:
            h2 = h + d1
            w2 = w + d2
            if grid[h2][w2] <= m:
                stack.append((h2, w2))
    # 右上から左下へ向かう
    stack = []
    # 最上段で左端以外のマスが始点候補
    for i in range(2, W-1):
        if grid[1][i] <= m:
            stack.append((1, i))
    directions = ((0, -1), (1, -1), (1, 0))
    visited = list2d(H, W, 0)
    while stack:
        h, w = stack.pop()
        # 最下段で右端以外に到達したらOK
        if h == H-2 and w != W-2:
            return True
        if visited[h][w]:
            continue
        visited[h][w] = 1
        for d1, d2 in directions:
            h2 = h + d1
            w2 = w + d2
            if grid[h2][w2] <= m:
                stack.append((h2, w2))
    # どちらの向きも辿り着かなかったらNG
    return False

print(bisearch_min(-1, 10**9, check))
