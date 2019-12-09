# -*- coding: utf-8 -*-

"""
・自力TLE
・グリッドDP
・計算量80^3*160=8200万くらいで、pypyでもTLE。。
・コンテスト終了後、C++でAC確認。
・配るDPにして、ちょっと遷移元次第で枝刈りするようにしたけどTLE変わらず。
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

H, W = MAP()
MAXV = 160 * 80

grid1 =build_grid(H, W, 0, int)
grid2 =build_grid(H, W, 0, int)
grid = list2d(H+1, W+1, 0)
for i in range(H):
    for j in range(W):
        grid[i][j] = abs(grid1[i][j]-grid2[i][j])

# dp[i][j][k] := グリッド(i, j)に数値kで到達可能かどうか
dp = list3d(H+1, W+1, MAXV+1, 0)
dp[0][0][0] = 1
for i in range(H):
    for j in range(W):
        v = grid[i][j]
        for k in range(MAXV+1):
            # 枝刈り
            if (i+j+1) * 80 < k:
                break
            # 上からの遷移
            dp[i+1][j][abs(k-v)] |= dp[i][j][k]
            if k+v <= MAXV:
                dp[i+1][j][k+v] |= dp[i][j][k]
            # 左からの遷移
            dp[i][j+1][abs(k-v)] |= dp[i][j][k]
            if k+v <= MAXV:
                dp[i][j+1][k+v] |= dp[i][j][k]
ans = INF
for k in range(MAXV+1):
    if dp[H][W-1][k]:
        ans = min(ans, k)
    if dp[H-1][W][k]:
        ans = min(ans, k)
print(ans)
