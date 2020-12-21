# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-13
・グリッドDP
・python2.3秒AC
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
INF = float('inf')
MOD = 10 ** 9 + 7

while True:

    H, W, N = MAP()
    if H == W == H == 0:
        break

    grid = list2d(H, W, 0)
    for i in range(H):
        row = LIST()
        for j in range(W):
            grid[i][j] = row[j]

    # dp[i][j] := (i, j)のマスをN回目の前までに何回通るか
    dp = list2d(H+1, W+1, 0)
    dp[0][0] = N - 1
    for i in range(H):
        for j in range(W):
            # 通る回数が偶数なら半分ずつ
            if dp[i][j] % 2 == 0:
                dp[i+1][j] += dp[i][j] // 2
                dp[i][j+1] += dp[i][j] // 2
            else:
                # 通る回数が奇数なら、元のグリッドの状態に応じて1回多く行く方が決まる
                if grid[i][j] == 0:
                    dp[i+1][j] += dp[i][j] // 2 + 1
                    dp[i][j+1] += dp[i][j] // 2
                else:
                    dp[i+1][j] += dp[i][j] // 2
                    dp[i][j+1] += dp[i][j] // 2 + 1

    # N-1回終了時の状態にグリッドを更新
    for i in range(H):
        for j in range(W):
            grid[i][j] ^= dp[i][j]
            grid[i][j] &= 1

    # N回目の散歩をシミュレーション
    i = j = 0
    while i < H and j < W:
        if grid[i][j] == 0:
            i += 1
        else:
            j += 1
    print(i+1, j+1)
