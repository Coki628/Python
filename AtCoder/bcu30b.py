# -*- coding: utf-8 -*-

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

# 2回り大きいグリッド
grid = list2d(13, 13, -1)
for i in range(2, 11):
    row = input()
    for j in range(2, 11):
        grid[i][j] = int(row[j-2])

# 縦横の確認
for i in range(2, 11):
    for j in range(10):
        if grid[i].count(j) >= 2:
            No()
            exit()
grid = list(zip(*grid))
for i in range(2, 11):
    for j in range(10):
        if grid[i].count(j) >= 2:
            No()
            exit()

# ナイトの動きを確認
for i in range(2, 11):
    for j in range(2, 11):
        x = grid[i][j]
        if x == grid[i-1][j-2] \
                or x == grid[i+1][j-2] \
                or x == grid[i-1][j+2] \
                or x == grid[i+1][j+2] \
                or x == grid[i-2][j-1] \
                or x == grid[i+2][j-1] \
                or x == grid[i-2][j+1] \
                or x == grid[i+2][j+1]:
            No()
            exit()
Yes()
