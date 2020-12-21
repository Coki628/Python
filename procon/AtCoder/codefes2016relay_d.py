# -*- coding: utf-8 -*-

"""
・自力AC
・式変形
・ちょっと時間かかって面倒な感じになった。
　ちゃんと手早く導けるようになりたい。
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

a = INT()
b = INT()
c = INT()

grid = list2d(3, 3, 0)
grid[0][0] = a
grid[0][1] = b
grid[1][1] = c

# 3マスの合計を決め打ち
for sm in range(301):
    grid[0][2] = sm - a - b
    grid[2][2] = sm - a - c
    grid[2][1] = sm - b - c
    grid[1][2] = sm - grid[0][2] - grid[2][2]
    grid[1][0] = sm - grid[1][1] - grid[1][2]
    grid[2][0] = sm - grid[0][0] - grid[1][0]
    # 矛盾なくグリッドが構築できたらOK
    if grid[2][0] == sm - grid[1][1] - grid[0][2] \
            and grid[2][0] == sm - grid[2][1] - grid[2][2]:
        for i in range(3):
            print(*grid[i])
        exit()
