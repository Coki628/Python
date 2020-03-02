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

N, M = MAP()
A = LIST()
B = LIST()

grid = list2d(4, 7, ' ')
D = {}
# ピンの位置と番号の割り当て
D[0] = (0, 6)
D[1] = (3, 3)
D[2] = (2, 2)
D[3] = (2, 4)
D[4] = (1, 1)
D[5] = (1, 3)
D[6] = (1, 5)
D[7] = (0, 0)
D[8] = (0, 2)
D[9] = (0, 4)
# 倒せなかった状態で初期化
for i in range(10):
    h, w = D[i]
    grid[h][w] = 'x'
# 1投目
for a in A:
    h, w = D[a]
    grid[h][w] = '.'
# 2投目
for b in B:
    h, w = D[b]
    grid[h][w] = 'o'

for i in range(4):
    print(''.join(grid[i]))
