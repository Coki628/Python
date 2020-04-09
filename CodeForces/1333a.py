# -*- coding: utf-8 -*-

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    H, W = MAP()

    grid = list2d(H, W, '')
    if H * W % 2 == 1:
        for i in range(H):
            for j in range(W):
                if (i+j) % 2 == 0:
                    grid[i][j] = 'B'
                else:
                    grid[i][j] = 'W'
    else:
        for i in range(H):
            for j in range(W):
                if (i+j) % 2 == 0:
                    grid[i][j] = 'W'
                else:
                    grid[i][j] = 'B'
        grid[0][0] = 'B'
    
    for i in range(H):
        print(''.join(grid[i]))
