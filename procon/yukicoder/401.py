# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・グリッド、実装ちょっとだるいやつ。
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
EPS = 10 ** -10

N = INT()

grid = list2d(N+2, N+2, '*')
for i in range(1, N+1):
    for j in range(1, N+1):
        grid[i][j] = ''

def f(num, i, j, d):
    di, dj = d
    while grid[i+di][j+dj] == '':
        grid[i+di][j+dj] = str(num).zfill(3)
        num += 1
        i += di
        j += dj
    return num, i, j

directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
i = 0
h = w = 1
grid[h][w] = '001'
num = 2
while 1:
    tmp = num
    num, h, w = f(num, h, w, directions[i])
    if tmp == num:
        break
    i = (i+1) % 4

for i in range(1, N+1):
    print(*grid[i][1:-1])
