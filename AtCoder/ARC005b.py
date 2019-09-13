# -*- coding: utf-8 -*-

"""
・まあさくっと自力AC
・グリッド、やっぱ斜めもあると実装だるいわー。
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

X, Y, D = input().split()
X = int(X)
Y = int(Y)
grid = list2d(11, 11, '*')
for i in range(1, 10):
    row = list(input())
    for j in range(1, 10):
        grid[i][j] = row[j-1]

def rec(h, w, ans, d):
    ans += grid[h][w]
    if len(ans) == 4:
        return ans
    if d == 'L':
        if grid[h][w-1] == '*':
            return rec(h, w+1, ans, 'R')
        else:
            return rec(h, w-1, ans, d)
    elif d == 'R':
        if grid[h][w+1] == '*':
            return rec(h, w-1, ans, 'L')
        else:
            return rec(h, w+1, ans, d)
    elif d == 'U':
        if grid[h-1][w] == '*':
            return rec(h+1, w, ans, 'D')
        else:
            return rec(h-1, w, ans, d)
    elif d == 'D':
        if grid[h+1][w] == '*':
            return rec(h-1, w, ans, 'U')
        else:
            return rec(h+1, w, ans, d)
    elif d == 'LU':
        if grid[h-1][w] == grid[h][w-1] == '*':
            return rec(h+1, w+1, ans, 'RD')
        elif grid[h-1][w] == '*':
            return rec(h+1, w-1, ans, 'LD')
        elif grid[h][w-1] == '*':
            return rec(h-1, w+1, ans, 'RU')
        else:
            return rec(h-1, w-1, ans, d)
    elif d == 'LD':
        if grid[h+1][w] == grid[h][w-1] == '*':
            return rec(h-1, w+1, ans, 'RU')
        elif grid[h+1][w] == '*':
            return rec(h-1, w-1, ans, 'LU')
        elif grid[h][w-1] == '*':
            return rec(h+1, w+1, ans, 'RD')
        else:
            return rec(h+1, w-1, ans, d)
    elif d == 'RU':
        if grid[h-1][w] == grid[h][w+1] == '*':
            return rec(h+1, w-1, ans, 'LD')
        elif grid[h-1][w] == '*':
            return rec(h+1, w+1, ans, 'RD')
        elif grid[h][w+1] == '*':
            return rec(h-1, w-1, ans, 'LU')
        else:
            return rec(h-1, w+1, ans, d)
    elif d == 'RD':
        if grid[h+1][w] == grid[h][w+1] == '*':
            return rec(h-1, w-1, ans, 'LU')
        elif grid[h+1][w] == '*':
            return rec(h-1, w+1, ans, 'RU')
        elif grid[h][w+1] == '*':
            return rec(h+1, w-1, ans, 'LD')
        else:
            return rec(h+1, w+1, ans, d)

print(rec(Y, X, '', D))
