# -*- coding: utf-8 -*-

"""
・速攻自力AC
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
grid = [None] * N
for i in range(N):
    grid[i] = list(input())

# 行列逆転
grid = list(zip(*grid))

cnt = 0
for i in range(9):
    hold = False
    for j in range(N):
        if grid[i][j] == 'o':
            if not hold:
                cnt += 1
            hold = True
        else:
            hold = False
    for j in range(N):
        if grid[i][j] == 'x':
            cnt += 1
print(cnt)
