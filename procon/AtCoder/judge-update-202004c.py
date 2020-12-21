# -*- coding: utf-8 -*-

"""
・終了後自力AC。バチャ中はWA。。
・また沼にはまった。。終了後、落ち着いてコードを確認してバグ発覚。
　置ける位置の設定間違えてただけ。
・後は順列全探索
"""

import sys
from itertools import permutations

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

a, b, c = MAP()
N = a + b + c

ok = list2d(3, 3, 0)
for i in range(3):
    if a > i:
        ok[i][0] = 1
    if b > i:
        ok[i][1] = 1
    if c > i:
        ok[i][2] = 1

def check(grid):
    for i in range(3):
        for j in range(3):
            if i != 0 and grid[i-1][j] > grid[i][j]:
                return False
            if j != 0 and grid[i][j-1] > grid[i][j]:
                return False
    return True

ans = 0
for perm in permutations(range(1, N+1)):
    grid = list2d(3, 3, N)
    idx = 0
    for i in range(3):
        for j in range(3):
            if ok[i][j]:
                grid[i][j] = perm[idx]
                idx += 1
    if check(grid):
        ans += 1
print(ans)
