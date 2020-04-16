# -*- coding: utf-8 -*-

"""
・ここまで自力AC
・グリッド
・こっちの制約小さい方はマス1つずつ愚直に確認できる。
"""

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

def build_grid(H, W, intv, _type, space=True, padding=False):
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
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
grid = build_grid(H, W, '#', str, space=0, padding=1)

ans = []
def check(i, j):
    sz = 1
    while grid[i+sz][j] == grid[i-sz][j] == grid[i][j+sz] == grid[i][j-sz] == '*':
        ok[i+sz][j] = ok[i-sz][j] = ok[i][j+sz] = ok[i][j-sz] = True
        sz += 1
    if sz > 1:
        ok[i][j] = True
        ans.append((i, j, sz-1))

def check2():
    for i in range(1, H+1):
        for j in range(1, W+1):
            if not ok[i][j]:
                return False
    return True

ok = list2d(H+2, W+2, True)
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == '*':
            ok[i][j] = False

for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == '*':
            check(i, j)

if check2():
    print(len(ans))
    [print(h, w, sz) for h, w, sz in ans]
else:
    print(-1)
