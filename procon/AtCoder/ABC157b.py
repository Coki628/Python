# -*- coding: utf-8 -*-

import sys
from collections import Counter

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

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
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

grid = build_grid(3, 3, 0, int)
D = {}
for i in range(3):
    for j in range(3):
        D[grid[i][j]] = (i, j)

N = INT()
res = list2d(3, 3, 0)
for _ in range(N):
    a = INT()
    if a in D:
        i, j = D[a]
        res[i][j] = 1

for i in range(3):
    if sum(res[i]) == 3:
        Yes()
        exit()
res = list(zip(*res))
for i in range(3):
    if sum(res[i]) == 3:
        Yes()
        exit()
if res[0][0] and res[1][1] and res[2][2] or res[0][2] and res[1][1] and res[2][0]:
    Yes()
else:
    No()
