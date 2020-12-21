# -*- coding: utf-8 -*-

"""
・こどふぉ名物、ギャグ枠。数独って何だよ。。今回は通せたから良かったものの。
・各行列被んないような位置(自分ノートの図参照)を、数字1つずらす。
・これで通ったけど、本当はもっと簡単で良かったって話。。
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

for _ in range(INT()):
    grid = build_grid(9, 9, '', str, space=False)

    for i in range(3):
        for j in range(3):
            h = i * 3 + j
            w = j * 3 + i
            grid[h][w] = str(int(grid[h][w]) % 9 + 1)

    for i in range(9):
        print(''.join(grid[i]))
