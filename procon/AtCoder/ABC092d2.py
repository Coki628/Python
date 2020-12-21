# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・グリッド構築。まあこれは当時も割とすんなり通した気がする。
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

A, B = MAP()

# 上半分白、下半分黒
grid = list2d(100, 100, '.')
for i in range(50, 100):
    for j in range(100):
        grid[i][j] = '#'

A -= 1; B -= 1
# 黒の数だけ白い中に黒点
for i in range(0, 50, 2):
    for j in range(0, 100, 2):
        if B == 0:
            break
        grid[i][j] = '#'
        B -= 1
    else:
        continue
    break
# 白の数だけ黒い中に白点
for i in range(99, 49, -2):
    for j in range(0, 100, 2):
        if A == 0:
            break
        grid[i][j] = '.'
        A -= 1
    else:
        continue
    break

print(100, 100)
for i in range(100):
    print(''.join(grid[i]))
