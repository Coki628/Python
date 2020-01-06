# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・DP…？ってなって時間食ったけど、なんとか正解に辿り着けた。
・詰まるところ、区切りの4パターンどれかによって*2か*3するだけ。
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

N = INT()
grid = build_grid(2, N, '*', str, space=0, padding=1)

# 最初
if grid[1][1] == grid[2][1]:
    cnt = 3
else:
    cnt = 6
for i in range(2, N+1):
    # 直前が縦向きなら*2
    if grid[1][i-1] == grid[2][i-1]:
        cnt *= 2
    # 直前もここも横向きなら*3
    elif grid[1][i-1] != grid[2][i-1] and grid[1][i] != grid[2][i] and grid[1][i-1] != grid[1][i]:
        cnt *= 3
print(cnt % MOD)
