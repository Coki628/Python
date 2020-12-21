# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・グリッド、2次元累積和
・100万マスグリッドのループはpythonだとしんどい気がしたんだけど、意外と2秒かからずAC1.8秒。(制約6秒なので余裕)
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

W, H = MAP()
b, a = MAP()
grid = build_grid(H, W, 0, int, padding=1)
INF = 10 ** 8 + 7
for i in range(1, H+1):
    for j in range(1, W+1):
        # 使用できないマスはコストINFとみなす
        if grid[i][j] == -1:
            grid[i][j] = INF
for i in range(1, H+1):
    for j in range(1, W):
        grid[i][j+1] += grid[i][j]
for j in range(1, W+1):
    for i in range(1, H):
        grid[i+1][j] += grid[i][j]

ans = INF
for lh in range(1, H+1-a+1):
    rh = lh + a - 1
    for lw in range(1, W+1-b+1):
        rw = lw + b - 1
        # 2次元累積和から区間和取得
        cnt = grid[rh][rw] - grid[rh][lw-1] - grid[lh-1][rw] + grid[lh-1][lw-1]
        ans = min(ans, cnt)
print(ans)
