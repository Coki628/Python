# -*- coding: utf-8 -*-

"""
・実装おつ系
・グリッド、階段状になる形を集めるので、ループの条件がややこしい。
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

def rot90(grid, H, W):
    """ グリッドを90度回転 """
    res = list2d(W, H, 0)
    for i in range(H):
        for j in range(W):
            res[j][H-i-1] = grid[i][j]
    return res

H, W = MAP()
grid = build_grid(H, W, '', str, space=False)
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'X':
            grid[i][j] = -1
        else:
            grid[i][j] = int(grid[i][j])

def check(y, x, k):
    res = 0
    for i in range(y, y+k):
        for j in range(x, x+k+(y-i)):
            if grid[i][j] == -1:
                return -1
            else:
                res += grid[i][j]
    return res

def solve(H, W):
    res = 0
    for y in range(H):
        for x in range(W):
            k = 1
            while y+k <= H and x+k <= W:
                res = max(res, check(y, x, k))
                k += 1
    return res

# 4方向から確認
ans = solve(H, W)
grid = rot90(grid, H, W)
ans = max(ans, solve(W, H))
grid = rot90(grid, W, H)
ans = max(ans, solve(H, W))
grid = rot90(grid, H, W)
ans = max(ans, solve(W, H))
print(ans)
