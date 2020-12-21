# -*- coding: utf-8 -*-

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

def check(act, exp):
    cnt = 0
    for i in range(N):
        for j in range(N):
            if act[i][j] != exp[i][j]:
                cnt += 1
    return cnt

N = INT()
grid1 = build_grid(N, N, '', str, space=False)
exp = build_grid(N, N, '', str, space=False)

grid2 = rot90(grid1, N, N)
grid3 = rot90(grid2, N, N)
grid4 = rot90(grid3, N, N)

ans = min(
    check(grid1, exp),
    check(grid2, exp) + 1,
    check(grid3, exp) + 2,
    check(grid4, exp) + 1,
)
print(ans)
