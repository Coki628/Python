# -*- coding: utf-8 -*-

"""
・100万グリッドの2次元累積和で、制約0.5秒と64MB。
・絶対pythonTLE,pypyMLEだと思ったから、最初からC++で書いてAC。
・一応後からこっちでも書いたけどやっぱり予想通り。。。
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

H, W = MAP()
Q = INT()

grid = build_grid(H, W, '', str, space=0, padding=1)
accj = list2d(H+1, W+1, 0)
acco = list2d(H+1, W+1, 0)
acci = list2d(H+1, W+1, 0)
for i in range(1, H+1):
    for j in range(1, W+1):
        # 3種類別々に2次元累積和を作る
        if grid[i][j] == 'J':
            accj[i][j] = 1
        elif grid[i][j] == 'O':
            acco[i][j] = 1
        else:
            acci[i][j] = 1
for i in range(1, H+1):
    for j in range(1, W):
        accj[i][j+1] += accj[i][j]
        acco[i][j+1] += acco[i][j]
        acci[i][j+1] += acci[i][j]
for j in range(1, W+1):
    for i in range(1, H):
        accj[i+1][j] += accj[i][j]
        acco[i+1][j] += acco[i][j]
        acci[i+1][j] += acci[i][j]

ans = [0] * 3
for _ in range(Q):
    lh, lw, rh, rw = MAP()
    ans[0] = accj[rh][rw] - accj[rh][lw-1] - accj[lh-1][rw] + accj[lh-1][lw-1]
    ans[1] = acco[rh][rw] - acco[rh][lw-1] - acco[lh-1][rw] + acco[lh-1][lw-1]
    ans[2] = acci[rh][rw] - acci[rh][lw-1] - acci[lh-1][rw] + acci[lh-1][lw-1]
    print(*ans)
