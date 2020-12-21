# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2008/2009-yo-prob_and_sol/2009-yo-t4/review/2009-yo-t4-review.html
・自力ならず。。
・グリッド、DFSっぽく再帰探索(メモしない)
・メモ化再帰にしないことで、移動可能な全ルートをチェックできる。
・氷が割れた状態は、自分が再帰を始める前に割って、自分から先の再帰が全て終わった所で戻すとうまくいく。
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

W = INT()
H = INT()
grid = build_grid(H, W, 0, int, padding=1)

ans = 0
def rec(h, w, cnt):
    global ans
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    # 氷を割る
    grid[h][w] = 0
    for dh, dw in directions:
        h2 = h + dh
        w2 = w + dw
        # 行き先に氷があれば進む
        if grid[h2][w2]:
            rec(h2, w2, cnt+1)
    # 氷を戻す
    grid[h][w] = 1
    ans = max(ans, cnt + 1)
    return

# 開始位置は全箇所試す
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j]:
            rec(i, j, 0)
print(ans)
