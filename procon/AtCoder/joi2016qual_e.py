# -*- coding: utf-8 -*-

"""
・100万グリッドにメモ化再帰したら、python,pypyともMLE。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
grid = build_grid(H, W, INF, int, padding=1)

memo = list2d(H+2, W+2, -1)
directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
def rec(h, w):
    if memo[h][w] != -1:
        return memo[h][w]
    prev = cur = -1
    for dh, dw in directions:
        h2 = h + dh
        w2 = w + dw
        # 低い所にだけ流れる
        if grid[h2][w2] < grid[h][w]:
            cur = rec(h2, w2)
            # この先で複数箇所に流れる or ここから複数箇所に流れる ならOK
            if cur == 0 or (prev != -1 and prev != cur):
                memo[h][w] = 0
                return 0
            else:
                prev = cur
    # どこへも行けなかったら、このマスを到着地として記録
    if cur == -1:
        memo[h][w] = h*(W+2) + w
        return h*(W+2) + w
    # どこか一箇所に流れるなら、そこを記録
    else:
        memo[h][w] = cur
        return cur

for i in range(1, H+1):
    for j in range(1, W+1):
        rec(i, j)
ans = 0
for i in range(1, H+1):
    ans += memo[i].count(0)
print(ans)
