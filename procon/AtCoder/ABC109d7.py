# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC！
・ほぼ迷わず正しい方針に進めて、実装もスムーズだった。いい感じ。
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
grid = build_grid(H, W, 0, int)

ans = []
moving = False
for i in range(H):
    for j in range(W):
        # 各行ジグザグに進めば切れ間なく移動できる
        if i % 2 == 1:
            j = W - j - 1
        # 1つ前から持ってきてるなら置く
        if moving:
            grid[i][j] += 1
            moving = False
            t = (i, j)
            ans.append((s, t))
        # 奇数枚の場所に来たら動かす
        if grid[i][j] % 2 == 1:
            moving = True
            grid[i][j] -= 1
            s = (i, j)

print(len(ans))
for (h1, w1), (h2, w2) in ans:
    print(h1+1, w1+1, h2+1, w2+1)
