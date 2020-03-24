# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2017/2018-yo/2018-yo-t5-review.html
・公式解より
・グリッドDP
・距離の小さい方から埋めていくので、添字の順番変えた。
・確かにこっちのがlog乗らない分速いかなー、と思ったら意外な結果。
・pypy1.6→0.3秒まで速くなったけど、python1.6→2.0秒で遅くなった。。
・pypyがheapq強くないのがある気がするので、実はダイクストラのが速い解法だったのかも。
・試しに枝刈り入れたりしてみたんだけど、変化なしだったり若干遅くなったり。
　グリッドがせいぜい30*30しかないから、そこが削れても効果が薄いみたい。。
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

# dp[i][h][w] := (h, w)マスにいて、(1, 1)からの移動距離がiの時の最小コスト
dp = list3d(H*W, H+2, W+2, INF)
dp[0][1][1] = 0
directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
for i in range(H*W-1):
    for h in range(1, H+1):
        for w in range(1, W+1):
            if dp[i][h][w] == INF:
                continue
            for dh, dw in directions:
                nh, nw = h + dh, w + dw
                if grid[nh][nw] == INF:
                    continue
                cost = (i*2+1) * grid[nh][nw]
                dp[i+1][nh][nw] = min(dp[i+1][nh][nw], dp[i][h][w] + cost)
ans = INF
for i in range(H*W):
    ans = min(ans, dp[i][H][W])
print(ans)
