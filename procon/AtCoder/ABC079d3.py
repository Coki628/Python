# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・ワーシャルフロイド
"""

import sys

from scipy.sparse.csgraph import floyd_warshall

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
G = build_grid(10, 10, 0, int)
grid = build_grid(H, W, 0, int)

# 全頂点(数字0~9)間の最短距離
wf = floyd_warshall(G)

ans = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] != -1:
            a = grid[i][j]
            # 数字aを1に変える最小コスト
            ans += int(wf[a,1])
print(ans)
