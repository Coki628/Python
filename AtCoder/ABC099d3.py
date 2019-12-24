# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・前計算、全探索
"""

import sys
from itertools import combinations, permutations

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

N, C = MAP()
table = build_grid(C, C, 0, int)
grid = build_grid(N, N, 0, int)

# mod3=0,1,2と色毎で数を集計
cnts = list2d(3, C, 0)
for i in range(N):
    for j in range(N):
        c = grid[i][j] - 1
        cnts[(i+j)%3][c] += 1

# 全色から3色選んで、0,1,2どこに使うかを全通り試す
ans = INF
for comb in combinations(range(C), 3):
    for perm in permutations(comb):
        sm = 0
        for i, to in enumerate(perm):
            for frm in range(C):
                cnt = cnts[i][frm]
                sm += table[frm][to] * cnt
        ans = min(ans, sm)
print(ans)
