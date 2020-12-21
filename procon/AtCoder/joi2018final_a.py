# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・グリッド、その先に出てくる数を前計算しといて後でO(1)で使うやつ。
・900万のグリッドがpypyでも間に合わなくてC++でAC0.2秒。
　(制約1秒、ちなみにメモリも256MB制限のところ365MBも使ってたのでどのみちpypyダメだった。。)
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
grid = build_grid(H, W, '', str, space=0)

# あるマスから先にあるOとIの数を前計算しておく
cnto = list2d(H, W, 0)
for i in range(H):
    for j in range(W-1, 0, -1):
        cnto[i][j-1] = cnto[i][j] + (1 if grid[i][j] == 'O' else 0)
cnti = list2d(H, W, 0)
for j in range(W):
    for i in range(H-1, 0, -1):
        cnti[i-1][j] = cnti[i][j] + (1 if grid[i][j] == 'I' else 0)

ans = 0
for i in range(H):
    for j in range(W):
        # Jを見つけたら、前計算したO,Iと組み合わせる数を数える
        if grid[i][j] == 'J':
            ans += cnto[i][j] * cnti[i][j]
print(ans)
