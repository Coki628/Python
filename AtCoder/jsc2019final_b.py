# -*- coding: utf-8 -*-

"""
・自力AC
・グラフ、構築、実装
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

N = INT()
G1 = build_grid(N, N, 0, int, space=0)
G2 = build_grid(N, N, 0, int, space=0)

# 各yについて、どのxと接続済かを持っておく
Ys = list2d(N, N, 0)
for x in range(N):
    for y in range(N):
        if G1[x][y]:
            Ys[y][x] = 1
# 各zについて、どのxと接続したいかを持っておく
Zs = list2d(N, N, 0)
for x in range(N):
    for z in range(N):
        if G2[x][z]:
            Zs[z][x] = 1

ans = list2d(N, N, '0')
xtoz = list2d(N, N, 0)
for i in range(N):
    z = Zs[i]
    for j in range(N):
        y = Ys[j]
        # 繋ぎたくない x -> z がこのyに含まれていないか
        ok = True
        for x in range(N):
            if not z[x] and y[x]:
                ok = False
                break
        # 問題なければ辺を張る
        if ok:
            for x in range(N):
                if z[x] and y[x]:
                    ans[j][i] = '1'
                    # x -> z について、どこが接続済かを持っておく
                    xtoz[x][i] = 1
for z in range(N):
    for x in range(N):
        # 当初の希望(Zs)の中で、1つでも実際に繋がっていない場所あればNG
        if Zs[z][x] and not xtoz[x][z]:
            print(-1)
            exit()
for i in range(N):
    print(''.join(ans[i]))
