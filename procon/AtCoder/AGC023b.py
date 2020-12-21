# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc023/editorial.pdf
・自力ならず。これは気付かなかった。。
・実は縦横同じだけ動かすと結果は同じ。
　よって片側だけやって最後に*Nすればいい。
・計算量300*300*150=1350万でpythonTLE,pypyAC0.7秒。
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
grid = build_grid(N, N, '', str, space=0)

def check(i, j):
    for k in range(N):
        for l in range(k+1, N):
            if grid[(k+i)%N][(l+j)%N] != grid[(l+i)%N][(k+j)%N]:
                return False
    return True

# 片側しかやらなくていい
cnt = 0
for i in range(N):
    if check(0, i):
        cnt += 1
print(cnt * N)
