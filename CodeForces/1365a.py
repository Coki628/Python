"""
参考：https://twitter.com/koboshi_kyopro/status/1269673238533033984
　　　https://twitter.com/titia_til/status/1269674174802345986
・自力ならず。。
・端的に言うと、「全部0の行か列小さい方の偶奇で勝敗が決まる」だけらしい。
　どうしてそうなるんだ。。。
→誤読だった。置けなくなるの、上下左右1マスずつかと思ってた。
　1つでも1がある行列は全て置けなくなるってことか、、それなら納得。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def build_grid(H, W, intv, _type, space=True, padding=False):
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
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

for _ in range(INT()):
    H, W = MAP()

    grid = build_grid(H, W, 0, int)
    row0 = 0
    for i in range(H):
        if not grid[i].count(1):
            row0 += 1
    grid = list(zip(*grid))
    col0 = 0
    for i in range(W):
        if not grid[i].count(1):
            col0 += 1
    if min(row0, col0) % 2 == 0:
        print('Vivek')
    else:
        print('Ashish')
