# -*- coding: utf-8 -*-

"""
・自力AC
・各行後ろから見れば無駄なく塗れる。けどこの性質気づくのにちょっと時間かかった。
・より効率良くまとめた版。
・実際に塗らなくても、次行開始位置を適切に設定すればOK。
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
grid = build_grid(N, N, 'o', str, space=False, padding=True)

ans = 0
k = N
for i in range(1, N+1):
    # 各行後ろから見る
    for j in range(k, 0, -1):
        if grid[i][j] == '.':
            # 次行探索開始位置
            k = j - 1
            ans += 1
            break
    else:
        # 最後まで行ったら次行は最初から
        k = N
print(ans)
