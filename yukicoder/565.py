"""
・自力AC
・グリッドの回転、拡大
・回転はやったことあってライブラリ化してたからすぐだったけど、拡大は多分初めてやった。
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
EPS = 10 ** -10

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

def rot90(grid, H, W):
    """ グリッドを時計回りに90度回転 """
    res = list2d(W, H, 0)
    for i in range(H):
        for j in range(W):
            res[j][H-i-1] = grid[i][j]
    return res

R, K = MAP()
R //= 90
H, W = MAP()
grid = build_grid(H, W, '', str, space=0)

# 回転
for _ in range(R):
    grid = rot90(grid, H, W)
    H, W = W, H

# 拡大
ans = list2d(H*K, W*K, '')
for i in range(H*K):
    for j in range(W*K):
        ans[i][j] = grid[i//K][j//K]

for i in range(H*K):
    print(''.join(ans[i]))
