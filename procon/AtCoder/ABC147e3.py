# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/armeria_betrue/status/1203681298826452993
・bitset高速化
・pythonは多倍長なので、int型変数の値をそのままビットとして使えばOK。
・今回は数万桁くらいある値が格納されることになるけど、全然速度は問題なかった。pythonAC0.1秒。
・boolでたくさんの状態管理しないといけない時、これは使えそうだ。
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
MAXV = 160 * 80

grid1 =build_grid(H, W, 0, int)
grid2 =build_grid(H, W, 0, int)
grid = list2d(H+1, W+1, 0)
for i in range(H):
    for j in range(W):
        grid[i+1][j+1] = abs(grid1[i][j]-grid2[i][j])

# dp[i][j] := グリッド(i, j)に到達可能な値の集合
dp = list2d(H+1, W+1, 0)
# MAXVビット動かした地点を0と考えると、途中で絶対値を考慮しなくて済む
dp[1][1] = (1<<MAXV-grid[1][1]) | (1<<MAXV+grid[1][1])
for i in range(1, H+1):
    for j in range(1, W+1):
        v = grid[i][j]
        # k+vの遷移
        dp[i][j] |= dp[i-1][j] << v
        dp[i][j] |= dp[i][j-1] << v
        # abs(k-v)の遷移
        dp[i][j] |= dp[i-1][j] >> v
        dp[i][j] |= dp[i][j-1] >> v
ans = INF
# 最終的に負数の部分は要らないので削る
S = dp[H][W] >> MAXV
for k in range(MAXV+1):
    if S & (1<<k):
        # 立っている1番小さいビットが答え
        ans = min(ans, k)
print(ans)
