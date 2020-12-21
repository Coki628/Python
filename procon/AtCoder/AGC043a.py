# -*- coding: utf-8 -*-

"""
・なんとか自力AC！
・グリッドDP
・考察が整えば、後はシンプルにまとまる。
・長方形範囲への反転操作は、行えばその先同じ色の所にはどこまでもノーコストで行けるようになる。
・結果的に、操作回数が増えるのは、白→黒の移動をしたい時のみ。
・これがすぐに分かれば良かったんだけど、色々迷走して、通せたのは100分以上もの激戦の末、、遅いなぁ。。
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
grid = build_grid(H, W, '*', str, space=0, padding=1)

# dp[i][j] := マス(i, j)に到達するための最小コスト
dp = list2d(H+2, W+2, INF)
if grid[1][1] == '.':
    dp[1][1] = 0
else:
    dp[1][1] = 1
for i in range(H+1):
    for j in range(W+1):
        # 白から黒への移動のみコストがかかる
        if grid[i][j] == '.' and grid[i][j+1] == '#':
            dp[i][j+1] = min(dp[i][j+1], dp[i][j] + 1)
        else:
            dp[i][j+1] = min(dp[i][j+1], dp[i][j])  
        if grid[i][j] == '.' and grid[i+1][j] == '#':  
            dp[i+1][j] = min(dp[i+1][j], dp[i][j] + 1)
        else:
            dp[i+1][j] = min(dp[i+1][j], dp[i][j])
ans = dp[H][W]
print(ans)
