# -*- coding: utf-8 -*-

"""
・さくっと自力AC！
・ゲーム木の探索、後退解析
・メモ化再帰、負けを押し付ける手があるかどうかを返していく。(戻り値がFalseならそのターンはTrue)
・今回は先手後手で行動に差がないので、ターンは区別しないで考えられる。
・さすがに遷移先100*100はメモ化再帰でも余裕でpythonAC0.03秒。
・見やすさのため、遷移不可の判定を遷移先でやるようにしたら、意外と速度も遅くならなかった。
　それならこっちでもいいよね。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

H, W = MAP()
grid = list2d(H+2, W+2, '#')
for i in range(1, H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

memo = list2d(H+2, W+2, -1)
def rec(i, j):
    if memo[i][j] != -1:
        return memo[i][j]
    # この場所には遷移してこれないので、遷移前が負け=>ここは勝ち
    if grid[i][j] == '#':
        return 1
    if i == H and j == W:
        return 0
    # 3方向のうち1つでも勝てる(負けを押し付ける)手があればOK
    if rec(i+1, j) == 0:
        memo[i][j] = 1
        return 1
    if rec(i, j+1) == 0:
        memo[i][j] = 1
        return 1
    if rec(i+1, j+1) == 0:
        memo[i][j] = 1
        return 1
    # なければこの盤面は負け
    memo[i][j] = 0
    return 0

if rec(1, 1):
    print('First')
else:
    print('Second')
