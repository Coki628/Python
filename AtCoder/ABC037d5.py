# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-12、なんとか自力AC。色々迷走した。。
・DAG、メモ化再帰
・グリッドのDAG、依存性の管理が面倒なので、メモ化再帰がやっぱり良いと思った。
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
INF = float('inf')
MOD = 10 ** 9 + 7

directions = ((1,0),(-1,0),(0,1),(0,-1))
H, W = MAP()
grid = list2d(H+2, W+2, -1)
for i in range(1, H+1):
    row = LIST()
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

memo = list2d(H+2, W+2, -1)
def rec(h, w):
    if memo[h][w] != -1:
        return memo[h][w]
    res = 1
    for d in directions:
        h2 = h + d[0]
        w2 = w + d[1]
        if grid[h2][w2] > grid[h][w]:
            res += rec(h2, w2)
            res %= MOD
    memo[h][w] = res
    return res

ans = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        ans += rec(i, j)
        ans %= MOD
print(ans)
