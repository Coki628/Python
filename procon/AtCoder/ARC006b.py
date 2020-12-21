# -*- coding: utf-8 -*-

"""
・まあさくっと自力AC
・実装疲れた。入力受け取り時の空白の扱いとか。
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

N, L = MAP()
grid = list2d(L+2, 2*N+1, ' ')
for i in range(1, L+2):
    if i == L+1:
        # strip()が空白を消すので別処理
        row = list(sys.stdin.readline())[:2*N-1]
    else:
        row = list(input())
    for j in range(1, 2*N):
        if row[j-1] == 'o':
            start = (i, j)
        grid[i][j] = row[j-1]

def rec(i, j):
    # 上まで着いたら位置を返す
    if i == 0:
        return ceil(j, 2)
    # あみだくじのルール通りに移動していく
    if grid[i][j] == 'o':
        return rec(i-1, j)
    elif grid[i][j] == '|':
        if grid[i][j-1] == '-':
            return rec(i-1, j-2)
        elif grid[i][j+1] == '-':
            return rec(i-1, j+2)
        else:
            return rec(i-1, j)

print(rec(start[0], start[1]))
