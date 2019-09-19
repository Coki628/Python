# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/ttpc2019/editorial.pdf
・グリッドで構築
・初期状態を連番で作って、全行列0(modN)から加減する。
・対角線を入れ替えれば、縦横両方を-i,+iできる。
　でもこんなんどうやって導き出せと…？
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

N = INT()

if N%2 == 0:
    No()
    exit()

grid = list2d(N, N, 0)
for i in range(N):
    for j in range(N):
        grid[i][j] = i*N + j + 1

for i in range(N//2):
    j = N - i - 1
    grid[i][i], grid[j][j] = grid[j][j], grid[i][i]

Yes()
for i in range(N):
    print(*grid[i])
