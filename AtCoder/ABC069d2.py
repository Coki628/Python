# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
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

H, W = MAP()
N = INT()
A = LIST()

# 使いやすいようにデータの持ち方を変える
B = []
for i, a in enumerate(A):
    for _ in range(a):
        B.append(i + 1)

grid = list2d(H, W, 0)
k = 0
for i in range(H):
    for j in range(W):
        # 左右ジグザグに行を進む
        if i % 2 == 1:
            j = W - j - 1
        grid[i][j] = B[k]
        k += 1

for i in range(H):
    print(*grid[i])
