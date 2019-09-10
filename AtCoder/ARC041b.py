# -*- coding: utf-8 -*-

"""
・自力AC
・グリッド、実装頑張る系
・ループ前後から交互に回すのが結構どうやろうか悩んだ。
　→解説見た。てか、別に普通に前からやるだけでも良かったじゃんこれ。
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
grid = [None] * H
ans = list2d(H, W, 0)
for i in range(H):
    grid[i] = list(map(int, input()))

for k in range(H):
    # 先頭と末尾を交互にやる
    if k%2 == 0:
        i = k // 2
    else:
        i = (H*2-k) // 2
    for j in range(W):
        if grid[i][j] != 0:
            cnt = grid[i][j]
            # 下から来たやつ
            if k%2 == 0:
                grid[i][j] -= cnt
                grid[i+2][j] -= cnt
                grid[i+1][j-1] -= cnt
                grid[i+1][j+1] -= cnt
                ans[i+1][j] += cnt
            # 上から来たやつ
            else:
                grid[i][j] -= cnt
                grid[i-2][j] -= cnt
                grid[i-1][j-1] -= cnt
                grid[i-1][j+1] -= cnt
                ans[i-1][j] += cnt

for i in range(H):
    print(''.join(map(str, ans[i])))
