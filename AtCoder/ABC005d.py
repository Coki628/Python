# -*- coding: utf-8 -*-

"""
・D自力AC！
・2次元累積和
・計算量はN^4=625万、pythonで1.6秒、pypyで0.3秒AC。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

def deepcopy(li): return [x[:] for x in li]

N = INT()
grid = list2d(N+2, N+2, 0)
for i in range(1, N+1):
    row = LIST()
    for j in range(1, N+1):
        grid[i][j] = row[j-1]

# 2次元累積和
acc = deepcopy(grid)
for i in range(1, N+1):
    for j in range(1, N+1):
        acc[i][j] += acc[i][j-1]
for j in range(1, N+1):
    for i in range(1, N+1):
        acc[i][j] += acc[i-1][j]

# mx[s] := 面積がsの時の最大値
mx = [0] * (N**2+1)
for y1 in range(1, N+1):
    for x1 in range(1, N+1):
        for y2 in range(y1, N+1):
            for x2 in range(x1, N+1):
                s = (x2-x1+1) * (y2-y1+1)
                # 各長方形の区間和から、面積毎に最大値を保持する
                mx[s] = max(mx[s], acc[y2][x2] - acc[y2][x1-1] - acc[y1-1][x2] + acc[y1-1][x1-1])
# 累積maxして間を埋める
for i in range(1, N**2+1):
    mx[i] = max(mx[i], mx[i-1])

Q = INT()
for i in range(Q):
    p = INT()
    print(mx[p])
