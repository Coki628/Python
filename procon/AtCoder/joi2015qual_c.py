# -*- coding: utf-8 -*-

"""
・numpy実装
・計算量H^2*HW=50^4=625万はpythonだと不安な物量だけど、
　numpyでHWの数え上げを一括処理。0.2秒で安心のAC。
"""

import sys
import numpy as np

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

H, W = MAP()
grid = np.zeros((H, W), dtype=np.int8)
for i in range(H):
    for j, w in enumerate(list(input())):
        if w == 'B':
            grid[i,j] = 1
        elif w == 'R':
            grid[i,j] = 2

def check(b, r):
    cnt = 0
    # 条件を満たすマスの数を、2次元まとめて数える
    cnt += np.count_nonzero(grid[:b] != 0)
    cnt += np.count_nonzero(grid[b:r] != 1)
    cnt += np.count_nonzero(grid[r:] != 2)
    return cnt

ans = INF
for i in range(1, H):
    for j in range(i+1, H):
        ans = min(ans, check(i, j))
print(ans)
