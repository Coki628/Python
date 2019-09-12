# -*- coding: utf-8 -*-

"""
・自力AC！
・2値化して累積和
"""

import sys
from itertools import accumulate

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

N, K = MAP()
A = [INT() for i in range(N)]

# 各2点間において上昇しているかどうか
B = [0] * (N-1)
for i in range(N-1):
    if A[i] < A[i+1]:
        B[i] = 1

# 累積和
acc = [0] + list(accumulate(B))
cnt = 0
for i in range(K-1, N):
    # 各K-1の長さの区間で、全て上昇(K-1)を達成しているかどうか
    sm = acc[i] - acc[i-(K-1)]
    if sm == K-1:
        cnt += 1
print(cnt)
