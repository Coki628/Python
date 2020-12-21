# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・期待値、累積和
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
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def get_sum(a, b, c):
    """ 等差数列の和：(初項a, 末項b, 項数c) """
    return (a+b) * c // 2

N, K = MAP()
A = LIST()

B = [0] * N
for i in range(N):
    # 個別の期待値(普通に平均取る感じ)
    B[i] = get_sum(1, A[i], A[i]) / A[i]

# 累積和作って区間和取る
acc = [0] + list(accumulate(B))
ans = 0
for i in range(N-K+1):
    ans = max(ans, acc[i+K] - acc[i])
print(ans)
