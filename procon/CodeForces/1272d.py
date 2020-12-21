# -*- coding: utf-8 -*-

"""
・自力AC
・真ん中の1個を抜いてみて全箇所試すやつ。
・左右から累積和っぽい前計算をやっておけば、各箇所のチェックはO(1)で済む。
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

N = INT()
A = LIST()

# 左右から累積和っぽく、そこからいくつ増加列が連続するかを取っておく
B = [1] * N
for i, a in enumerate(A[1:], 1):
    if A[i-1] < a:
        B[i] = B[i-1] + 1
B2 = [1] * N
for i in range(N-2, -1, -1):
    if A[i] < A[i+1]:
        B2[i] = B2[i+1] + 1

ans = max(B)
for i in range(1, N-1):
    # 条件を満たすなら、間を抜いて前後を繋げてみる、を各箇所でやる
    if A[i-1] < A[i+1]:
        ans = max(ans, B[i-1] + B2[i+1])
print(ans)
