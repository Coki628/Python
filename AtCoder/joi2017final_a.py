# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・ソートして貪欲に。
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

N, K = MAP()
A = LIST(N)

A.sort()
B = []
for i in range(N-1):
    # 来客と来客の間隔
    B.append(A[i+1] - A[i] - 1)
# 大きい所から除きたい
B.sort(reverse=1)

# 1回も途中で消さない合計
ans = A[-1] - A[0] + 1
for i in range(K-1):
    # 2本目以降のマッチで消せるだけ消す
    ans -= B[i]
print(ans)
