# -*- coding: utf-8 -*-

"""
・自力AC！
・真ん中のjを固定して、i,jとj,kで組めるペア数を掛け合わせる。
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

# i,jで組めるペア
cnt1 = [0] * N
for i in range(N-1):
    a = A[i]
    for j in range(i+1, N):
        if a < A[j]:
            cnt1[j] += 1

# j,kで組めるペア
cnt2 = [0] * N
for j in range(N-2, -1, -1):
    a = A[j]
    for k in range(j+1, N):
        if a > A[k]:
            cnt2[j] += 1

# 両ペア数を掛け合わせる
ans = 0
for j in range(N):
    ans += cnt1[j] * cnt2[j]
print(ans)
