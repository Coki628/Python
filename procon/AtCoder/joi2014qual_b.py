# -*- coding: utf-8 -*-

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
M = INT()
T = [a-1 for a in LIST()]

A = list2d(M, N, 0)
for i in range(M):
    A[i] = [a-1 for a in LIST()]

ans = [0] * N
for i in range(M):
    t = T[i]
    for j in range(N):
        # ターゲットは自分に得点
        if j == t:
            ans[t] += 1
        else:
            # それ以外は当たっていれば自分、外れていればターゲットに得点
            if A[i][j] == t:
                ans[j] += 1
            else:
                ans[t] += 1
[print(a) for a in ans]
