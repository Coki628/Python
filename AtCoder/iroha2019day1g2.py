# -*- coding: utf-8 -*-

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

N,M,K=MAP()
A=LIST()

memo=list3d(N+1, M+1, K+1, -INF)
def rec(cur, gap, m, sm):
    if m>M:
        return -INF
    if gap>=K:
        return -INF
    if memo[cur][m][gap]>=sm:
        return memo[cur][m][gap]
    else:
        memo[cur][m][gap]=sm
    if cur==N:
        return sm
    res=-INF
    res=max(res, rec(cur+1, gap+1, m, sm))
    res=max(res, rec(cur+1, 0, m+1, sm+A[cur]))
    return res

res=rec(0, 0, 0, 0)
if res!=-INF:
    print(res)
else:
    print(-1)
