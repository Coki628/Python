# -*- coding: utf-8 -*-

"""
・300点自力AC
・数え上げDP
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

N,M=MAP()
broken=[False]*(N+1)
for i in range(M):
    a=INT()
    broken[a]=True

dp=[0]*(N+1)
dp[0]=1
for i in range(N):
    if not broken[i+1]:
        dp[i+1]+=dp[i]
        dp[i+1]%=MOD
    if i!=N-1 and not broken[i+2]:
        dp[i+2]+=dp[i]
        dp[i+2]%=MOD
print(dp[N])
