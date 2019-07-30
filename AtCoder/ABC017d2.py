# -*- coding: utf-8 -*-

"""
・1次元数え上げDP
・愚直解(TLE)
・配るDPでやると遷移がすごく分かりやすい。
・でもこれだと高速化できないので、貰うDPにする必要がある。
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
A=[INT() for i in range(N)]

# dp[i] := サプリiまで摂取した状態の通り数
dp=[0]*(N+1)
dp[0]=1
for i in range(N):
    visited=set()
    for j in range(i, N):
        if A[j] not in visited:
            dp[j+1]+=dp[i]
            dp[j+1]%=MOD
            visited.add(A[j])
        else:
            break
print(dp[N])
