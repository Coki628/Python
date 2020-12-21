# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day1/editorial-G.pdf
　　　https://atcoder.jp/contests/iroha2019-day1/submissions/5713292
・いわゆる普通のminmax系DP(2次元+内ループ)
・これは貰うDP実装
・この後のセグ木実装のためにNとMの添え字入れ替えた。
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

N,M,K=MAP()
A=[0]+LIST()

# dp[i][j] := j日目にi回使った状態
dp=list2d(M+1, N+1, -INF)
dp[0][0]=0
for i in range(1, M+1):
    for j in range(1, N+1):
        for k in range(max(j-K, 0), j):
            # K日前から今までの範囲内で全ての日から遷移してみる
            dp[i][j]=max(dp[i][j], dp[i-1][k]+A[j])

# M回終わって、あとK日で終わりまで行ければOK
ans=-INF
for i in range(N+1-K, N+1):
    ans=max(ans, dp[M][i])
if ans!=-INF:
    print(ans)
else:
    print(-1)
