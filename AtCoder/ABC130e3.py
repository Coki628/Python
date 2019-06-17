# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=ERZuLAxZffQ
・LCSっぽいものを、最長じゃなくて、数え上げる。
・別解。添字を1つ増やして、数え上げをi方向とj方向で重複がないように遷移させる。
・添字が0ならi方向、1ならj方向に遷移させる。
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
S=LIST()
T=LIST()

# dp[i][j][k] := Sをi文字目、Tをj文字目まで考えた時、kが0ならi方向、1ならj方向に遷移させる
dp=list3d(N+2, M+2, 2, 0)
dp[0][0][0]=1
for i in range(N+1):
    for j in range(M+1):
        dp[i][j][0]%=MOD
        # 数え上げをi方向とj方向で重複がないように遷移させる
        dp[i+1][j][0]+=dp[i][j][0]
        dp[i][j][1]+=dp[i][j][0]
        dp[i][j+1][1]+=dp[i][j][1]
        # 今回増やす数値が一致するなら
        if i<N and j<M and S[i]==T[j]:
            # 斜めに遷移させる(1→0)
            dp[i+1][j+1][0]+=dp[i][j][1]
print((dp[N][M][1])%MOD)
