# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc029
　　　https://qiita.com/drken/items/a5e6fe22863b7992efdb
・部分和問題のDPに帰着させる。
・minmaxDP(ほぼナップザックと同じ)
・計算量O(NM)
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

N = INT()
A = [INT() for i in range(N)]

M = sum(A)
# dp[i][j] := i個目まで考えた時、和をjにすることができるか
dp = list2d(N+1, M+1, False)
dp[0][0] = True
dp[N][M] = True
for i in range(1, N+1):
    for j in range(M+1):
        dp[i][j] = dp[i][j] | dp[i-1][j]
        if j >= A[i-1]:
            dp[i][j] = dp[i][j] | dp[i-1][j-A[i-1]]

# 総和の半分切り上げの中で一番小さい場所
for j in range(ceil(M, 2), M+1):
    if dp[N][j]:
        print(j)
        break
