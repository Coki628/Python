# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・練習用に桁DP解の方もやっとく。
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
A = LIST()

M = max(A+[K]).bit_length()

bitcnt = [0] * M
for a in A:
    # ビット毎に立っている数を数える
    i = 0
    while (a>>i) > 0:
        bitcnt[i] += a >> i & 1
        i += 1

# dp[i][j] := i桁目まで見て、Kより小さいことが確定しているorしていない(j=0,1)場合の最大値
dp = list2d(M+1, 2, -INF)
dp[M][0] = 0
for i in range(M-1, -1, -1):
    bit = 1 << i
    for j in range(2):
        # 今回のビットを立てない
        if j == 0:
            # Kが立っていれば、今回でsmaller確定する
            if bit & K:
                dp[i][1] = max(dp[i][1], dp[i+1][0] + bit*bitcnt[i])
            else:
                dp[i][0] = max(dp[i][0], dp[i+1][0] + bit*bitcnt[i])
            dp[i][1] = max(dp[i][1], dp[i+1][1] + bit*bitcnt[i])
        # 今回のビットを立てる
        else:
            # まだsmaller確定前なら、Kも立っている場合のみ遷移できる
            if bit & K:
                dp[i][0] = max(dp[i][0], dp[i+1][0] + bit*(N-bitcnt[i]))
            dp[i][1] = max(dp[i][1], dp[i+1][1] + bit*(N-bitcnt[i]))
print(max(dp[0][0], dp[0][1]))
