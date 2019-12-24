# -*- coding: utf-8 -*-

"""
・bitset高速化
・3万*4=12万ループの内側で1000ビット使っても全然大丈夫。
・pypyAC0.3秒、pythonでもAC0.2秒。
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
S = input()

# # dp[i][j] := 左からi番目まで見て、今j桁決まっていて、可能な数字0〜999の集合
dp = list2d(N+1, 4, 0)
dp[0][0] = 1
# 不要な1000ビット以降を削るマスク
msk = (1<<1000) - 1
for i in range(N):
    a = int(S[i])
    for j in range(4):
        # i番目を使わない
        dp[i+1][j] |= dp[i][j]
        # i番目を左から1桁目として使う
        if j == 0:
            dp[i+1][j+1] |= dp[i][j]<<a*100
            dp[i+1][j+1] &= msk
        # 2桁目として使う
        if j == 1:
            dp[i+1][j+1] |= dp[i][j]<<a*10
            dp[i+1][j+1] &= msk
        # 3桁目として使う
        if j == 2:
            dp[i+1][j+1] |= dp[i][j]<<a
            dp[i+1][j+1] &= msk
ans = 0
# N個全部見て3桁使い終わった状態で、0〜999から可能な数を数える
S = dp[N][3]
for k in range(1000):
    if S & 1<<k:
        ans += 1
print(ans)
