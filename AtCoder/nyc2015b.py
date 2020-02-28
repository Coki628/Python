# -*- coding: utf-8 -*-

"""
・ソートしてナップザック系
・遷移の条件を工夫して、ダメな積み方をできないようにする。
・計算量はN<=1000のN^2なので100万だけど、pythonでも大丈夫でAC0.8秒。
・内側ループは外側より大きいのやる必要ないので枝刈りして約半分になったらAC0.4秒。
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
A = LIST(N)

# 小さい方から見る(上の段から決めていく)
A.sort()
# dp[i][j] := i個目の餅まで見て、j個選んだ状態での、重さの合計の最小値
dp = list2d(N+1, N+1, INF)
dp[0][0] = 0
for i, a in enumerate(A):
    for j in range(i+1):
        dp[i+1][j] = min(dp[i+1][j], dp[i][j])
        # 遷移元の重さの合計が、a(今回追加したい餅)より小さければ遷移できる
        if j+1 <= N and dp[i][j] < a:
            dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + a)
ans = 0
for j in range(1, N+1):
    if dp[N][j] != INF:
        ans += 1
    else:
        break
print(ans)
