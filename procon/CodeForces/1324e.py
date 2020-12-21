# -*- coding: utf-8 -*-

"""
・自力AC、問題が分かりにくかったけど、理解したら方針はすぐ出た。
・ナップザック系、あるmodの範囲内で遷移を続けるDP
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, H, L, R = MAP()
A = LIST()

# dp[i][j] := i回目の睡眠まで終わって、時刻jの状態の時の最大値
dp = list2d(N+1, H, -INF)
dp[0][0] = 0
for i, a in enumerate(A):
    for h in range(H):
        # a,a-1時間後にもう1回寝る。その際[L,R]の範囲内であれば+1する。
        dp[i+1][(h+a)%H] = max(dp[i+1][(h+a)%H], dp[i][h] + (L <= (h+a)%H <= R)) 
        dp[i+1][(h+a-1)%H] = max(dp[i+1][(h+a-1)%H], dp[i][h] + (L <= (h+a-1)%H <= R))
ans = max(dp[N])
print(ans)
