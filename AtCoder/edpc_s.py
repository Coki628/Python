# -*- coding: utf-8 -*-

"""
・さくっと自力AC！
・知ってればド典型の桁DPって感じ。
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

S = list(map(int, input()))
N = len(S)
D = INT()

# dp[i][j] := i桁目まで見て、桁和がmod Dでjの時の通り数
dp0 = list2d(N+1, D, 0)
dp1 = list2d(N+1, D, 0)
dp0[0][0] = 1
for i, a in enumerate(S):
    for j in range(D):
        if not dp0[i][j] and not dp1[i][j]:
            continue
        for k in range(10):
            if k < a:
                dp1[i+1][(j+k)%D] += dp0[i][j]
                dp1[i+1][(j+k)%D] += dp1[i][j]
            elif k == a:
                dp0[i+1][(j+k)%D] += dp0[i][j]
                dp1[i+1][(j+k)%D] += dp1[i][j]
            else:
                dp1[i+1][(j+k)%D] += dp1[i][j]
        dp0[i+1][j] %= MOD
        dp1[i+1][j] %= MOD
ans = (dp0[N][0] + dp1[N][0] - 1) % MOD
print(ans)
