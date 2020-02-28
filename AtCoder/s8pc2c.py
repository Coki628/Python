# -*- coding: utf-8 -*-

"""
・自力ならず。添字のイメージは合ってたけど、遷移が作れなかった。
・桁DP系
・遷移のループは自分より左の桁に向かって遡っていく。
　遡った分だけ、分割しない桁が続いたことになる。
・計算量が100桁^2*値Dの範囲10万で1万*10万=10億で、だいたい遷移が半分程度のようなので
　実際は5億、制約5秒でもつらいかと思ったけどpypyAC2.1秒。
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

S = [0] + list(map(int, input()))
N = len(S) - 1
D = INT()

dp = list2d(N+1, D+1, 0)
dp[0][0] = 1
for i in range(1, N+1):
    for j in range(D+1):
        sm = 0
        for k in range(i):
            # 数値が0~Dの範囲で桁を遡っていく
            sm += 10**k * S[i-k]
            if j-sm < 0:
                break
            dp[i][j] += dp[i-k-1][j-sm]
        dp[i][j] %= MOD
print(sum(dp[N])%MOD)
