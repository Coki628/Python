# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1287/submission/68266764
・なぎたさんの見た。
・全体-奇数で偶数の数も分かるので、添字に使うのは奇数だけでいい。
・これだとループ1つ減るので、制約N<=2000とかでも十分通ると思う。
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
A = LIST()

A = [a%2 if a != 0 else -1 for a in A]
M = ceil(N, 2)

# dp[i][j][k] := i番目まで見て、奇数をj回使って、直前が奇数or偶数(k=0,1)の状態の最小値
dp = list3d(N+1, M+1, 2, INF)
# 初期化：最初は直前が関係ないので、k=0,1両方を始点にする
dp[0][0][0] = dp[0][0][1] = 0
for i in range(N):
    a = A[i]
    for j in range(M+1):
        for k in range(2):
            # 今回の位置に奇数を使う遷移
            if (a == 1 or a == -1) and j+1 <= M:
                dp[i+1][j+1][1] = min(dp[i+1][j+1][1], dp[i][j][k] + (1^k))
            # 今回の位置に偶数を使う遷移
            if (a == 0 or a == -1):
                dp[i+1][j][0] = min(dp[i+1][j][0], dp[i][j][k] + (0^k))
ans = min(dp[N][M][0], dp[N][M][1])
print(ans)
