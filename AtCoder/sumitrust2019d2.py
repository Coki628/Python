# -*- coding: utf-8 -*-

"""
・C++でACしたDP解。計算量1億2000万はやっぱりpypyでもTLE。
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

# dp[i][j][k] := 左からi番目まで見て、今j桁決まっていて、数字kが可能か
dp = list3d(N+1, 4, 1000, 0)
dp[0][0][0] = 1
for i in range(N):
    a = int(S[i])
    for j in range(4):
        for k in range(1000):
            # i番目を使わない
            dp[i+1][j][k] |= dp[i][j][k]
            # i番目を左から1桁目として使う
            if j == 0 and k+a*100 < 1000:
                dp[i+1][j+1][k+a*100] |= dp[i][j][k]
            # 2桁目として使う
            if j == 1 and k+a*10 < 1000:
                dp[i+1][j+1][k+a*10] |= dp[i][j][k]
            # 3桁目として使う
            if j == 2 and k+a < 1000:
                dp[i+1][j+1][k+a] |= dp[i][j][k]
ans = 0
# N個全部見て3桁使い終わった状態で、0〜999から可能な数を数える
for k in range(1000):
    ans += dp[N][3][k]
print(ans)
