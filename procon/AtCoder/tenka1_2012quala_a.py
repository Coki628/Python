# -*- coding: utf-8 -*-

"""
・1次元DP+フラグ
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

# dp[i][j] := iヶ月後の子ども(j=0)と大人(j=1)の人数
dp = list2d(N+1, 2, 0)
dp[0][0] = 1
for i in range(N):
    # 今いる大人 + 子どもが大人になる
    dp[i+1][1] = dp[i][1] + dp[i][0]
    # 今いる大人が子どもを生む
    dp[i+1][0] = dp[i][1]

print(dp[N][0]+dp[N][1])
