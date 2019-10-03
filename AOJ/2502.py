# -*- coding: utf-8 -*-

"""
・蟻本演習
・個数制限なしナップザック系、2次元DP+遷移でもう1ループ系
・計算量約400^3=約6400万は想定通りTLE。
・これから内側の遷移を高速化する。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
phrases = []
for i in range(N):
    l, r, p = MAP()
    phrases.append((l, r, p))
M = INT()
melody = [0] * M
for i in range(M):
    melody[i] = INT()

mxm = max(melody)
dp = list2d(N+1, mxm+1, 0)
for i in range(1, N+1):
    l, r, p = phrases[i-1]
    for j in range(mxm+1):
        dp[i][j] = max(dp[i][j], dp[i-1][j])
        for k in range(max(0, j-r), max(0, j-l+1)):
            # 左上だけでなく、左からも遷移させる
            dp[i][j] = max(dp[i][j], dp[i-1][k]+p, dp[i][k]+p)

ans = [0] * M
for i, mel in enumerate(melody):
    ans[i] = dp[N][mel]
if min(ans) == 0:
    print(-1)
else:
    [print(a) for a in ans]
