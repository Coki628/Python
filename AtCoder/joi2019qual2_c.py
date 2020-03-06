# -*- coding: utf-8 -*-

"""
・自力AC
・x + 桁和(x) の答えは必ずxより大きくなるので、小さい方からDPで自身の通り数を遷移させていく。
・計算量は100万のNに桁和の計算で、pythonAC1.8秒、pypyAC0.7秒。
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

def digit_sum(a):
    s = str(a)
    res = 0
    for c in s:
        res += int(c)
    return res

N = INT()

# dp[i] := x + 桁和(x) を繰り返して、iになりうる通り数
dp = [1] * (N+1)
for i in range(1, N):
    nxt = i + digit_sum(i)
    if nxt <= N:
        dp[nxt] += dp[i]
print(dp[N])
