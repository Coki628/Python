# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・部分和DP(通り数)、平均を総和0で求める。
・危なかった。半分全列挙の失敗から頑張って頭切り替えた。
"""

import sys
from collections import defaultdict

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

N, K = MAP()
# 総和0を平均KとするためKを引く
A = [a-K for a in LIST()]

# dp[i][j] := i番目まで見て、和がjとなる通り数
dp = [defaultdict(int) for i in range(N+1)]
dp[0][0] = 1
for i, a in enumerate(A):
    for k, v in dp[i].items():
        dp[i+1][k] += v
        dp[i+1][k+a] += v
# 部分和が0になった通り数から、1つも選ばない分の1を引く
print(dp[N][0] - 1)
