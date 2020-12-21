# -*- coding: utf-8 -*-

"""
・部分和DP、取る時は1つ飛ばして遷移させる。
・部分点獲得
・調査のため、数列を見やすくするようにdictをlistにした版
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

N = 88

def get_piano(a, b):
    dp = [0] * N
    dp[0] = a
    dp[1] = b
    for i in range(2, N):
        dp[i] = dp[i-1] + dp[i-2]
    return dp

def get_sum(A, x):
    dp = list2d(N+2, x+1, 0)
    dp[0][0] = 1
    for i in range(N+1):
        if 2 <= i <= N and A[i-1] > x:
            return dp[i]
        for j in range(x+1):
            if dp[i][j] != 0:
                dp[i+1][j] += dp[i][j]
                if i != N and j+A[i] <= x:
                    dp[i+2][j+A[i]] += dp[i][j]
    return dp[N+1]

for _ in range(INT()):
    a, b, x = MAP()
    piano = get_piano(a, b)
    res = get_sum(piano, x)
    print(res[x])
