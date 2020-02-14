# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/armeria_betrue/status/1227633568643350529
　　　https://codeforces.com/contest/1303/submission/70872386
・文字列DP
・N^4の愚直解(TLE想定)
・このDPの考え方を元に高速化を図ることで正解できるはず。
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

def check(x):
    T1 = T[:x] + '*'
    T2 = T[x:] + '*'
    m1 = len(T1)
    m2 = len(T2)

    # dp[i][j][k] := Sのi文字目まで見て、T1のj文字とT2のk文字を作れるかどうか
    dp = list3d(N+1, m1, m2, 0)
    dp[0][0][0] = 1
    for i in range(N):
        s = S[i]
        for j in range(m1):
            for k in range(m2):
                if dp[i][j][k]:
                    # この文字sは使わない
                    dp[i+1][j][k] = 1
                    # この文字sを使う
                    if T1[j] == s:
                        # T1で使う
                        dp[i+1][j+1][k] = 1
                    if T2[k] == s:
                        # T2で使う
                        dp[i+1][j][k+1] = 1
    return dp[N][m1-1][m2-1]

for _ in range(INT()):
    S = input()
    T = input()

    N = len(S)
    M = len(T)
    # 前後半をどこで区切るかは全部試す
    for x in range(M):
        if check(x):
            YES()
            break
    else:
        NO()
