# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・DP、文字列比較を工夫する。
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

tocnt = {
    1: 2,
    2: 5,
    3: 5,
    4: 4,
    5: 5,
    6: 6,
    7: 3,
    8: 7,
    9: 6,
}

def compare(a, b):
    if a is None:
        return b
    if len(a) > len(b):
        return a
    if len(a) < len(b):
        return b
    return max(a, b)

N, M = MAP()
A = LIST()

# dp[i] := マッチをi本使った状態での最大値
dp = [None] * (N+1)
dp[0] = ''
for i in range(N):
    if dp[i] is None:
        continue
    for j in A:
        cnt = tocnt[j]
        if i+cnt <= N:
            dp[i+cnt] = compare(dp[i+cnt], dp[i] + str(j))
print(dp[N])
