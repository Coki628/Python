# -*- coding: utf-8 -*-

"""
・文字列みたいなDP
・ABC118dのマッチのやつと似た感じでやる。
・と思ったけど、他のみんなもっと簡単に貪欲して通してた。。
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

def compare(a, b):
    if a is None:
        return b
    if len(a) > len(b):
        return a
    if len(a) < len(b):
        return b
    return max(a, b)

for _ in range(INT()):
    N = INT()
    
    dp = [''] * (N+1)
    dp[0] = ''
    for i in range(N):
        if i+2 <= N:
            dp[i+2] = compare(dp[i+2], '1' + dp[i])
        if i+3 <= N:
            dp[i+3] = compare(dp[i+3], '7' + dp[i])
        dp[i] = ''
    print(dp[N])
