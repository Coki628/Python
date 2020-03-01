# -*- coding: utf-8 -*-

"""
参考：https://tenka1.klab.jp/2014/explain/qualb_b.html
・自力ならず。。悔しい。
・文字列DP、貰うDP
・DP考えてたけど、うまく形にできなかった。
　公式解説のソースコード見て、方針理解したらすぐ作れた。うーん。。
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
M = len(S)
A = [input() for i in range(N)]

# dp[i] := 文字列Sのi文字目までの部分列を作れる通り数
dp = [0] * (M+1)
dp[0] = 1
for i in range(1, M+1):
    for s in A:
        n = len(s)
        if i-n < 0:
            continue
        # 作りたい文字列の直前n文字と、今回のsが一致していれば遷移できる
        if S[i-n:i] == s:
            dp[i] += dp[i-n]
    dp[i] %= MOD
print(dp[M])
