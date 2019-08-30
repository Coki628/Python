# -*- coding: utf-8 -*-

"""
参考：http://arc051.contest.atcoder.jp/data/arc/051/editorial.pdf
・実はフィボナッチ数列はGCDの中の人の逆走みたいな動きをする。
　へー、知らなかったー。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

dp = [0] * 42
dp[0] = dp[1] = 1
for i in range(2, 42):
    dp[i] = dp[i-1] + dp[i-2]

K = INT()

print(dp[K+1], dp[K])
