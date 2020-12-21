# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・全列挙の中で1次元DP。遷移は2つ前からある。
"""

import sys
from itertools import product, combinations

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

prod = list(product(['A', 'B', 'X', 'Y'], repeat=2))
ans = INF
for comb in combinations(prod, 2):
    L = ''.join(comb[0])
    R = ''.join(comb[1])
    dp = [INF] * (N+1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2, N+1):
        # 普通に押す
        dp[i] = min(dp[i], dp[i-1] + 1)
        # 短縮が使えるなら使ってみる
        if S[i-2:i] == L or S[i-2:i] == R:
            dp[i] = min(dp[i], dp[i-2] + 1)
    ans = min(ans, dp[N])
print(ans)
