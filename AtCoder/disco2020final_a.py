# -*- coding: utf-8 -*-

"""
・自力AC
・Grundy数、Nim
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
A = LIST()

# dp[i] := 数値iからゲームを続けられる最大ターン数
dp = [0] * 101
for a in range(1, 101):
    nxt = a * 2
    while nxt <= 100:
        dp[nxt] = max(dp[nxt], dp[a] + 1)
        nxt += a

B = [0] * N
for i in range(N):
    B[i] = dp[A[i]]

# Nimする
ans = 0
for b in B:
    ans ^= b
if ans:
    Yes()
else:
    No()
