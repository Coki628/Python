# -*- coding: utf-8 -*-

"""
・自力、部分点のみ。
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

# 1371 2091
# 1372 2092
# 1373 2093

N, M = MAP()

if M != 1999:
    raise Exception

# 1371以降は720周期で繰り返し
dp = [0] * 2091
dp[1] = 1
for i in range(2, 2091):
    dp[i] = (dp[i-1]**2 + dp[i-2]**2) % M

if N <= 1370:
    print(dp[N-1])
else:
    N -= 1371
    mod = 2091 - 1371
    dp = dp[1371:]
    print(dp[(N-1)%mod])
