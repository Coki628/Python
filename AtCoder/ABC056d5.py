# -*- coding: utf-8 -*-

"""
・3重ループをなくして5000^2のDPを1回だけにして高速化。
・でもこれはTLE。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

# 部分和DPを変形させて、その値になるのに使われた値の集合を持っておく
dp = defaultdict(set)
dp[0].add(0)
for i in range(N):
    for j in range(K-1, -1, -1):
        if len(dp[j]):
            dp[j+A[i]] |= dp[j]
            dp[j+A[i]].add(A[i])

ans = 0
for i in range(N):
    for k, s in dp.items():
        # A[i]減らしたらK未満になる状況で、一度でもA[i]が使われていれば必要
        if k >= K and k-A[i] < K and A[i] in s:
            break
    else:
        # 該当するどの集合でも、一度も使われていなければ不要
        ans += 1
print(ans)
