# -*- coding: utf-8 -*-

"""
・これは自力TLE
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
S = input()

dp = [set() for i in range(N+1)]
dp[0].add('')
for i in range(N):
    for s in dp[i]:
        dp[i+1].add(s)
        dp[i+1].add(s+S[i])

li = sorted(dp[N], reverse=1)
if len(li) < K:
    print(-1)
    exit()
li2 = []
for s in li:
    li2.append(len(s))
li2.sort(reverse=1)
ans = 0
for ln in li2[:K]:
    ans += N - ln
print(ans)
