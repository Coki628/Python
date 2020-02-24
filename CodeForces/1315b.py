# -*- coding: utf-8 -*-

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    a, b, p = MAP()
    S = input()
    N = len(S)

    nxt = [0] * N
    tmp = N - 1
    nxt[N-1] = tmp
    for i in range(N-2, -1, -1):
        if S[i] == S[i+1]:
            nxt[i] = tmp
        else:
            tmp = i + 1
            nxt[i] = tmp
    
    dp = [INF] * N
    dp[0] = 0
    for i in range(N-1):
        if S[i] == 'A':
            dp[nxt[i]] = min(dp[nxt[i]], dp[i] + a)
        else:
            dp[nxt[i]] = min(dp[nxt[i]], dp[i] + b)

    ans = N
    total = dp[N-1]
    for i in range(N-1, -1, -1):
        if dp[i] == INF:
            continue
        cost = total - dp[i]
        if cost <= p:
            ans = i + 1
        else:
            break
    print(ans)
