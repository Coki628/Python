# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-14、自力ならず。。
・うまくジャンル毎に進められないかと思ったけどWA。
・まあ、解法見たら今までちょっとやってない感じのだったからしょうがない。
"""

import sys
from bisect import bisect_right

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
genre = [[] for i in range(10)]
for i in range(N):
    c, g = MAP()
    genre[g-1].append(c)

CG = []
idxs = [0]
for i in range(N):
    for g in range(10):
        if i < len(genre[g]):
            CG.append((genre[g][i], g))
    if idxs[-1] == len(CG):
        break
    else:
        idxs.append(len(CG))
idxs = idxs[:-1]

def calc(i):
    idx = bisect_right(idxs, i) - 1
    return idx * 2

dp = list2d(N+1, K+1, 0)
for i, (c, g) in enumerate(CG):
    for j in range(K+1):
        dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        if j < K:
            dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j]+c+calc(i))
print(dp[N][K])
