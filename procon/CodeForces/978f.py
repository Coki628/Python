# -*- coding: utf-8 -*-

"""
・二分探索
・グラフと見せかけての本質はにぶたん。
・それぞれ自分が嫌いなやつは全部持っておいてソートしておく。
・全体でにぶたんした時の該当する数から、嫌いなやつ内でにぶたんした時の該当する数を引けば答え。
"""

import sys
from bisect import bisect_left

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

N, M = MAP()
A = LIST()
hates = [[] for i in range(N)]
for _ in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    hates[a].append(A[b])
    hates[b].append(A[a])

A2 = sorted(A)
for i in range(N):
    hates[i].sort()

ans = [0] * N
for i, a in enumerate(A):
    cnt1 = bisect_left(A2, a)
    cnt2 = bisect_left(hates[i], a)
    ans[i] = cnt1 - cnt2
print(*ans)
