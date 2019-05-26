# -*- coding: utf-8 -*-

import sys
from collections import deque
from copy import copy

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

N,K=MAP()
V=deque(LIST())

memo=list3d(INF, K+1, N+1, N+1)
def rec(v, k, l, r):
    if memo[k][l][r]<=v:
        return memo[k][l][r]
    if k==K:
        return v
