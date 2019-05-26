# -*- coding: utf-8 -*-

"""
・300点自力AC
・imos
"""

import sys
from itertools import accumulate

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

N,M=MAP()

imos=[0]*(N+1)
for i in range(M):
    l,r=MAP()
    imos[l-1]+=1
    imos[r]-=1

acc=list(accumulate(imos))

mx=max(acc)
if mx==M:
    print(acc.count(mx))
else:
    print(0)
