# -*- coding: utf-8 -*-

"""
・300点自力AC
"""

import sys

def input(): return sys.stdin.readline().strip()
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

N=INT()

sm=idx=0
ans=set()
for i in range(1, N+1):
    sm+=i
    ans.add(i)
    if sm>=N:
        idx=i
        break

if sm>N:
    ans.remove(sm-N)
for a in ans:
    print(a)
