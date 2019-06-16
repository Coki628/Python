# -*- coding: utf-8 -*-

import sys

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

N,M,K=MAP()
A=LIST()

def rec(cur, gap, m):
    if m>M:
        return -1
    if gap>=K:
        return -1
    if cur==N:
        return 0
    ans=-1
    res=rec(cur+1, gap+1, m)
    if res!=-1:
        ans=max(ans, res)
    res=rec(cur+1, 0, m+1)
    if res!=-1:
        ans=max(ans, res+A[cur])
    return ans

res=rec(0, 0, 0)
print(res)
