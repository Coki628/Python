# -*- coding: utf-8 -*-

"""
・300点自力AC
・ビット全探索
"""

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

N,M=MAP()
lights=[[False]*N for i in range(M)]
for i in range(M):
    K=LIST()
    for j in range(1, K[0]+1):
        lights[i][K[j]-1]=True
P=LIST()

ans=0
for i in range(1<<N):
    cnts=[0]*M
    for j in range(N):
        for k in range(M):
            if i>>j&1:
                if lights[k][j]:
                    cnts[k]+=1
    for j in range(M):
        if cnts[j]%2!=P[j]:
            break
    else:
        ans+=1
print(ans)
