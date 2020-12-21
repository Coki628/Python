# -*- coding: utf-8 -*-

"""
・上下左右から累積してこうとしたけど、
　斜めも考えるのが面倒過ぎることに気付いてやめた。
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

H,W=MAP()

grid=[None]*H
for i in range(H):
    grid[i]=list(input())

dist=list2d(H, W, INF)
for i in range(H):
    tmp=INF
    for j in range(W):
        if grid[i][j]=='#':
            tmp=0
        else:
            tmp+=1
        dist[i][j]=min(dist[i][j], tmp)
for i in range(H):
    tmp=INF
    for j in range(W-1, -1, -1):
        if grid[i][j]=='#':
            tmp=0
        else:
            tmp+=1
        dist[i][j]=min(dist[i][j], tmp)
for i in range(W):
    tmp=INF
    for j in range(H):
        if grid[j][i]=='#':
            tmp=0
        else:
            tmp+=1
        dist[j][i]=min(dist[j][i], tmp)
for i in range(W):
    tmp=INF
    for j in range(H-1, -1, -1):
        if grid[j][i]=='#':
            tmp=0
        else:
            tmp+=1
        dist[j][i]=min(dist[j][i], tmp)
print()
