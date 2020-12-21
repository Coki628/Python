# -*- coding: utf-8 -*-

"""
・愚直解、TLE想定
"""

import sys
from itertools import permutations

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

W,H=MAP()
N=INT()
XY=[]
for i in range(N):
    x,y=MAP()
    XY.append((x-1, y-1))

ans=0
for perm in permutations(XY):
    cnt=0
    visited=list2d(H, W, False)
    for x, y in perm:
        cnt+=1
        visited[y][x]=True
        curx=x-1
        while curx>=0 and not visited[y][curx]:
            visited[y][curx]=True
            curx-=1
            cnt+=1
        curx=x+1
        while curx<W and not visited[y][curx]:
            visited[y][curx]=True
            curx+=1
            cnt+=1
        cury=y-1
        while cury>=0 and not visited[cury][x]:
            visited[cury][x]=True
            cury-=1
            cnt+=1
        cury=y+1
        while cury<H and not visited[cury][x]:
            visited[cury][x]=True
            cury+=1
            cnt+=1
    ans=max(ans, cnt)
print(ans)
