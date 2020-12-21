# -*- coding: utf-8 -*-

"""
・400点自力AC
・4方向から前計算
・400万のループ5回はpythonだとTLE、pypyでAC1.8秒のギリギリ。
　本番中最初からpypyで提出してよかった。
・ループの形変えたらかえって遅くなった…。
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
grid=[]
for i in range(H):
    grid.append(list(input()))

L=list2d(H, W, 0)
R=list2d(H, W, 0)
D=list2d(H, W, 0)
U=list2d(H, W, 0)

for i in range(H):
    cnt=0
    for j in range(W):
        if grid[i][j]=='#':
            cnt=0
        else:
            cnt+=1
        L[i][j]=cnt
    cnt=0
    for j in range(W-1, -1, -1):
        if grid[i][j]=='#':
            cnt=0
        else:
            cnt+=1
        R[i][j]=cnt

for i in range(W):
    cnt=0
    for j in range(H):
        if grid[j][i]=='#':
            cnt=0
        else:
            cnt+=1
        D[j][i]=cnt
    cnt=0
    for j in range(H-1, -1, -1):
        if grid[j][i]=='#':
            cnt=0
        else:
            cnt+=1
        U[j][i]=cnt

ans=0
for i in range(H):
    for j in range(W):
        if grid[i][j]!='#':
            ans=max(ans, L[i][j]+R[i][j]+D[i][j]+U[i][j]-3)
print(ans)
