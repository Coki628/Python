# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc025
・部分点解法
・並びを全部試す。
　空き数をNとして、計算量は順列のN!*チェックに30くらい。
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

def deepcopy(li): return [x[:] for x in li]

grid=[None]*5
for i in range(5):
    grid[i]=LIST()

used=[False]*26
HW=[]
for i in range(5):
    for j in range(5):
        used[grid[i][j]]=True
        if grid[i][j]==0:
            HW.append((i, j))

unused=[]
for i in range(1, 26):
    if not used[i]:
        unused.append(i)

def check(grid):
    for i in range(5):
        for j in range(1, 4):
            a=grid[i][j-1]
            b=grid[i][j]
            c=grid[i][j+1]
            if a<b<c or a>b>c:
                return False
    for j in range(5):
        for i in range(1, 4):
            a=grid[i-1][j]
            b=grid[i][j]
            c=grid[i+1][j]
            if a<b<c or a>b>c:
                return False
    return True

N=len(unused)
cnt=0
for perm in permutations(unused):
    grid2=deepcopy(grid)
    for i in range(N):
        h,w=HW[i]
        grid2[h][w]=perm[i]
    if check(grid2):
        cnt+=1
print(cnt%MOD)
