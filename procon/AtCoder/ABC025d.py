# -*- coding: utf-8 -*-

"""
・メモ化再帰
・何かメモうまくできてないのか、小さいケースでもTLE…。
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

N=len(unused)
memo={}
filled=set()
done=set()

def check(grid):
    for i in range(5):
        for j in range(1, 4):
            a=grid[i][j-1]
            b=grid[i][j]
            c=grid[i][j+1]
            if a!=0 and b!=0 and c!=0 and (a<b<c or a>b>c):
                return False
    for j in range(5):
        for i in range(1, 4):
            a=grid[i-1][j]
            b=grid[i][j]
            c=grid[i+1][j]
            if a!=0 and b!=0 and c!=0 and (a<b<c or a>b>c):
                return False
    return True

def get_key(grid):
    key=''
    for h, w in HW:
        key+=str(grid[h][w]).zfill(2)
    return key

def rec(cur, filled, done, grid):
    key=get_key(grid)
    if key in memo:
        return 0
    memo[key]=1
    if not check(grid):
        return 0
    if cur==N:
        return 1
    res=0
    for i, hw in enumerate(HW):
        h,w=hw
        if (h, w) in filled:
            continue  
        filled.add((h, w))
        for x in unused:
            if x in done:
                continue
            grid[h][w]=x
            done.add(x)
            res+=rec(cur+1, filled.copy(), done.copy(), deepcopy(grid))
            done.remove(x)
        grid[h][w]=0
        filled.remove((h, w))
    return res

print(rec(0, filled, done, grid))
