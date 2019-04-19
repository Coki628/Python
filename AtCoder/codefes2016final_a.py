# -*- coding: utf-8 -*-

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

H,W=MAP()
grid=[None]*H
for i in range(H):
    grid[i]=input().split()
for i in range(H):
    for j in range(W):
        if grid[i][j]=='snuke':
            print(chr(j+65)+str(i+1))
            exit()
