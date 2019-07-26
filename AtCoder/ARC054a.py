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

L,X,Y,S,D=MAP()

if S<=D:
    if -X+Y<=0: 
        ans=abs(D-S)/(X+Y)
    else:
        ans=min(abs(D-S)/(X+Y), (L-abs(D-S))/(-X+Y))
else:
    if -X+Y<=0: 
        ans=(L-abs(D-S))/(X+Y)
    else:
        ans=min((L-abs(D-S))/(X+Y), abs(D-S)/(-X+Y))
print(ans)
