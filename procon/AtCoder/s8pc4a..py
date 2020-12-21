# -*- coding: utf-8 -*-

import sys
from bisect import bisect_left, bisect_right

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

N=INT()
L1=['']*N
L2=['']*N
for i in range(N):
    S=input()
    L1[i]=S.replace('?', 'a')
    L2[i]=S.replace('?', 'z')
L1.sort()
L2.sort()
T=input()

idx1=bisect_left(L2, T)
idx2=bisect_right(L1, T)
print(*list(range(idx1+1, idx2+2)))
