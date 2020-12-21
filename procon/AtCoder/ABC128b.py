# -*- coding: utf-8 -*-

import sys
from operator import itemgetter

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
SP=[]
for i in range(N):
    s,p=input().split()
    SP.append((i, s, int(p)))

SP.sort(key=itemgetter(2), reverse=True)
SP.sort(key=itemgetter(1))

for i in range(N):
    print(SP[i][0]+1)
