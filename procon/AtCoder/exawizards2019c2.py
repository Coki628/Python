# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

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

N,Q=MAP()
S=input()

D=defaultdict(lambda: [])
golems=[1]*N

for i in range(N):
    D[S[i]].append(i)

for i in range(Q):
    t,d=input().split()
    if d=='L':
        for j in D[t]:
            if j!=0:
                golems[j-1]+=golems[j]
            golems[j]=0   
    elif d=='R':
        for j in reversed(D[t]):
            if j!=N-1:
                golems[j+1]+=golems[j]
            golems[j]=0

print(sum(golems))
