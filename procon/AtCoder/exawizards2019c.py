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

D=defaultdict(lambda: [0, 0])
cur=defaultdict(int)

for i in range(Q):
    t,d=input().split()
    if d=='L':
        cur[t]-=1
        D[t][0]=min(D[t][0], cur[t])
    elif d=='R':
        cur[t]+=1
        D[t][1]=max(D[t][1], cur[t])

ans=N
for i in range(N):
    if i+D[S[i]][0]<0:
        ans-=1
    elif i+D[S[i]][1]>=N:
        ans-=1
print(ans)
